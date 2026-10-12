# Experimento de hiperparametros: retinanet-wandbSementesSuperaTese
# Melhor configuracao do optunaSuperaTese (trial 6, alpha 0,25) com cinco
# sementes e QUATRO criterios de escolha do checkpoint, acompanhado no W&B
# --------------------------------------------------------------
# Ponto de entrada do experimento na Flautim.
#
# Pergunta: o optunaSuperaTese superou a tese nas 4 metricas em 3 de 3
# sementes, mas com F1 0,68. A escolha do checkpoint exigia precisao de
# pelo menos 0,944 na validacao (tese + folga 0,01), onde a precisao sai
# menor que no teste, e ficou com epocas conservadoras. Uma folga menor
# mantem as 4 metricas acima da tese com F1 maior?
#
# Para responder sem treinar uma vez por regra, cada semente treina UMA vez
# e guarda o melhor checkpoint de quatro criterios, todos medidos SO na
# validacao (o teste nunca escolhe nada):
#   - sem_folga  (PRINCIPAL, declarado antes de rodar): a epoca tem de
#                superar a tese nas 4 metricas da validacao; entre essas,
#                a de maior F1. Se nenhuma passa, a que chega mais perto.
#   - folga_001: o mesmo com folga de 0,01 (o criterio do optunaSuperaTese).
#   - ap_voc:    maior AP_VOC na validacao (o criterio da 1a rodada).
#   - ultima:    a ultima epoca.
# So o PRINCIPAL e avaliado pelo evaluate_test_set e entra nas tabelas e no
# painel (linhas "rtn/teste > limiar 0.50 (tese)"). Os outros tres vao para
# o log como analise secundaria (linhas "rtn/criterio >").
#
# Mantido igual ao optunaSuperaTese: hiperparametros do trial 6, split fixo
# (semente 42, mesmas 40 laminas de teste), 30 epocas, limiar 0,5 e IoU 0,1.
# Sementes 42-46, como os outros experimentos com sementes.
#
# Usa os MESMOS RetinaDataset.py, RetinaModel.py e RetinaExperiment.py da
# raiz do projeto, sem alteracao.
# --------------------------------------------------------------
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import flautim as fl
import numpy as np
import torch
import torchvision.models.detection.retinanet as tv_retinanet
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import sigmoid_focal_loss

import RetinaDataset
import RetinaExperiment
import RetinaModel

log = RetinaDataset.log
log_erro = RetinaDataset.log_erro
etapa = RetinaDataset.etapa
RetinaError = RetinaDataset.RetinaError

NOME_EXPERIMENTO = "retinanet-wandbSementesSuperaTese"

DATASET_PATH = os.getenv(
    "RETINA_DATASET_PATH", "/home/flautim_workplace/datasets/db_tb/Dataset/"
)
TILE_SIZE = int(os.getenv("RETINA_TILE_SIZE", "640"))
TILE_OVERLAP = int(os.getenv("RETINA_TILE_OVERLAP", "128"))
CROPS_PER_IMAGE = int(os.getenv("RETINA_CROPS_PER_IMAGE", "8"))
TARGET_BOX = float(os.getenv("RETINA_TARGET_BOX", "40"))
SCALE = os.getenv("RETINA_SCALE")
SCALE = float(SCALE) if SCALE else None
NUM_WORKERS = int(os.getenv("RETINA_NUM_WORKERS", "0"))
SCORE_THRESHOLD = float(os.getenv("RETINA_SCORE_THRESHOLD", "0.5"))
EVAL_IOU = float(os.getenv("RETINA_EVAL_IOU", "0.1"))
EPOCHS = int(os.getenv("RETINA_EPOCHS", "30"))
MAX_IMAGES = os.getenv("RETINA_MAX_IMAGES")
MAX_IMAGES = int(MAX_IMAGES) if MAX_IMAGES else None
SPLIT_SEED = int(os.getenv("RETINA_SPLIT_SEED", "42"))
SEMENTES = [int(s) for s in os.getenv("RETINA_SEEDS", "42,43,44,45,46").split(",")]

# Melhor configuracao do optunaSuperaTese (trial 6). focal_alpha e o unico
# valor que muda entre este experimento (0,25) e o retinanet-wandbSementesAlpha030.
PARAMS = {
    "learning_rate": float(os.getenv("RETINA_LR", "0.010784588548197664")),
    "momentum": float(os.getenv("RETINA_MOMENTUM", "0.8856673220564906")),
    "weight_decay": float(os.getenv("RETINA_WEIGHT_DECAY", "0.00011777176017312901")),
    "warmup_steps": int(os.getenv("RETINA_WARMUP_STEPS", "300")),
    "grad_clip": float(os.getenv("RETINA_GRAD_CLIP", "10.0")),
    "batch_size": int(os.getenv("RETINA_BATCH_SIZE", "4")),
    "trainable_layers": int(os.getenv("RETINA_TRAINABLE_LAYERS", "5")),
    "focal_alpha": float(os.getenv("RETINA_FOCAL_ALPHA", "0.25")),
    "focal_gamma": float(os.getenv("RETINA_FOCAL_GAMMA", "2.0")),
    "fg_iou": float(os.getenv("RETINA_FG_IOU", "0.5")),
    "nms_thresh": float(os.getenv("RETINA_NMS_THRESH", "0.4")),
    "merge_nms": float(os.getenv("RETINA_MERGE_NMS", "0.4")),
}
CONFIG = {
    "tile_size": TILE_SIZE, "tile_overlap": TILE_OVERLAP, "crops_per_image": CROPS_PER_IMAGE,
    "target_box": TARGET_BOX, "epochs": EPOCHS, "score_threshold": SCORE_THRESHOLD,
    "eval_iou": EVAL_IOU, "split_seed": SPLIT_SEED, **PARAMS,
}

# Alvo: media do RetinaNet da tese nos datasets 1-4 (limiar 0,5).
REF_TESE = {"PRECISION": 0.933875, "RECALL": 0.47405, "F1": 0.624425, "AP_TESE": 0.453325}
NOMES = {"PRECISION": "Precisao", "RECALL": "Sensibilidade", "F1": "F1", "AP_TESE": "AP"}
RESUMO = ["PRECISION", "RECALL", "F1", "AP_TESE", "AP_VOC", "AP50", "F1_IOU50"]

PRINCIPAL = "sem_folga"
SECUNDARIOS = ["folga_001", "ap_voc", "ultima"]
CRITERIOS = [PRINCIPAL] + SECUNDARIOS

WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "retinanet-bacilos")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None


# ##################################################################
# Criterios de escolha do checkpoint (todos sobre a validacao)
# ##################################################################
def folgas_tese(metricas):
    return {k: float(metricas[k]) - ref for k, ref in REF_TESE.items()}


def objetivo_tese(metricas, folga):
    """1 + F1 se as 4 metricas passam a tese + folga; senao, -(o que falta)."""
    falta = sum(max(0.0, folga - f) for f in folgas_tese(metricas).values())
    return 1.0 + float(metricas["F1"]) if falta == 0.0 else -falta


def valor_criterio(nome, metricas, epoca):
    if nome == "sem_folga":
        return objetivo_tese(metricas, 0.0)
    if nome == "folga_001":
        return objetivo_tese(metricas, 0.01)
    if nome == "ap_voc":
        return float(metricas["AP_VOC"])
    if nome == "ultima":
        return float(epoca)
    raise ValueError(nome)


def supera_todas(metricas):
    return all(float(metricas[k]) > ref for k, ref in REF_TESE.items())


def formatar_folgas(folgas):
    return ", ".join(f"{NOMES[k]} {v:+.4f}" for k, v in folgas.items())


# ##################################################################
# Ajustes no detector sem tocar no RetinaModel.py (iguais aos anteriores)
# ##################################################################
def aplicar_perda_focal(alpha, gamma):
    """Troca alpha/gamma da perda focal do RetinaNet do torchvision."""
    if not hasattr(tv_retinanet, "sigmoid_focal_loss"):
        raise RetinaError(
            "esta versao do torchvision nao expoe sigmoid_focal_loss no modulo "
            "retinanet; nao da para aplicar focal_alpha/focal_gamma"
        )
    tv_retinanet.sigmoid_focal_loss = partial(
        sigmoid_focal_loss, alpha=float(alpha), gamma=float(gamma)
    )


def ajustar_matcher(modelo, fg_iou):
    """IoU minimo para uma ancora virar positiva (bg = fg - 0,1)."""
    modelo.detector.proposal_matcher = det_utils.Matcher(
        float(fg_iou), float(fg_iou) - 0.1, allow_low_quality_matches=True
    )


def semear(dataset, semente):
    torch.manual_seed(semente)
    np.random.seed(semente)
    dataset.particao_treino.rng = np.random.default_rng(semente)


def liberar_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ##################################################################
# W&B (igual ao retinanet-wandbSementes)
# ##################################################################
def importar_wandb():
    try:
        import wandb
    except ImportError:
        try:
            log("rtn/wandb > wandb ausente; instalando")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "wandb", "numpy==1.26.4"]
            )
            import wandb
        except Exception as erro:
            log_erro("nao consegui instalar o wandb; seguindo sem ele", erro)
            return None, None
    modo = "online"
    if WANDB_API_KEY:
        try:
            wandb.login(key=WANDB_API_KEY, relogin=True, timeout=30)
        except Exception as erro:
            log_erro("login no W&B falhou; usando modo offline", erro)
            modo = "offline"
    else:
        log("rtn/wandb > WANDB_API_KEY vazia; usando modo offline")
        modo = "offline"
    return wandb, modo


def abrir_run(wandb, modo, pasta, nome, config, tipo):
    if wandb is None:
        return None
    for tentativa in dict.fromkeys([modo, "offline"]):
        try:
            run = wandb.init(
                project=WANDB_PROJECT, entity=WANDB_ENTITY, group=NOME_EXPERIMENTO,
                job_type=tipo, name=nome, config=config, dir=pasta, mode=tentativa,
                tags=["retinanet", "flautim", "sementes", "criterios"], reinit=True,
            )
            run.define_metric("epoca")
            run.define_metric("*", step_metric="epoca")
            log(f"rtn/wandb > run '{nome}' aberto em modo {tentativa}",
                details=f"url={getattr(run, 'url', None)} dir={pasta}")
            return run
        except Exception as erro:
            log_erro(f"wandb.init em modo {tentativa} falhou", erro)
    return None


def fechar_run(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as erro:
        log(f"rtn/wandb > AVISO falha ao encerrar: {type(erro).__name__}: {erro}")


# ##################################################################
# Experimento: um treino, quatro checkpoints
# ##################################################################
class RetinaExperimentCriterios(RetinaExperiment.RetinaExperiment):
    """Guarda o melhor checkpoint de cada criterio, todos medidos na validacao.

    O validation_loop do RetinaExperiment salva o checkpoint pela AP_VOC no
    model_path; aqui ele roda com model_path apontando para um arquivo
    descartavel, e cada criterio grava o proprio arquivo em `caminhos`.
    """

    def __init__(self, model, dataset, context, caminhos, descarte, wandb_run=None, **kwargs):
        super().__init__(model, dataset, context, **kwargs)
        self.caminhos = {nome: Path(c) for nome, c in caminhos.items()}
        self.descarte = Path(descarte)
        self.wandb_run = wandb_run
        self.melhor = {nome: -math.inf for nome in CRITERIOS}
        self.epoca_escolhida = {nome: None for nome in CRITERIOS}
        self.registros = {}
        self.curva_pr = None

    def _enviar(self, dados):
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log({"epoca": self.current_epoch, **dados})
        except Exception as erro:
            log(f"rtn/wandb > AVISO falha ao enviar metricas: {type(erro).__name__}: {erro}")

    def inferir_particao(self, particao, nome_particao):
        registros = super().inferir_particao(particao, nome_particao)
        self.registros[nome_particao] = registros
        return registros

    def training_loop(self, data_loader):
        t0 = time.time()
        loss, metricas = super().training_loop(data_loader)
        dados = {"treino/loss": loss, "treino/lr": self.optimizer.param_groups[0]["lr"]}
        dados.update({f"treino/{k.lower()}": v for k, v in metricas.items()})
        dados["sistema/min_treino"] = (time.time() - t0) / 60
        self._enviar(dados)
        return loss, metricas

    def validation_loop(self, data_loader):
        verdadeiro = self.model_path
        self.model_path = self.descarte
        try:
            loss, metricas = super().validation_loop(data_loader)
        finally:
            self.model_path = verdadeiro
        folgas = folgas_tese(metricas)
        melhoraram = []
        estado = None
        for nome in CRITERIOS:
            valor = valor_criterio(nome, metricas, self.current_epoch)
            if valor > self.melhor[nome]:
                self.melhor[nome] = valor
                self.epoca_escolhida[nome] = self.current_epoch
                if estado is None:
                    estado = {"modelo": self.model.state_dict(), "epoca": self.current_epoch,
                              "limiar": metricas.get("BEST_THRESHOLD", self.score_threshold),
                              "metricas_validacao": metricas}
                self.caminhos[nome].parent.mkdir(parents=True, exist_ok=True)
                torch.save(estado, self.caminhos[nome])
                melhoraram.append(nome)
        log(
            f"rtn/criterio > validacao epoca {self.current_epoch}: P={metricas['PRECISION']:.4f}, "
            f"R={metricas['RECALL']:.4f}, F1={metricas['F1']:.4f}, AP={metricas['AP_TESE']:.4f}; "
            f"folga sobre a tese: {formatar_folgas(folgas)}; checkpoint atualizado para: "
            f"{', '.join(melhoraram) or 'nenhum criterio'}"
        )
        dados = {f"validacao/{k.lower()}": v for k, v in metricas.items()}
        dados.update({f"validacao/folga_{k.lower()}": v for k, v in folgas.items()})
        dados["validacao/loss"] = loss
        self._enviar(dados)
        return loss, metricas

    def avaliar_secundario(self, nome):
        """Teste de um criterio secundario, sem passar pelo evaluate_test_set."""
        estado = torch.load(self.caminhos[nome], map_location=self.device)
        self.model.load_state_dict(estado["modelo"])
        self.model.to(self.device)
        registros = self.inferir_particao(self.dataset.test(), "teste")
        metricas = self.metricas(registros, self.score_threshold)
        metricas["BEST_EPOCH"] = float(estado.get("epoca", -1))
        return metricas

    def evaluate_test_set(self):
        # O evaluate_test_set do RetinaExperiment carrega o model_path; aqui
        # ele e o checkpoint do criterio PRINCIPAL.
        self.model_path = self.caminhos[PRINCIPAL]
        metricas = super().evaluate_test_set()
        registros = self.registros.get("teste", [])
        if registros:
            scores, acertos, total = self._juntar(registros, "acertos")
            pontos = []
            for limiar in np.round(np.arange(0.05, 0.96, 0.05), 2):
                p, r, f1 = RetinaExperiment.prf_no_limiar(scores, acertos, total, limiar)
                tp = int(acertos[scores >= limiar].sum())
                pred = int((scores >= limiar).sum())
                pontos.append({"limiar": float(limiar), "precisao": p, "sensibilidade": r, "f1": f1,
                               "tp": tp, "fp": pred - tp, "fn": total - tp})
            self.curva_pr = {"ground_truth": total, "pontos": pontos}
        if self.wandb_run is not None:
            try:
                for chave, valor in metricas.items():
                    self.wandb_run.summary[f"teste/{chave.lower()}"] = valor
                self.wandb_run.summary["teste/supera_todas_tese"] = supera_todas(metricas)
            except Exception as erro:
                log(f"rtn/wandb > AVISO falha ao enviar o teste: {type(erro).__name__}: {erro}")
        return metricas


def construir(context, dataset, caminhos, descarte, wandb_run):
    aplicar_perda_focal(PARAMS["focal_alpha"], PARAMS["focal_gamma"])
    dataset.batch_size = PARAMS["batch_size"]
    model = RetinaModel.RetinaModel(
        context, tile_size=TILE_SIZE, architecture="retinanet",
        trainable_backbone_layers=PARAMS["trainable_layers"],
        nms_thresh=PARAMS["nms_thresh"],
    )
    ajustar_matcher(model, PARAMS["fg_iou"])
    experiment = RetinaExperimentCriterios(
        model, dataset, context, caminhos=caminhos, descarte=descarte, wandb_run=wandb_run,
        epochs=EPOCHS,
        learning_rate=PARAMS["learning_rate"],
        momentum=PARAMS["momentum"],
        weight_decay=PARAMS["weight_decay"],
        warmup_steps=PARAMS["warmup_steps"],
        grad_clip=PARAMS["grad_clip"],
        merge_nms=PARAMS["merge_nms"],
        score_threshold=SCORE_THRESHOLD,
        eval_iou=EVAL_IOU,
        model_path=caminhos[PRINCIPAL],
    )
    return model, experiment


def linha(metricas):
    return ", ".join(f"{k}={metricas[k]:.4f}" for k in RESUMO)


def resumir(resultados):
    resumo = {}
    for chave in RESUMO:
        valores = np.asarray([m[chave] for m in resultados.values()], dtype=np.float64)
        resumo[chave] = {"media": float(valores.mean()),
                         "desvio": float(valores.std(ddof=1)) if len(valores) > 1 else 0.0,
                         "min": float(valores.min()), "max": float(valores.max())}
    return resumo


if __name__ == "__main__":
    try:
        context = fl.init()
    except Exception:
        log_erro("fl.init() falhou — sem contexto nao ha log na plataforma")
        raise SystemExit(1)

    log(f"rtn/run > experimento de hiperparametros: {NOME_EXPERIMENTO}")
    log(f"rtn/run > configuracao: {CONFIG}")
    log(f"rtn/run > sementes de treino: {SEMENTES} (split fixo, semente {SPLIT_SEED})")
    log(f"rtn/run > criterio PRINCIPAL de checkpoint: {PRINCIPAL}; secundarios: {SECUNDARIOS}")
    log("rtn/run > flautim inicializado", details=(
        f"experiment_id={context.experiment.id} path={context.filesystem.path} "
        f"output_path={context.filesystem.output_path}"))

    saida = context.filesystem.output_path
    pasta_sementes = os.path.join(saida, "sementes")
    pasta_wandb = os.path.join(saida, "wandb")
    os.makedirs(pasta_sementes, exist_ok=True)
    os.makedirs(pasta_wandb, exist_ok=True)

    with etapa("importar o W&B", object="wandb"):
        wandb, modo = importar_wandb()

    try:
        with etapa("construcao do Dataset", object="dataset"):
            dataset = RetinaDataset.RetinaDataset(
                dataset_path=DATASET_PATH, tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP,
                crops_per_image=CROPS_PER_IMAGE, target_box_size=TARGET_BOX, scale=SCALE,
                max_images=MAX_IMAGES, batch_size=PARAMS["batch_size"], num_workers=NUM_WORKERS,
                seed=SPLIT_SEED,
            )
    except RetinaError as erro:
        log(f"rtn/run > ERRO FATAL no dataset: {erro}", object="dataset")
        raise SystemExit(1)
    except Exception as erro:
        log_erro("falha inesperada ao construir o dataset", erro)
        raise SystemExit(1)

    resultados = {nome: {} for nome in CRITERIOS}
    curvas = {}
    with tempfile.TemporaryDirectory(prefix="rtn_criterios_") as pasta_tmp:
        for numero, semente in enumerate(SEMENTES, 1):
            ultima = numero == len(SEMENTES)
            caminhos = {nome: os.path.join(pasta_tmp, f"semente_{semente}_{nome}.pt") for nome in CRITERIOS}
            if ultima:
                # O checkpoint principal da ultima semente fica no output_path.
                caminhos[PRINCIPAL] = os.path.join(saida, "retinanet_bacilos.pt")
            run = abrir_run(wandb, modo, pasta_wandb, f"semente-{semente}", {**CONFIG, "seed": semente}, "semente")
            t0 = time.time()
            model = experiment = None
            try:
                with etapa(f"semente {semente} ({numero}/{len(SEMENTES)})", object="experiment"):
                    semear(dataset, semente)
                    model, experiment = construir(context, dataset, caminhos,
                                                  os.path.join(pasta_tmp, "descarte.pt"), run)
                    # So a ultima semente passa pelo experiment.run(), que marca
                    # o run como concluido na plataforma.
                    if ultima:
                        experiment.run(metrics={})
                    else:
                        for _ in range(EPOCHS):
                            experiment.training_loop(dataset.dataloader())
                            experiment.validation_loop(dataset.dataloader(validation=True))
                    # Secundarios primeiro; o principal por ultimo, pelo
                    # evaluate_test_set, que deixa o modelo no checkpoint principal.
                    for nome in SECUNDARIOS:
                        m = experiment.avaliar_secundario(nome)
                        resultados[nome][semente] = m
                        log(f"rtn/criterio > semente {semente} criterio={nome} epoca={m['BEST_EPOCH']:.0f}: "
                            f"{linha(m)}; SUPERA AS 4 DA TESE: {'SIM' if supera_todas(m) else 'NAO'} "
                            f"({formatar_folgas(folgas_tese(m))})")
                    m = experiment.evaluate_test_set()
                    m["BEST_EPOCH"] = float(experiment.epoca_escolhida[PRINCIPAL] or -1)
                resultados[PRINCIPAL][semente] = m
                curvas[semente] = experiment.curva_pr
                log(f"rtn/criterio > semente {semente} criterio={PRINCIPAL} (principal) epoca={m['BEST_EPOCH']:.0f}: "
                    f"{linha(m)}; SUPERA AS 4 DA TESE: {'SIM' if supera_todas(m) else 'NAO'} "
                    f"({formatar_folgas(folgas_tese(m))})")
                log(f"rtn/sementes > semente {semente}: {linha(m)}, melhor epoca={m['BEST_EPOCH']:.0f}, "
                    f"{(time.time() - t0) / 60:.1f} min; SUPERA AS 4 DA TESE: {'SIM' if supera_todas(m) else 'NAO'} "
                    f"({formatar_folgas(folgas_tese(m))})", object="test")
                log(f"rtn/curva_pr > semente {semente}: {json.dumps(experiment.curva_pr)}")
            except Exception as erro:
                log_erro(f"semente {semente} falhou; seguindo com as demais", erro)
            finally:
                fechar_run(run)
                del model, experiment
                liberar_gpu()
            with open(os.path.join(pasta_sementes, "criterios.json"), "w") as arquivo:
                json.dump({"config": CONFIG, "principal": PRINCIPAL, "resultados": resultados}, arquivo, indent=2)
            with open(os.path.join(pasta_sementes, "curvas_pr.json"), "w") as arquivo:
                json.dump(curvas, arquivo, indent=2)

    if not resultados[PRINCIPAL]:
        log("rtn/sementes > ERRO: nenhuma semente terminou")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Resumo: o principal no formato dos outros experimentos com sementes,
    # e uma linha por criterio para a analise secundaria.
    # ------------------------------------------------------------------
    resumos = {}
    for nome in CRITERIOS:
        if not resultados[nome]:
            continue
        resumos[nome] = resumir(resultados[nome])
        vencedoras = [s for s, m in resultados[nome].items() if supera_todas(m)]
        media = {k: v["media"] for k, v in resumos[nome].items()}
        log(
            f"rtn/criterio > RESUMO criterio={nome}{' (principal)' if nome == PRINCIPAL else ''}: "
            + ", ".join(f"{k}={v['media']:.4f} (±{v['desvio']:.4f})" for k, v in resumos[nome].items())
            + f"; sementes que superam as 4: {len(vencedoras)}/{len(resultados[nome])}; "
            f"a media supera as 4: {'SIM' if supera_todas(media) else 'NAO'}"
        )
    principal = resumos[PRINCIPAL]
    vencedoras = [s for s, m in resultados[PRINCIPAL].items() if supera_todas(m)]
    media = {k: v["media"] for k, v in principal.items()}
    log(
        f"rtn/sementes > RESUMO ({len(resultados[PRINCIPAL])} sementes, limiar {SCORE_THRESHOLD}): "
        + ", ".join(f"{k}={v['media']:.4f} (±{v['desvio']:.4f})" for k, v in principal.items()),
        object="test",
    )
    log(
        f"rtn/tese > sementes que superam as 4 metricas da tese: {len(vencedoras)}/{len(resultados[PRINCIPAL])} "
        f"{vencedoras}; a MEDIA supera as 4: {'SIM' if supera_todas(media) else 'NAO'} "
        f"({formatar_folgas(folgas_tese(media))})",
        object="test",
    )
    with open(os.path.join(pasta_sementes, "resumo.json"), "w") as arquivo:
        json.dump({"config": CONFIG, "principal": PRINCIPAL, "resumos": resumos,
                   "alvo_tese": REF_TESE}, arquivo, indent=2)

    run = abrir_run(wandb, modo, pasta_wandb, "resumo", CONFIG, "resumo")
    if run is not None:
        try:
            for nome, resumo in resumos.items():
                for chave, valores in resumo.items():
                    for estatistica, valor in valores.items():
                        run.summary[f"{nome}/{chave.lower()}_{estatistica}"] = valor
        except Exception as erro:
            log(f"rtn/wandb > AVISO falha ao enviar o resumo: {type(erro).__name__}: {erro}")
        fechar_run(run)

    log("rtn/run > execucao finalizada")
