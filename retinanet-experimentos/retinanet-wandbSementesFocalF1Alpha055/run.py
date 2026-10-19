# Experimento de hiperparametros: retinanet-wandbSementesFocalF1Alpha055
# Configuracao do retinanet-optunaFocalF1 (trial 1) com focal_alpha 0,55 em
# vez de 0,6, DEZ sementes novas (47-56) e a ULTIMA EPOCA como criterio
# principal de checkpoint, no W&B
#
# Por que: o optunaFocalF1 e o de maior F1 do projeto (0,8673 numa semente;
# 0,8643 +- 0,0037 em 5 sementes no wandbSementesFocalF1), mas com a
# calibracao passada do ponto: precisao 0,846 contra sensibilidade 0,884 e
# limiar ideal na validacao em 0,60, nao em 0,5. O focal_alpha foi o
# hiperparametro mais importante da busca (0,33); os 5 melhores trials usaram
# 0,6, e o alpha 0,5 (wandbSementes) deu mais precisao e menos F1. Alpha 0,55
# fica entre os dois: o esperado e a precisao subir e o F1 ficar parecido ou
# um pouco maior. Todo o resto e identico ao trial 1.
#
# O experimento faz tres coisas ao mesmo tempo:
#   1. Confiabilidade: 10 sementes novas, com uma configuracao parecida com a
#      do wandbSementesFocalF1 (15 sementes no total da familia).
#   2. Alpha 0,55: o unico hiperparametro que muda.
#   3. Checkpoint pela ULTIMA epoca, declarada ANTES de rodar como criterio
#      principal. Nos dois experimentos de 12/10 (wandbSementesSuperaTese e
#      wandbSementesAlpha030) a ultima epoca teve o maior F1, e escolher a
#      epoca pelo maximo da validacao (20 laminas, ruidosa) premiou epocas de
#      sorte. Era um criterio secundario visto no teste; este experimento a
#      confirma (ou nao).
#
# Como separar os efeitos: o criterio ap_voc e o do wandbSementesFocalF1, e
# roda nos MESMOS treinos que a ultima. Ultima x ap_voc isola o criterio
# (pareado, mesmas sementes). Ap_voc aqui x wandbSementesFocalF1 isola o
# alpha (sementes diferentes, 10 x 5, entao a comparacao e mais fraca).
# --------------------------------------------------------------
# Ponto de entrada do experimento na Flautim.
#
# Cada semente treina UMA vez e guarda o melhor checkpoint de tres criterios,
# todos medidos SO na validacao (o teste nunca escolhe nada):
#   - ultima   (PRINCIPAL, declarado antes de rodar): a ultima epoca (30).
#   - ap_voc:  maior AP_VOC na validacao (criterio do wandbSementesFocalF1).
#   - f1_val:  maior F1 na validacao no limiar 0,5, o objetivo da busca.
# So o PRINCIPAL e avaliado pelo evaluate_test_set e entra nas tabelas e no
# painel (linhas "rtn/teste > limiar 0.50 (tese)"). Os outros dois vao para
# o log como analise secundaria (linhas "rtn/criterio >").
#
# Foco em F1, como no optunaFocalF1: nenhum criterio exige precisao acima da
# tese (ao contrario do wandbSementesSuperaTese/Alpha030). Os tres criterios
# sao de F1 ou AP, e as metricas esperadas sao as do optunaFocalF1: precisao
# ~0,83-0,86, sensibilidade ~0,86-0,91, F1 ~0,86-0,87. A linha "SUPERA AS 4
# DA TESE" do log sai NAO (a precisao fica abaixo de 0,934); e esperado, e
# nao e o objetivo aqui. Se preferir o criterio de F1 direto como principal,
# troque PRINCIPAL e SECUNDARIOS abaixo; os tres checkpoints ja sao guardados.
#
# Hipotese, para nao ler o resultado de forma enviesada: na curva de
# aprendizado a validacao deste modelo e bem mais estavel que a dos
# experimentos do alpha 0,25/0,30 (F1 de validacao plano em ~0,83 depois da
# epoca 5), entao a diferenca entre os criterios deve ser pequena aqui. Se
# for, o ganho do experimento esta nas 10 sementes e no alpha, e nao no
# criterio.
#
# Mantido igual ao wandbSementesFocalF1: split fixo (semente 42, mesmas 40
# laminas de teste), 30 epocas, limiar 0,5 e IoU 0,1. Sementes 47-56 (as
# 42-46 ja foram usadas).
#
# Duracao estimada: ~26 min por semente, ~4,3 h no total. As sementes podem
# ser trocadas por RETINA_SEEDS.
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

NOME_EXPERIMENTO = "retinanet-wandbSementesFocalF1Alpha055"

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
SEMENTES = [int(s) for s in os.getenv("RETINA_SEEDS", "47,48,49,50,51,52,53,54,55,56").split(",")]

# Melhor configuracao do optunaFocalF1 (trial 1), com focal_alpha 0,55 (no trial 1 e
# no wandbSementesFocalF1 era 0,6). E o unico valor que muda.
PARAMS = {
    "learning_rate": float(os.getenv("RETINA_LR", "0.0025210814666085596")),
    "momentum": float(os.getenv("RETINA_MOMENTUM", "0.9465500014486941")),
    "weight_decay": float(os.getenv("RETINA_WEIGHT_DECAY", "0.00027586986748117065")),
    "warmup_steps": int(os.getenv("RETINA_WARMUP_STEPS", "300")),
    "grad_clip": float(os.getenv("RETINA_GRAD_CLIP", "2.5")),
    "batch_size": int(os.getenv("RETINA_BATCH_SIZE", "8")),
    "trainable_layers": int(os.getenv("RETINA_TRAINABLE_LAYERS", "5")),
    "focal_alpha": float(os.getenv("RETINA_FOCAL_ALPHA", "0.55")),
    "focal_gamma": float(os.getenv("RETINA_FOCAL_GAMMA", "1.5")),
    "fg_iou": float(os.getenv("RETINA_FG_IOU", "0.5")),
    "nms_thresh": float(os.getenv("RETINA_NMS_THRESH", "0.3")),
    "merge_nms": float(os.getenv("RETINA_MERGE_NMS", "0.25")),
}
CONFIG = {
    "tile_size": TILE_SIZE, "tile_overlap": TILE_OVERLAP, "crops_per_image": CROPS_PER_IMAGE,
    "target_box": TARGET_BOX, "epochs": EPOCHS, "score_threshold": SCORE_THRESHOLD,
    "eval_iou": EVAL_IOU, "split_seed": SPLIT_SEED, **PARAMS,
}

# Alvo: media do RetinaNet da tese nos datasets 1-4 (limiar 0,5). Aqui so serve
# para o log dizer quais metricas passam; nenhum criterio de checkpoint o usa.
REF_TESE = {"PRECISION": 0.933875, "RECALL": 0.47405, "F1": 0.624425, "AP_TESE": 0.453325}
NOMES = {"PRECISION": "Precisao", "RECALL": "Sensibilidade", "F1": "F1", "AP_TESE": "AP"}
RESUMO = ["PRECISION", "RECALL", "F1", "AP_TESE", "AP_VOC", "AP50", "F1_IOU50"]

PRINCIPAL = "ultima"
SECUNDARIOS = ["ap_voc", "f1_val"]
CRITERIOS = [PRINCIPAL] + SECUNDARIOS

WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "retinanet-bacilos")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None


# ##################################################################
# Criterios de escolha do checkpoint (todos sobre a validacao)
# ##################################################################
def folgas_tese(metricas):
    return {k: float(metricas[k]) - ref for k, ref in REF_TESE.items()}


def valor_criterio(nome, metricas, epoca):
    if nome == "ultima":
        return float(epoca)
    if nome == "ap_voc":
        return float(metricas["AP_VOC"])
    if nome == "f1_val":
        return float(metricas["F1"])
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
                tags=["retinanet", "flautim", "sementes", "criterios", "ultima-epoca"], reinit=True,
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
# Experimento: um treino, tres checkpoints
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
