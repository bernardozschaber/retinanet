# Experimento de hiperparametros: retinanet-wandbSementes
# Melhor configuracao do Optuna (retinanet-optuna, trial 12) repetida com
# varias sementes, acompanhada no Weights & Biases (W&B)
# --------------------------------------------------------------
# Ponto de entrada do experimento na Flautim.
#
# O que mudou em relacao ao retinanet-wandb:
#   - hiperparametros: os do trial 12 do Optuna (focal_alpha 0,5,
#     fg_iou 0,4, lr 2,6e-3, grad_clip 5, merge_nms 0,3...) em vez de
#     lote 8 / lr 1e-2 / 40 epocas, que ficou abaixo do baseline;
#   - N sementes de treino (padrao 5) no mesmo split de teste, para
#     reportar media +- desvio, como a tese faz;
#   - um run do W&B por semente, todos no mesmo grupo: a interface do
#     W&B ja mostra a media e a faixa das curvas do grupo.
#
# Usa os MESMOS RetinaDataset.py (retinanet-dataset/), RetinaModel.py
# (retinanet-model/) e RetinaExperiment.py (copia nesta pasta), sem alteracao. Perda focal e matcher sao ajustados
# aqui, do mesmo jeito que no retinanet-optuna.
#
# Sem chave do W&B (ou sem internet no pod) os runs ficam em modo
# offline em <output_path>/wandb e podem ser enviados depois com
# "wandb sync". O W&B nunca derruba o treino.
# --------------------------------------------------------------
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from functools import partial

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

NOME_EXPERIMENTO = "retinanet-wandbSementes"

# Chave da API do W&B (https://wandb.ai/authorize). Se a plataforma nao
# deixar definir variaveis de ambiente, cole a chave aqui — e NAO faca
# commit dela.
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "retinanet-bacilos")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None

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

# A semente do split fica fixa: o teste e sempre o mesmo conjunto de 40
# laminas, e a variacao medida e so a do treino (pesos da cabeca, ordem
# e posicao dos recortes, augmentation).
SPLIT_SEED = int(os.getenv("RETINA_SPLIT_SEED", "42"))
SEMENTES = [int(s) for s in os.getenv("RETINA_SEEDS", "42,43,44,45,46").split(",")]

# Melhor configuracao da 1a busca (retinanet-optuna, trial 12).
PARAMS = {
    "learning_rate": float(os.getenv("RETINA_LR", "0.0026159960682001554")),
    "momentum": float(os.getenv("RETINA_MOMENTUM", "0.9185585731063599")),
    "weight_decay": float(os.getenv("RETINA_WEIGHT_DECAY", "0.0002902988766898052")),
    "warmup_steps": int(os.getenv("RETINA_WARMUP_STEPS", "300")),
    "grad_clip": float(os.getenv("RETINA_GRAD_CLIP", "5.0")),
    "batch_size": int(os.getenv("RETINA_BATCH_SIZE", "4")),
    "trainable_layers": int(os.getenv("RETINA_TRAINABLE_LAYERS", "5")),
    "focal_alpha": float(os.getenv("RETINA_FOCAL_ALPHA", "0.5")),
    "focal_gamma": float(os.getenv("RETINA_FOCAL_GAMMA", "2.0")),
    "fg_iou": float(os.getenv("RETINA_FG_IOU", "0.4")),
    "nms_thresh": float(os.getenv("RETINA_NMS_THRESH", "0.4")),
    "merge_nms": float(os.getenv("RETINA_MERGE_NMS", "0.3")),
}
CONFIG = {
    "tile_size": TILE_SIZE,
    "tile_overlap": TILE_OVERLAP,
    "crops_per_image": CROPS_PER_IMAGE,
    "target_box": TARGET_BOX,
    "epochs": EPOCHS,
    "score_threshold": SCORE_THRESHOLD,
    "eval_iou": EVAL_IOU,
    "split_seed": SPLIT_SEED,
    **PARAMS,
}

# Metricas resumidas em media +- desvio no fim.
RESUMO = ["PRECISION", "RECALL", "F1", "AP_TESE", "AP_VOC", "AP50", "F1_IOU50"]

# RetinaNet da tese (Thales, Tabelas 22-25), para o painel do W&B.
REFERENCIA_TESE = [
    ["dataset-1", 0.9660, 0.4071, 0.5723, 0.3971],
    ["dataset-2", 0.9078, 0.5002, 0.6447, 0.4707],
    ["dataset-3", 0.9285, 0.5314, 0.6748, 0.5040],
    ["dataset-4", 0.9332, 0.4575, 0.6059, 0.4415],
]


# ##################################################################
# W&B
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
    """Um run do W&B; devolve None se nao der (o treino segue sem ele)."""
    if wandb is None:
        return None
    for tentativa in dict.fromkeys([modo, "offline"]):
        try:
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                group=NOME_EXPERIMENTO,
                job_type=tipo,
                name=nome,
                config=config,
                dir=pasta,
                mode=tentativa,
                tags=["retinanet", "flautim", "sementes"],
                reinit=True,
            )
            run.define_metric("epoca")
            run.define_metric("*", step_metric="epoca")
            log(
                f"rtn/wandb > run '{nome}' aberto em modo {tentativa}",
                details=f"url={getattr(run, 'url', None)} dir={pasta}",
            )
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


class RetinaExperimentWandb(RetinaExperiment.RetinaExperiment):
    """RetinaExperiment que tambem envia as metricas de cada epoca ao W&B."""

    def __init__(self, model, dataset, context, wandb_run=None, **kwargs):
        super().__init__(model, dataset, context, **kwargs)
        self.wandb_run = wandb_run

    def _enviar(self, dados):
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log({"epoca": self.current_epoch, **dados})
        except Exception as erro:
            log(f"rtn/wandb > AVISO falha ao enviar metricas: {type(erro).__name__}: {erro}")

    def training_loop(self, data_loader):
        t0 = time.time()
        loss, metricas = super().training_loop(data_loader)
        dados = {"treino/loss": loss, "treino/lr": self.optimizer.param_groups[0]["lr"]}
        dados.update({f"treino/{k.lower()}": v for k, v in metricas.items()})
        if self.device.type == "cuda":
            dados["sistema/vram_pico_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        dados["sistema/min_treino"] = (time.time() - t0) / 60
        self._enviar(dados)
        return loss, metricas

    def validation_loop(self, data_loader):
        t0 = time.time()
        loss, metricas = super().validation_loop(data_loader)
        dados = {"validacao/loss": loss, "validacao/melhor_ap_voc": self.best_score}
        dados.update({f"validacao/{k.lower()}": v for k, v in metricas.items()})
        dados["sistema/min_validacao"] = (time.time() - t0) / 60
        self._enviar(dados)
        return loss, metricas

    def evaluate_test_set(self):
        metricas = super().evaluate_test_set()
        if self.wandb_run is not None:
            try:
                for chave, valor in metricas.items():
                    self.wandb_run.summary[f"teste/{chave.lower()}"] = valor
            except Exception as erro:
                log(f"rtn/wandb > AVISO falha ao enviar o teste: {type(erro).__name__}: {erro}")
        return metricas


# ##################################################################
# Ajustes no detector sem tocar no RetinaModel.py (iguais ao optuna)
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


def treinar_semente(context, dataset, semente, model_path, wandb_run, pelo_flautim):
    """Treina e avalia uma semente; devolve as metricas de teste.

    So a ultima semente passa pelo experiment.run() da Flautim, que marca
    o run como concluido na plataforma; as demais rodam o mesmo ciclo
    (treino + validacao por epoca, checkpoint da melhor AP) chamando os
    loops diretamente.
    """
    semear(dataset, semente)
    aplicar_perda_focal(PARAMS["focal_alpha"], PARAMS["focal_gamma"])
    dataset.batch_size = PARAMS["batch_size"]
    model = RetinaModel.RetinaModel(
        context,
        tile_size=TILE_SIZE,
        architecture="retinanet",
        trainable_backbone_layers=PARAMS["trainable_layers"],
        nms_thresh=PARAMS["nms_thresh"],
    )
    ajustar_matcher(model, PARAMS["fg_iou"])
    experiment = RetinaExperimentWandb(
        model,
        dataset,
        context,
        wandb_run=wandb_run,
        epochs=EPOCHS,
        learning_rate=PARAMS["learning_rate"],
        momentum=PARAMS["momentum"],
        weight_decay=PARAMS["weight_decay"],
        warmup_steps=PARAMS["warmup_steps"],
        grad_clip=PARAMS["grad_clip"],
        merge_nms=PARAMS["merge_nms"],
        score_threshold=SCORE_THRESHOLD,
        eval_iou=EVAL_IOU,
        model_path=model_path,
    )
    if pelo_flautim:
        experiment.run(metrics={})
    else:
        for _ in range(EPOCHS):
            experiment.training_loop(dataset.dataloader())
            experiment.validation_loop(dataset.dataloader(validation=True))
    metricas = experiment.evaluate_test_set()
    metricas["BEST_EPOCH"] = float(torch.load(model_path, map_location="cpu").get("epoca", -1))
    del model, experiment
    liberar_gpu()
    return metricas


if __name__ == "__main__":
    try:
        context = fl.init()
    except Exception:
        log_erro("fl.init() falhou — sem contexto nao ha log na plataforma")
        raise SystemExit(1)

    log(f"rtn/run > experimento de hiperparametros: {NOME_EXPERIMENTO}")
    log(f"rtn/run > configuracao: {CONFIG}")
    log(f"rtn/run > sementes de treino: {SEMENTES} (split fixo, semente {SPLIT_SEED})")
    log(
        "rtn/run > flautim inicializado",
        details=(
            f"experiment_id={context.experiment.id} "
            f"path={context.filesystem.path} "
            f"output_path={context.filesystem.output_path}"
        ),
    )

    pasta_saida = os.path.join(context.filesystem.output_path, "sementes")
    pasta_wandb = os.path.join(context.filesystem.output_path, "wandb")
    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(pasta_wandb, exist_ok=True)

    with etapa("importar o W&B", object="wandb"):
        wandb, modo = importar_wandb()

    try:
        with etapa("construcao do Dataset", object="dataset"):
            dataset = RetinaDataset.RetinaDataset(
                dataset_path=DATASET_PATH,
                tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP,
                crops_per_image=CROPS_PER_IMAGE,
                target_box_size=TARGET_BOX,
                scale=SCALE,
                max_images=MAX_IMAGES,
                batch_size=PARAMS["batch_size"],
                num_workers=NUM_WORKERS,
                seed=SPLIT_SEED,
            )
    except RetinaError as erro:
        log(f"rtn/run > ERRO FATAL no dataset: {erro}", object="dataset")
        raise SystemExit(1)
    except Exception as erro:
        log_erro("falha inesperada ao construir o dataset", erro)
        raise SystemExit(1)

    resultados = {}
    with tempfile.TemporaryDirectory(prefix="rtn_sementes_") as pasta_tmp:
        for numero, semente in enumerate(SEMENTES, 1):
            ultima = numero == len(SEMENTES)
            # O checkpoint da ultima semente fica no output_path, como nos
            # outros experimentos; os demais so existem durante o run.
            caminho = (
                os.path.join(context.filesystem.output_path, "retinanet_bacilos.pt")
                if ultima
                else os.path.join(pasta_tmp, f"semente_{semente}.pt")
            )
            run = abrir_run(
                wandb, modo, pasta_wandb, f"semente-{semente}",
                {**CONFIG, "seed": semente}, "semente",
            )
            t0 = time.time()
            try:
                with etapa(f"semente {semente} ({numero}/{len(SEMENTES)})", object="experiment"):
                    metricas = treinar_semente(context, dataset, semente, caminho, run, ultima)
                resultados[semente] = metricas
                log(
                    f"rtn/sementes > semente {semente}: "
                    + ", ".join(f"{k}={metricas[k]:.4f}" for k in RESUMO)
                    + f", melhor epoca={metricas['BEST_EPOCH']:.0f}, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    object="test",
                )
            except Exception as erro:
                log_erro(f"semente {semente} falhou; seguindo com as demais", erro)
                liberar_gpu()
            finally:
                fechar_run(run)
            with open(os.path.join(pasta_saida, "sementes.json"), "w") as arquivo:
                json.dump({"config": CONFIG, "resultados": resultados}, arquivo, indent=2)

    if not resultados:
        log("rtn/sementes > ERRO: nenhuma semente terminou")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Media +- desvio (desvio amostral, ddof=1) entre as sementes
    # ------------------------------------------------------------------
    resumo = {}
    for chave in RESUMO:
        valores = np.asarray([m[chave] for m in resultados.values()], dtype=np.float64)
        desvio = float(valores.std(ddof=1)) if len(valores) > 1 else 0.0
        resumo[chave] = {"media": float(valores.mean()), "desvio": desvio,
                         "min": float(valores.min()), "max": float(valores.max())}
    log(
        f"rtn/sementes > RESUMO ({len(resultados)} sementes, limiar {SCORE_THRESHOLD}): "
        + ", ".join(
            f"{k}={v['media']:.4f} (±{v['desvio']:.4f})" for k, v in resumo.items()
        ),
        object="test",
    )
    log(
        "rtn/teste > referencia RetinaNet da tese (Thales, media dos 4 datasets): "
        "P=0.9339, R=0.4741, F1=0.6244, AP=0.4533"
    )
    with open(os.path.join(pasta_saida, "resumo.json"), "w") as arquivo:
        json.dump({"config": CONFIG, "sementes": list(resultados), "resumo": resumo}, arquivo, indent=2)
    with open(os.path.join(pasta_saida, "sementes.csv"), "w") as arquivo:
        arquivo.write("semente," + ",".join(RESUMO) + ",BEST_EPOCH\n")
        for semente, m in resultados.items():
            arquivo.write(
                f"{semente}," + ",".join(f"{m[k]:.6f}" for k in RESUMO) + f",{m['BEST_EPOCH']:.0f}\n"
            )

    # Run de resumo no mesmo grupo: media/desvio e a tabela com a tese.
    run = abrir_run(wandb, modo, pasta_wandb, "resumo", CONFIG, "resumo")
    if run is not None:
        try:
            for chave, valores in resumo.items():
                for estatistica, valor in valores.items():
                    run.summary[f"teste/{chave.lower()}_{estatistica}"] = valor
            linhas = [[
                f"este trabalho ({len(resultados)} sementes)",
                resumo["PRECISION"]["media"], resumo["RECALL"]["media"],
                resumo["F1"]["media"], resumo["AP_TESE"]["media"],
            ]]
            linhas += [[f"tese RetinaNet {d}", p, r, f, ap] for d, p, r, f, ap in REFERENCIA_TESE]
            run.log({"teste/comparacao_tese": wandb.Table(
                columns=["execucao", "precisao", "sensibilidade", "f1", "ap"], data=linhas)})
        except Exception as erro:
            log(f"rtn/wandb > AVISO falha ao enviar o resumo: {type(erro).__name__}: {erro}")
        fechar_run(run)

    log("rtn/run > execucao finalizada")
