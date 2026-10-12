# Experimento de hiperparametros: retinanet-wandb
# Treino unico com acompanhamento no Weights & Biases (W&B)
# --------------------------------------------------------------
# Ponto de entrada do experimento na Flautim.
#
# Usa os MESMOS RetinaDataset.py / RetinaModel.py / RetinaExperiment.py
# do retinanet-baseline, sem alteracao. O W&B entra por uma subclasse do
# RetinaExperiment definida aqui, que repassa ao W&B as metricas que os
# loops ja calculam (perdas, lr, metricas de validacao por epoca, teste).
#
# Configuracao: junta as duas variacoes que melhoraram o F1 no limiar
# 0,5 na primeira rodada (altoLearningRate e altoEpochs) e usa lote 8,
# aproveitando a VRAM que sobrava com lote 4.
#
# Sem chave do W&B (ou sem internet no pod) o run segue em modo
# offline: os arquivos ficam em <output_path>/wandb e podem ser enviados
# depois com "wandb sync <pasta offline-run-...>". O W&B nunca derruba o
# treino: qualquer falha dele vira um aviso no log.
# --------------------------------------------------------------
import os
import subprocess
import sys
import time

import flautim as fl
import torch

import RetinaDataset
import RetinaExperiment
import RetinaModel

log = RetinaDataset.log
log_erro = RetinaDataset.log_erro
etapa = RetinaDataset.etapa
RetinaError = RetinaDataset.RetinaError

NOME_EXPERIMENTO = "retinanet-wandb"

# Chave da API do W&B (https://wandb.ai/authorize). Se a plataforma nao
# deixar definir variaveis de ambiente, cole a chave aqui — e NAO faca
# commit dela.
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "retinanet-bacilos")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", NOME_EXPERIMENTO)

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
BATCH_SIZE = int(os.getenv("RETINA_BATCH_SIZE", "8"))
ARCHITECTURE = os.getenv("RETINA_ARCH", "retinanet")
TRAINABLE_LAYERS = int(os.getenv("RETINA_TRAINABLE_LAYERS", "5"))
EPOCHS = int(os.getenv("RETINA_EPOCHS", "40"))
LEARNING_RATE = float(os.getenv("RETINA_LR", "1e-2"))
MOMENTUM = float(os.getenv("RETINA_MOMENTUM", "0.9"))
WEIGHT_DECAY = float(os.getenv("RETINA_WEIGHT_DECAY", "1e-4"))
WARMUP_STEPS = int(os.getenv("RETINA_WARMUP_STEPS", "300"))
GRAD_CLIP = float(os.getenv("RETINA_GRAD_CLIP", "10.0"))
SCORE_THRESHOLD = float(os.getenv("RETINA_SCORE_THRESHOLD", "0.5"))
EVAL_IOU = float(os.getenv("RETINA_EVAL_IOU", "0.1"))
MERGE_NMS = float(os.getenv("RETINA_MERGE_NMS", "0.4"))
MAX_IMAGES = os.getenv("RETINA_MAX_IMAGES")
MAX_IMAGES = int(MAX_IMAGES) if MAX_IMAGES else None

CONFIG = {
    "tile_size": TILE_SIZE,
    "tile_overlap": TILE_OVERLAP,
    "crops_per_image": CROPS_PER_IMAGE,
    "target_box": TARGET_BOX,
    "scale": SCALE,
    "batch_size": BATCH_SIZE,
    "architecture": ARCHITECTURE,
    "trainable_layers": TRAINABLE_LAYERS,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "warmup_steps": WARMUP_STEPS,
    "grad_clip": GRAD_CLIP,
    "score_threshold": SCORE_THRESHOLD,
    "eval_iou": EVAL_IOU,
    "merge_nms": MERGE_NMS,
    "max_images": MAX_IMAGES,
}

# RetinaNet da tese (Thales, Tabelas 22-25), para o painel do W&B.
REFERENCIA_TESE = [
    ["dataset-1", 0.9660, 0.4071, 0.5723, 0.3971],
    ["dataset-2", 0.9078, 0.5002, 0.6447, 0.4707],
    ["dataset-3", 0.9285, 0.5314, 0.6748, 0.5040],
    ["dataset-4", 0.9332, 0.4575, 0.6059, 0.4415],
]


def iniciar_wandb(pasta):
    """Abre o run do W&B; devolve None se nao der (o treino segue sem ele)."""
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
            return None

    os.makedirs(pasta, exist_ok=True)
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

    for tentativa in dict.fromkeys([modo, "offline"]):
        try:
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=WANDB_RUN_NAME,
                config=CONFIG,
                dir=pasta,
                mode=tentativa,
                tags=["retinanet", "flautim"],
            )
            # Eixo x de todas as curvas = epoca, nao o passo interno do W&B.
            run.define_metric("epoca")
            run.define_metric("*", step_metric="epoca")
            log(
                f"rtn/wandb > run aberto em modo {tentativa}",
                details=f"url={getattr(run, 'url', None)} dir={pasta}",
            )
            return run
        except Exception as erro:
            log_erro(f"wandb.init em modo {tentativa} falhou", erro)
    return None


class RetinaExperimentWandb(RetinaExperiment.RetinaExperiment):
    """RetinaExperiment que tambem envia as metricas de cada epoca ao W&B."""

    def __init__(self, model, dataset, context, wandb_run=None, **kwargs):
        super().__init__(model, dataset, context, **kwargs)
        self.wandb_run = wandb_run
        self._t_epoca = None

    def _enviar(self, dados):
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log({"epoca": self.current_epoch, **dados})
        except Exception as erro:
            log(f"rtn/wandb > AVISO falha ao enviar metricas: {type(erro).__name__}: {erro}")

    def training_loop(self, data_loader):
        self._t_epoca = time.time()
        loss, metricas = super().training_loop(data_loader)
        dados = {"treino/loss": loss, "treino/lr": self.optimizer.param_groups[0]["lr"]}
        dados.update({f"treino/{k.lower()}": v for k, v in metricas.items()})
        if self.device.type == "cuda":
            dados["sistema/vram_pico_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        dados["sistema/min_treino"] = (time.time() - self._t_epoca) / 60
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
                import wandb

                linhas = [["este run (limiar 0,5)", metricas["PRECISION"], metricas["RECALL"], metricas["F1"], metricas["AP_TESE"]]]
                if "F1_VAL_THRESHOLD" in metricas:
                    linhas.append(
                        [
                            f"este run (limiar {metricas['VAL_THRESHOLD']:.2f} da validacao)",
                            metricas["PRECISION_VAL_THRESHOLD"],
                            metricas["RECALL_VAL_THRESHOLD"],
                            metricas["F1_VAL_THRESHOLD"],
                            metricas["AP_TESE_VAL_THRESHOLD"],
                        ]
                    )
                linhas += [[f"tese RetinaNet {d}", p, r, f, ap] for d, p, r, f, ap in REFERENCIA_TESE]
                self.wandb_run.log(
                    {
                        "teste/comparacao_tese": wandb.Table(
                            columns=["execucao", "precisao", "sensibilidade", "f1", "ap"],
                            data=linhas,
                        )
                    }
                )
            except Exception as erro:
                log(f"rtn/wandb > AVISO falha ao enviar o teste: {type(erro).__name__}: {erro}")
        return metricas


if __name__ == "__main__":
    try:
        context = fl.init()
    except Exception:
        log_erro("fl.init() falhou — sem contexto nao ha log na plataforma")
        raise SystemExit(1)

    log(f"rtn/run > experimento de hiperparametros: {NOME_EXPERIMENTO}")
    log(f"rtn/run > configuracao: {CONFIG}")
    log(
        "rtn/run > flautim inicializado",
        details=(
            f"experiment_id={context.experiment.id} "
            f"path={context.filesystem.path} "
            f"output_path={context.filesystem.output_path}"
        ),
    )

    with etapa("iniciar o W&B", object="wandb"):
        wandb_run = iniciar_wandb(os.path.join(context.filesystem.output_path, "wandb"))

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
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
    except RetinaError as erro:
        log(f"rtn/run > ERRO FATAL no dataset: {erro}", object="dataset")
        raise SystemExit(1)
    except Exception as erro:
        log_erro("falha inesperada ao construir o dataset", erro)
        raise SystemExit(1)

    try:
        with etapa("construcao do Modelo", object="model"):
            model = RetinaModel.RetinaModel(
                context,
                tile_size=TILE_SIZE,
                architecture=ARCHITECTURE,
                trainable_backbone_layers=TRAINABLE_LAYERS,
            )
    except Exception as erro:
        log_erro(
            "falha ao construir o modelo — se foi download dos pesos "
            "pre-treinados, o pod pode estar sem saida para a internet",
            erro,
        )
        raise SystemExit(1)

    checkpoint = os.path.join(context.filesystem.output_path, "retinanet_bacilos.pt")

    try:
        with etapa("construcao do Experimento", object="experiment"):
            experiment = RetinaExperimentWandb(
                model,
                dataset,
                context,
                wandb_run=wandb_run,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                warmup_steps=WARMUP_STEPS,
                grad_clip=GRAD_CLIP,
                merge_nms=MERGE_NMS,
                score_threshold=SCORE_THRESHOLD,
                eval_iou=EVAL_IOU,
                model_path=checkpoint,
            )
    except Exception as erro:
        log_erro("falha ao construir o experimento", erro)
        raise SystemExit(1)

    with etapa("treino (Experiment.run)", object="experiment"):
        experiment.run(metrics={})

    try:
        metricas = experiment.evaluate_test_set()
        log(f"rtn/run > metricas no conjunto de teste: {metricas}", object="test")
    except Exception as erro:
        log_erro("falha ao avaliar o conjunto de teste (o treino nao foi afetado)", erro)

    if wandb_run is not None:
        try:
            wandb_run.finish()
            log("rtn/wandb > run do W&B encerrado")
        except Exception as erro:
            log(f"rtn/wandb > AVISO falha ao encerrar: {type(erro).__name__}: {erro}")

    log("rtn/run > execucao finalizada")
