# Experimento de hiperparametros: retinanet-baixoTargetBox
# Unica alteracao em relacao a base: TARGET_BOX = 32
# --------------------------------------------------------------
# Ponto de entrada do experimento na Flautim.
#
# O job da plataforma executa literalmente "python3 run.py" dentro
# do diretorio do experimento, entao este arquivo TEM de se chamar
# run.py. Os demais modulos sao importados de forma plana, porque a
# plataforma copia os arquivos achatados nesse mesmo diretorio.
# --------------------------------------------------------------
import os

import flautim as fl

import RetinaDataset
import RetinaExperiment
import RetinaModel

log = RetinaDataset.log
log_erro = RetinaDataset.log_erro
etapa = RetinaDataset.etapa
RetinaError = RetinaDataset.RetinaError


# Caminho onde a plataforma monta o dataset dentro do pod. O cluster que
# pega o job muda a raiz em que o volume compartilhado aparece (k3s-prod
# usa /home, k8s-bcm usa /mnt) — RetinaDataset.resolve_dataset_path()
# tenta as duas antes de desistir, entao o valor abaixo so precisa ser
# uma delas.
DATASET_PATH = os.getenv(
    "RETINA_DATASET_PATH", "/home/flautim_workplace/datasets/db_tb/Dataset/"
)

# Estrategia de particao da tese (Exp. 5 / 8.1): em vez de encolher a
# lamina inteira para um quadrado (o que distorcia a proporcao e
# deixava os bacilos com poucos pixels), cada imagem e reescalada para
# o bacilo mediano ter ~RETINA_TARGET_BOX px e processada em ladrilhos
# de RETINA_TILE_SIZE px — recortes aleatorios no treino, grade com
# sobreposicao na avaliacao. Se a GPU ficar sem memoria, baixe
# RETINA_BATCH_SIZE (ou RETINA_TILE_SIZE).
TILE_SIZE = int(os.getenv("RETINA_TILE_SIZE", "640"))
TILE_OVERLAP = int(os.getenv("RETINA_TILE_OVERLAP", "128"))
CROPS_PER_IMAGE = int(os.getenv("RETINA_CROPS_PER_IMAGE", "8"))
TARGET_BOX = float(os.getenv("RETINA_TARGET_BOX", "32"))
SCALE = os.getenv("RETINA_SCALE")
SCALE = float(SCALE) if SCALE else None

# O pod da Flautim nao monta um volume para /dev/shm, entao ele fica com
# os 64 MB padrao do runtime. Os workers do DataLoader passam os lotes
# por memoria compartilhada e isso estoura: "DataLoader worker ...
# killed by signal: Bus error" (retina4log.md de 24/09/2026). Com 0
# workers o carregamento acontece no processo principal, nao usa
# /dev/shm e aproveita o cache de imagens em RAM do RetinaDataset.
NUM_WORKERS = int(os.getenv("RETINA_NUM_WORKERS", "0"))

# Sem isto o Dataset base da Flautim assume batch_size=128 — foi o que
# derrubou o run de 24/09/2026 com CUDA out of memory.
BATCH_SIZE = int(os.getenv("RETINA_BATCH_SIZE", "4"))

# "retinanet" (padrao) ou "fasterrcnn" (o metodo do Exp. 8.1 da tese).
ARCHITECTURE = os.getenv("RETINA_ARCH", "retinanet")
TRAINABLE_LAYERS = int(os.getenv("RETINA_TRAINABLE_LAYERS", "5"))

EPOCHS = int(os.getenv("RETINA_EPOCHS", "30"))
LEARNING_RATE = float(os.getenv("RETINA_LR", "5e-3"))
MOMENTUM = float(os.getenv("RETINA_MOMENTUM", "0.9"))
WEIGHT_DECAY = float(os.getenv("RETINA_WEIGHT_DECAY", "1e-4"))
WARMUP_STEPS = int(os.getenv("RETINA_WARMUP_STEPS", "300"))
GRAD_CLIP = float(os.getenv("RETINA_GRAD_CLIP", "10.0"))

# Protocolo de avaliacao da tese / notebook do Thales: IoU 0,1 e
# limiar de score 0,5. O AP50 (IoU 0,5) continua no log.
SCORE_THRESHOLD = float(os.getenv("RETINA_SCORE_THRESHOLD", "0.5"))
EVAL_IOU = float(os.getenv("RETINA_EVAL_IOU", "0.1"))

# NMS usado para juntar as deteccoes dos ladrilhos de uma imagem.
MERGE_NMS = float(os.getenv("RETINA_MERGE_NMS", "0.4"))

# Limite opcional de imagens, util para um teste rapido de fumaca.
MAX_IMAGES = os.getenv("RETINA_MAX_IMAGES")
MAX_IMAGES = int(MAX_IMAGES) if MAX_IMAGES else None


if __name__ == "__main__":
    try:
        context = fl.init()
    except Exception:
        log_erro("fl.init() falhou — sem contexto nao ha log na plataforma")
        raise SystemExit(1)

    log("rtn/run > experimento de hiperparametros: retinanet-baixoTargetBox")
    log(
        "rtn/run > flautim inicializado",
        details=(
            f"experiment_id={context.experiment.id} "
            f"path={context.filesystem.path} "
            f"output_path={context.filesystem.output_path}"
        ),
    )

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
        # Falha ja diagnosticada (caminho errado, CSV invalido, etc): a
        # mensagem ja basta, sem precisar de traceback.
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

    # O checkpoint da melhor epoca vai para o output_path, que e o
    # diretorio que a plataforma recolhe no fim do run.
    checkpoint = os.path.join(
        context.filesystem.output_path, "retinanet_bacilos.pt"
    )

    try:
        with etapa("construcao do Experimento", object="experiment"):
            experiment = RetinaExperiment.RetinaExperiment(
                model,
                dataset,
                context,
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

    # O Experiment.run() da plataforma conduz as epocas e engole as
    # excecoes do treino, registrando apenas o status "error" — por
    # isso o log detalhado sai de dentro dos loops do experimento.
    with etapa("treino (Experiment.run)", object="experiment"):
        experiment.run(metrics={})

    # A avaliacao no teste roda depois que o run ja foi marcado como
    # concluido; protegida aqui para que uma falha nela nao derrube o
    # status de um treino que terminou bem.
    try:
        metricas = experiment.evaluate_test_set()
        log(f"rtn/run > metricas no conjunto de teste: {metricas}", object="test")
    except Exception as erro:
        log_erro(
            "falha ao avaliar o conjunto de teste (o treino nao foi afetado)",
            erro,
        )

    log("rtn/run > execucao finalizada")
