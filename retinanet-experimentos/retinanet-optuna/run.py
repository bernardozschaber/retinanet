# Experimento de hiperparametros: retinanet-optuna
# Busca automatica (Optuna, TPE + poda pela mediana) e retreino final
# --------------------------------------------------------------
# Ponto de entrada do experimento na Flautim.
#
# Usa os MESMOS RetinaDataset.py, RetinaModel.py e RetinaExperiment.py da
# raiz do projeto, sem alteracao. Tudo o que e novo (busca, perda
# focal configuravel, matcher configuravel) fica neste arquivo.
#
# Fluxo:
#   1. monta o dataset UMA vez (o cache de imagens em RAM e reaproveitado
#      por todos os trials);
#   2. Optuna roda N trials curtos (RETINA_TRIAL_EPOCHS epocas cada),
#      medindo o objetivo na validacao e podando os trials ruins cedo;
#   3. a melhor configuracao e retreinada do zero com RETINA_EPOCHS
#      epocas pelo ciclo normal da Flautim (experiment.run) e avaliada
#      no teste, exatamente como os demais experimentos.
#
# Os trials NAO passam pelo experiment.run(): ele marca o run como
# concluido na plataforma, entao so a etapa 3 o chama.
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

NOME_EXPERIMENTO = "retinanet-optuna"

DATASET_PATH = os.getenv(
    "RETINA_DATASET_PATH", "/home/flautim_workplace/datasets/db_tb/Dataset/"
)

# Parametros do dataset ficam FIXOS na busca: mudar a escala ou o lado
# do ladrilho obrigaria a reler e reescalar as 200 laminas a cada trial.
# Os valores sao os do baseline, que ficaram entre os melhores da
# primeira rodada (altoImageSize/baixoImageSize/alto/baixoTargetBox nao
# ganharam do 640/40).
TILE_SIZE = int(os.getenv("RETINA_TILE_SIZE", "640"))
TILE_OVERLAP = int(os.getenv("RETINA_TILE_OVERLAP", "128"))
CROPS_PER_IMAGE = int(os.getenv("RETINA_CROPS_PER_IMAGE", "8"))
TARGET_BOX = float(os.getenv("RETINA_TARGET_BOX", "40"))
SCALE = os.getenv("RETINA_SCALE")
SCALE = float(SCALE) if SCALE else None
NUM_WORKERS = int(os.getenv("RETINA_NUM_WORKERS", "0"))
SCORE_THRESHOLD = float(os.getenv("RETINA_SCORE_THRESHOLD", "0.5"))
EVAL_IOU = float(os.getenv("RETINA_EVAL_IOU", "0.1"))
MAX_IMAGES = os.getenv("RETINA_MAX_IMAGES")
MAX_IMAGES = int(MAX_IMAGES) if MAX_IMAGES else None
SEED = int(os.getenv("RETINA_SEED", "42"))

# Orcamento da busca. Cada trial e um treino curto; o retreino final usa
# RETINA_EPOCHS. A busca para no que vier primeiro: N trials ou timeout.
N_TRIALS = int(os.getenv("RETINA_OPTUNA_TRIALS", "16"))
TRIAL_EPOCHS = int(os.getenv("RETINA_TRIAL_EPOCHS", "12"))
TIMEOUT_HOURS = float(os.getenv("RETINA_OPTUNA_TIMEOUT_HOURS", "10"))
# Validar so a cada N epocas no trial: a inferencia em ladrilhos da
# validacao e cara, e a poda nao precisa de um ponto por epoca.
VAL_EVERY = max(1, int(os.getenv("RETINA_TRIAL_VAL_EVERY", "2")))
EPOCHS = int(os.getenv("RETINA_EPOCHS", "30"))
# 0 = so a busca, sem o retreino final (util para olhar os trials antes).
FINAL_RETRAIN = os.getenv("RETINA_FINAL_RETRAIN", "1") != "0"

# Metrica de validacao maximizada: F1 no limiar 0,5, o mesmo corte da
# comparacao com a tese, para a busca premiar configuracoes ja calibradas
# nele. (A 1a busca, de 10/10/2026, usou F1_BEST_THRESHOLD.) Qualquer
# chave do RetinaExperiment.metricas() serve (F1_BEST_THRESHOLD, AP_TESE,
# AP_VOC, AP50...).
OBJECTIVE = os.getenv("RETINA_OPTUNA_OBJECTIVE", "F1")

# Configuracoes ja medidas na primeira rodada, avaliadas primeiro para o
# TPE partir de pontos bons: o baseline e o altoLearningRate (o melhor
# no limiar 0,5).
PADRAO = {
    "learning_rate": 5e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "warmup_steps": 300,
    "grad_clip": 10.0,
    "batch_size": 4,
    "trainable_layers": 5,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "fg_iou": 0.5,
    "nms_thresh": 0.4,
    "merge_nms": 0.4,
}
CONFIGURACOES_INICIAIS = [PADRAO, {**PADRAO, "learning_rate": 1e-2}]


def garantir_pacote(modulo, requisito):
    """Importa `modulo`; se faltar, instala sem deixar o pip trocar o numpy.

    O requirements.txt ja instala o optuna. Isto so cobre o caso de a
    plataforma nao ter rodado o pip (o numpy fica travado na versao da
    imagem, ver requirements.txt do baseline).
    """
    try:
        return __import__(modulo)
    except ImportError:
        log(f"rtn/run > {modulo} ausente; instalando {requisito}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", requisito, "numpy==1.26.4"]
        )
        return __import__(modulo)


# ##################################################################
# Espaco de busca
# ##################################################################
def sugerir(trial):
    """Hiperparametros do trial.

    Faixas centradas no que a primeira rodada mostrou: lr maior ajudou
    (1e-2 > 5e-3 > 2,5e-3), wd 2e-4 nao ajudou, congelar camadas piorou.
    A perda focal e o IoU do matcher entram porque o problema principal
    no limiar 0,5 e calibracao: precisao ~0,95 e sensibilidade ~0,60,
    ou seja, scores baixos demais para bacilos corretos.
    """
    fg_iou = trial.suggest_categorical("fg_iou", [0.4, 0.5])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-3, 2e-2, log=True),
        "momentum": trial.suggest_float("momentum", 0.85, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 3e-5, 3e-4, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 600, step=100),
        "grad_clip": trial.suggest_categorical("grad_clip", [5.0, 10.0, 20.0]),
        # VRAM: lote 8 de ladrilhos 640 com AMP cabe na RTX 6000 (24 GB);
        # 8 tambem e o teto do RetinaDataset.
        "batch_size": trial.suggest_categorical("batch_size", [4, 8]),
        "trainable_layers": trial.suggest_int("trainable_layers", 3, 5),
        "focal_alpha": trial.suggest_categorical("focal_alpha", [0.25, 0.35, 0.5]),
        "focal_gamma": trial.suggest_categorical("focal_gamma", [1.0, 1.5, 2.0]),
        "fg_iou": fg_iou,
        "nms_thresh": trial.suggest_categorical("nms_thresh", [0.3, 0.4, 0.5]),
        "merge_nms": trial.suggest_categorical("merge_nms", [0.3, 0.4, 0.5]),
    }


# ##################################################################
# Ajustes no detector sem tocar no RetinaModel.py
# ##################################################################
def aplicar_perda_focal(alpha, gamma):
    """Troca alpha/gamma da perda focal do RetinaNet do torchvision.

    RetinaNetClassificationHead.compute_loss chama o nome global
    `sigmoid_focal_loss` do modulo retinanet com os padroes 0,25/2,0;
    substituir esse nome muda a perda de todos os modelos criados depois.
    """
    if not hasattr(tv_retinanet, "sigmoid_focal_loss"):
        raise RetinaError(
            "esta versao do torchvision nao expoe sigmoid_focal_loss no modulo "
            "retinanet; fixe focal_alpha=0.25 e focal_gamma=2.0"
        )
    tv_retinanet.sigmoid_focal_loss = partial(
        sigmoid_focal_loss, alpha=float(alpha), gamma=float(gamma)
    )


def ajustar_matcher(modelo, fg_iou):
    """IoU minimo para uma ancora virar positiva (padrao 0,5 / 0,4).

    Bacilo e fino: a caixa de uma ancora quadrada sobre um bastonete
    inclinado raramente passa de 0,5, entao baixar o limiar aumenta o
    numero de ancoras positivas por bacilo.
    """
    modelo.detector.proposal_matcher = det_utils.Matcher(
        float(fg_iou), float(fg_iou) - 0.1, allow_low_quality_matches=True
    )


def construir(context, dataset, params, epochs, model_path):
    """Modelo + experimento com os hiperparametros de `params`."""
    aplicar_perda_focal(params["focal_alpha"], params["focal_gamma"])
    dataset.batch_size = int(params["batch_size"])
    model = RetinaModel.RetinaModel(
        context,
        tile_size=TILE_SIZE,
        architecture="retinanet",
        trainable_backbone_layers=int(params["trainable_layers"]),
        nms_thresh=float(params["nms_thresh"]),
    )
    ajustar_matcher(model, params["fg_iou"])
    experiment = RetinaExperiment.RetinaExperiment(
        model,
        dataset,
        context,
        epochs=epochs,
        learning_rate=float(params["learning_rate"]),
        momentum=float(params["momentum"]),
        weight_decay=float(params["weight_decay"]),
        warmup_steps=int(params["warmup_steps"]),
        grad_clip=float(params["grad_clip"]),
        merge_nms=float(params["merge_nms"]),
        score_threshold=SCORE_THRESHOLD,
        eval_iou=EVAL_IOU,
        model_path=model_path,
    )
    return model, experiment


def liberar_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def semear(dataset, semente):
    torch.manual_seed(semente)
    np.random.seed(semente)
    dataset.particao_treino.rng = np.random.default_rng(semente)


# ##################################################################
# Busca
# ##################################################################
def criar_objetivo(optuna, context, dataset, pasta_tmp):
    def objetivo(trial):
        params = sugerir(trial)
        log(f"rtn/optuna > trial {trial.number} INICIO: {json.dumps(params)}")
        semear(dataset, SEED + trial.number)
        model = experiment = None
        caminho = os.path.join(pasta_tmp, f"trial_{trial.number}.pt")
        t0 = time.time()
        melhor = -1.0
        try:
            model, experiment = construir(context, dataset, params, TRIAL_EPOCHS, caminho)
            for epoca in range(1, TRIAL_EPOCHS + 1):
                t_epoca = time.time()
                loss, _ = experiment.training_loop(dataset.dataloader())
                if epoca % VAL_EVERY and epoca != TRIAL_EPOCHS:
                    continue
                # So a deteccao na validacao, sem a perda de validacao nem
                # checkpoint: o trial so precisa do valor do objetivo.
                registros = experiment.inferir_particao(dataset.validation(), "validacao")
                metricas = experiment.metricas(registros, SCORE_THRESHOLD)
                limiar, f1_limiar = experiment.melhor_limiar(registros)
                metricas["BEST_THRESHOLD"] = limiar
                metricas["F1_BEST_THRESHOLD"] = f1_limiar
                valor = float(metricas[OBJECTIVE])
                melhor = max(melhor, valor)
                trial.set_user_attr(f"epoca_{epoca}", {k: round(float(v), 4) for k, v in metricas.items()})
                log(
                    f"rtn/optuna > trial {trial.number} epoca {epoca}/{TRIAL_EPOCHS}: "
                    f"loss_treino={loss:.4f}, {OBJECTIVE}={valor:.4f} (melhor {melhor:.4f}), "
                    f"F1@{SCORE_THRESHOLD}={metricas['F1']:.4f}, AP_TESE={metricas['AP_TESE']:.4f}, "
                    f"AP_VOC={metricas['AP_VOC']:.4f}, limiar_val={limiar:.2f}, "
                    f"{time.time() - t_epoca:.0f}s"
                )
                trial.report(valor, epoca)
                if trial.should_prune():
                    log(f"rtn/optuna > trial {trial.number} PODADO na epoca {epoca}")
                    raise optuna.TrialPruned()
            return melhor
        except torch.cuda.OutOfMemoryError as erro:
            # Lote 8 com 5 camadas treinaveis e o pior caso; se estourar,
            # o trial e descartado e a busca segue.
            log_erro(f"trial {trial.number} sem memoria na GPU; podado", erro)
            raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as erro:
            log_erro(f"trial {trial.number} falhou", erro)
            raise
        finally:
            trial.set_user_attr("minutos", round((time.time() - t0) / 60, 1))
            if torch.cuda.is_available():
                trial.set_user_attr(
                    "pico_vram_mib", round(torch.cuda.max_memory_allocated() / 1024**2)
                )
            del model, experiment
            if os.path.exists(caminho):
                os.remove(caminho)
            liberar_gpu()

    return objetivo


def salvar_estudo(study, pasta):
    """Grava os trials a cada termino, para nao perder nada se o pod cair."""
    os.makedirs(pasta, exist_ok=True)
    try:
        study.trials_dataframe().to_csv(os.path.join(pasta, "trials.csv"), index=False)
    except Exception as erro:
        log_erro("nao consegui gravar trials.csv", erro)
    concluidos = [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
    if concluidos:
        with open(os.path.join(pasta, "best_params.json"), "w") as arquivo:
            json.dump(
                {
                    "trial": study.best_trial.number,
                    "objetivo": OBJECTIVE,
                    "valor": study.best_value,
                    "params": study.best_params,
                    "trial_epochs": TRIAL_EPOCHS,
                },
                arquivo,
                indent=2,
            )


def resumo_trial(study, trial, pasta):
    concluidos = [t for t in study.trials if t.state.name == "COMPLETE"]
    melhor = (
        f"melhor ate agora: trial {study.best_trial.number} = {study.best_value:.4f}"
        if concluidos
        else "nenhum trial concluido ainda"
    )
    valor = "-" if trial.value is None else f"{trial.value:.4f}"
    log(
        f"rtn/optuna > trial {trial.number} FIM: estado={trial.state.name}, "
        f"{OBJECTIVE}={valor}, {trial.user_attrs.get('minutos', '?')} min, "
        f"pico VRAM={trial.user_attrs.get('pico_vram_mib', '?')} MiB; {melhor}"
    )
    salvar_estudo(study, pasta)


if __name__ == "__main__":
    try:
        context = fl.init()
    except Exception:
        log_erro("fl.init() falhou — sem contexto nao ha log na plataforma")
        raise SystemExit(1)

    log(f"rtn/run > experimento de hiperparametros: {NOME_EXPERIMENTO}")
    log(
        "rtn/run > flautim inicializado",
        details=(
            f"experiment_id={context.experiment.id} "
            f"path={context.filesystem.path} "
            f"output_path={context.filesystem.output_path}"
        ),
    )

    try:
        with etapa("importar o optuna", object="optuna"):
            optuna = garantir_pacote("optuna", "optuna>=4.0,<5")
            log(f"rtn/optuna > versao {optuna.__version__}")
    except Exception as erro:
        log_erro("sem optuna nao ha busca", erro)
        raise SystemExit(1)

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
                batch_size=4,
                num_workers=NUM_WORKERS,
                seed=SEED,
            )
    except RetinaError as erro:
        log(f"rtn/run > ERRO FATAL no dataset: {erro}", object="dataset")
        raise SystemExit(1)
    except Exception as erro:
        log_erro("falha inesperada ao construir o dataset", erro)
        raise SystemExit(1)

    pasta_optuna = os.path.join(context.filesystem.output_path, "optuna")
    log(
        f"rtn/optuna > busca: {N_TRIALS} trials x {TRIAL_EPOCHS} epocas (validacao a "
        f"cada {VAL_EVERY}), timeout={TIMEOUT_HOURS}h, objetivo={OBJECTIVE} na validacao, "
        f"resultados em {pasta_optuna}"
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=NOME_EXPERIMENTO,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=6),
        # Nao poda antes de 3 trials completos nem antes da 4a epoca: o
        # aquecimento do lr deixa as primeiras epocas pouco informativas.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4),
    )
    for configuracao in CONFIGURACOES_INICIAIS:
        study.enqueue_trial(configuracao)

    with tempfile.TemporaryDirectory(prefix="rtn_optuna_") as pasta_tmp:
        with etapa("busca de hiperparametros (Optuna)", object="optuna"):
            study.optimize(
                criar_objetivo(optuna, context, dataset, pasta_tmp),
                n_trials=N_TRIALS,
                timeout=TIMEOUT_HOURS * 3600,
                callbacks=[lambda s, t: resumo_trial(s, t, pasta_optuna)],
                catch=(Exception,),
                gc_after_trial=True,
            )

    concluidos = [t for t in study.trials if t.state.name == "COMPLETE"]
    estados = {}
    for t in study.trials:
        estados[t.state.name] = estados.get(t.state.name, 0) + 1
    log(f"rtn/optuna > busca encerrada: {len(study.trials)} trials, estados={estados}")
    if not concluidos:
        log("rtn/optuna > ERRO: nenhum trial concluido; nada para retreinar")
        raise SystemExit(1)

    ranking = sorted(concluidos, key=lambda t: t.value, reverse=True)
    for posicao, t in enumerate(ranking[:5], 1):
        log(f"rtn/optuna > top {posicao}: trial {t.number} {OBJECTIVE}={t.value:.4f} {json.dumps(t.params)}")
    try:
        importancias = optuna.importance.get_param_importances(study)
        log(
            "rtn/optuna > importancia dos hiperparametros: "
            + ", ".join(f"{k}={v:.3f}" for k, v in importancias.items())
        )
    except Exception as erro:
        log(f"rtn/optuna > importancia nao calculada: {type(erro).__name__}: {erro}")

    melhores = study.best_params
    log(f"rtn/optuna > melhor configuracao (trial {study.best_trial.number}): {json.dumps(melhores)}")

    if not FINAL_RETRAIN:
        log("rtn/run > RETINA_FINAL_RETRAIN=0: sem retreino final")
        log("rtn/run > execucao finalizada")
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Retreino final com a melhor configuracao, pelo ciclo da Flautim
    # ------------------------------------------------------------------
    semear(dataset, SEED)
    checkpoint = os.path.join(context.filesystem.output_path, "retinanet_bacilos.pt")
    try:
        with etapa(f"construcao do modelo final ({EPOCHS} epocas)", object="model"):
            model, experiment = construir(context, dataset, melhores, EPOCHS, checkpoint)
    except Exception as erro:
        log_erro("falha ao construir o modelo final", erro)
        raise SystemExit(1)

    with etapa("treino final (Experiment.run)", object="experiment"):
        experiment.run(metrics={})

    try:
        metricas = experiment.evaluate_test_set()
        log(f"rtn/run > metricas no conjunto de teste: {metricas}", object="test")
        with open(os.path.join(pasta_optuna, "test_metrics.json"), "w") as arquivo:
            json.dump({"params": melhores, "epochs": EPOCHS, "metricas": metricas}, arquivo, indent=2)
        log(
            "rtn/teste > referencia RetinaNet da tese (Thales, dataset-2/3): "
            "P=0.9078/0.9285, R=0.5002/0.5314, F1=0.6447/0.6748, AP=0.4707/0.5040"
        )
    except Exception as erro:
        log_erro("falha ao avaliar o conjunto de teste (o treino nao foi afetado)", erro)

    log("rtn/run > execucao finalizada")
