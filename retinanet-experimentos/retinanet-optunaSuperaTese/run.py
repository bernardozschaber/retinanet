# Experimento de hiperparametros: retinanet-optunaSuperaTese
# Busca (Optuna) cujo objetivo e superar AO MESMO TEMPO as 4 metricas
# medias do RetinaNet da tese, seguida de retreino com varias sementes
# acompanhado no Weights & Biases (W&B)
# --------------------------------------------------------------
# Ponto de entrada do experimento na Flautim.
#
# Alvo (media dos datasets 1-4, Tabelas 22-25 da tese, limiar 0,5):
#   Precisao > 0,9339   Sensibilidade > 0,4741   F1 > 0,6244   AP > 0,4533
#
# O que se aprendeu nos experimentos anteriores e foi aplicado aqui:
#   - focal_alpha e a alavanca de calibracao: 0,25 da precisao ~0,95 e
#     sensibilidade ~0,60 no limiar 0,5 (1a rodada); 0,5 e 0,6 dao F1 maior
#     mas derrubam a precisao para 0,83-0,86, abaixo da tese. A busca
#     explora 0,25 a 0,5, onde fica o equilibrio;
#   - a precisao e a metrica que trava (e a unica abaixo da tese nos runs
#     do Optuna); o objetivo exige folga nela tambem;
#   - lr maior ajudou com alpha 0,25 (altoLearningRate, que supera as 4);
#     grad_clip 2,5-5, batch 8 e momentum ~0,92-0,95 dominaram as buscas;
#   - trainable_layers=5 e warmup=300 fixos (importancia ~0);
#   - um resultado de uma semente so engana (a semente 42 foi a melhor de
#     5 no wandbSementes): a configuracao final e treinada com varias
#     sementes e o log diz em quantas delas as 4 metricas superam a tese;
#   - o checkpoint de cada semente e escolhido pelo MESMO criterio da
#     busca (folga sobre a tese na validacao), e nao pela AP_VOC;
#   - novo: a curva precisao x sensibilidade do teste e os contadores
#     TP/FP/FN sao gravados em JSON, para a visualizacao.
#
# Objetivo na validacao (por epoca avaliada):
#   - se as 4 metricas passam a tese com folga: 1 + F1  (sempre > 0)
#   - senao: -(soma do que falta para cada metrica atingir tese + folga)
#   Assim qualquer configuracao que supera as 4 vale mais do que qualquer
#   uma que nao supera, e entre as que superam ganha a de maior F1.
#
# Usa os MESMOS RetinaDataset.py (retinanet-dataset/), RetinaModel.py
# (retinanet-model/) e RetinaExperiment.py (retinanet-experiment/), sem alteracao.
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

NOME_EXPERIMENTO = "retinanet-optunaSuperaTese"

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
MAX_IMAGES = os.getenv("RETINA_MAX_IMAGES")
MAX_IMAGES = int(MAX_IMAGES) if MAX_IMAGES else None
SEED = int(os.getenv("RETINA_SEED", "42"))

# Orcamento: busca (~9 min por trial completo) + sementes (~26 min cada).
N_TRIALS = int(os.getenv("RETINA_OPTUNA_TRIALS", "16"))
TRIAL_EPOCHS = int(os.getenv("RETINA_TRIAL_EPOCHS", "12"))
TIMEOUT_HOURS = float(os.getenv("RETINA_OPTUNA_TIMEOUT_HOURS", "5"))
VAL_EVERY = max(1, int(os.getenv("RETINA_TRIAL_VAL_EVERY", "2")))
EPOCHS = int(os.getenv("RETINA_EPOCHS", "30"))
SEMENTES = [int(s) for s in os.getenv("RETINA_SEEDS", "42,43,44").split(",")]

# Alvo: media do RetinaNet da tese nos datasets 1-4 (limiar 0,5).
REF_TESE = {"PRECISION": 0.933875, "RECALL": 0.47405, "F1": 0.624425, "AP_TESE": 0.453325}
NOMES = {"PRECISION": "Precisao", "RECALL": "Sensibilidade", "F1": "F1", "AP_TESE": "AP"}
# Folga exigida na validacao (20 laminas). Nao precisa ser maior na
# precisao: na validacao ela sai MAIS BAIXA que no teste (Baseline: 0,917
# na validacao x 0,954 no teste), entao passar a tese na validacao ja da
# margem no teste. RETINA_FOLGA_PRECISAO permite apertar so a precisao.
FOLGA = float(os.getenv("RETINA_FOLGA", "0.01"))
FOLGA_PRECISAO = float(os.getenv("RETINA_FOLGA_PRECISAO", "0.01"))
FOLGAS = {k: (FOLGA_PRECISAO if k == "PRECISION" else FOLGA) for k in REF_TESE}

FIXOS_BUSCA = {"trainable_layers": 5, "warmup_steps": 300}

# Pontos de partida avaliados primeiro:
#   1. altoLearningRate da 1a rodada (supera as 4 no teste);
#   2. trial 12 da 1a busca com alpha 0,4 (meio caminho entre 0,25 e 0,5);
#   3. trial 1 da 2a busca com alpha 0,4.
CONFIGURACOES_INICIAIS = [
    {"learning_rate": 1e-2, "momentum": 0.9, "weight_decay": 1e-4, "grad_clip": 10.0,
     "batch_size": 4, "focal_alpha": 0.25, "focal_gamma": 2.0, "fg_iou": 0.5,
     "nms_thresh": 0.4, "merge_nms": 0.4},
    {"learning_rate": 0.0026159960682001554, "momentum": 0.9185585731063599,
     "weight_decay": 0.0002902988766898052, "grad_clip": 5.0, "batch_size": 4,
     "focal_alpha": 0.4, "focal_gamma": 2.0, "fg_iou": 0.4, "nms_thresh": 0.4, "merge_nms": 0.3},
    {"learning_rate": 0.0025210814666085596, "momentum": 0.9465500014486941,
     "weight_decay": 0.00027586986748117065, "grad_clip": 2.5, "batch_size": 8,
     "focal_alpha": 0.4, "focal_gamma": 1.5, "fg_iou": 0.5, "nms_thresh": 0.3, "merge_nms": 0.25},
]

WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "retinanet-bacilos")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None

RESUMO = ["PRECISION", "RECALL", "F1", "AP_TESE", "AP_VOC", "AP50", "F1_IOU50"]


def garantir_pacote(modulo, requisito):
    """Importa `modulo`; se faltar, instala sem deixar o pip trocar o numpy."""
    try:
        return __import__(modulo)
    except ImportError:
        log(f"rtn/run > {modulo} ausente; instalando {requisito}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", requisito, "numpy==1.26.4"]
        )
        return __import__(modulo)


# ##################################################################
# Criterio da tese
# ##################################################################
def objetivo_tese(metricas):
    """(valor, folgas): folga = metrica - media da tese, para as 4 metricas."""
    folgas = {k: float(metricas[k]) - ref for k, ref in REF_TESE.items()}
    falta = sum(max(0.0, FOLGAS[k] - folga) for k, folga in folgas.items())
    if falta == 0.0:
        return 1.0 + float(metricas["F1"]), folgas
    return -falta, folgas


def supera_todas(metricas):
    return all(float(metricas[k]) > ref for k, ref in REF_TESE.items())


def formatar_folgas(folgas):
    return ", ".join(f"{NOMES[k]} {v:+.4f}" for k, v in folgas.items())


# ##################################################################
# Espaco de busca
# ##################################################################
def sugerir(trial):
    return {
        **FIXOS_BUSCA,
        "learning_rate": trial.suggest_float("learning_rate", 2e-3, 1.2e-2, log=True),
        "momentum": trial.suggest_float("momentum", 0.88, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 8e-5, 4e-4, log=True),
        "grad_clip": trial.suggest_categorical("grad_clip", [2.5, 5.0, 10.0]),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8]),
        "focal_alpha": trial.suggest_categorical("focal_alpha", [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
        "focal_gamma": trial.suggest_categorical("focal_gamma", [1.5, 2.0]),
        "fg_iou": trial.suggest_categorical("fg_iou", [0.4, 0.45, 0.5]),
        "nms_thresh": trial.suggest_categorical("nms_thresh", [0.3, 0.4]),
        "merge_nms": trial.suggest_categorical("merge_nms", [0.25, 0.3, 0.4]),
    }


# ##################################################################
# Ajustes no detector sem tocar no RetinaModel.py
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
                tags=["retinanet", "flautim", "sementes", "superaTese"], reinit=True,
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
# Experimento: W&B + checkpoint pelo criterio da tese + curva PR
# ##################################################################
class RetinaExperimentTese(RetinaExperiment.RetinaExperiment):
    """RetinaExperiment que escolhe o checkpoint pela folga sobre a tese.

    O validation_loop do RetinaExperiment salva o checkpoint pela AP_VOC.
    Aqui ele roda com model_path apontando para um arquivo descartavel, e
    o checkpoint de verdade e salvo quando o objetivo_tese melhora.
    """

    def __init__(self, model, dataset, context, wandb_run=None, descarte=None, **kwargs):
        super().__init__(model, dataset, context, **kwargs)
        self.wandb_run = wandb_run
        # Path, como o model_path do RetinaExperiment (ele usa .parent).
        self.descarte = Path(descarte) if descarte else self.model_path.with_name("descarte.pt")
        self.melhor_obj = -math.inf
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
        valor, folgas = objetivo_tese(metricas)
        metricas["OBJ_TESE"] = valor
        log(
            f"rtn/tese > validacao epoca {self.current_epoch}: objetivo={valor:.4f}; "
            f"folga sobre a tese: {formatar_folgas(folgas)}"
        )
        if valor > self.melhor_obj:
            self.melhor_obj = valor
            self.best_threshold = metricas.get("BEST_THRESHOLD", self.score_threshold)
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"modelo": self.model.state_dict(), "limiar": self.best_threshold,
                 "epoca": self.current_epoch, "metricas_validacao": metricas},
                self.model_path,
            )
            log(f"rtn/tese > checkpoint salvo (criterio da tese): epoca {self.current_epoch}, "
                f"objetivo={valor:.4f}")
        dados = {f"validacao/{k.lower()}": v for k, v in metricas.items()}
        dados.update({f"validacao/folga_{k.lower()}": v for k, v in folgas.items()})
        dados["validacao/loss"] = loss
        self._enviar(dados)
        return loss, metricas

    def evaluate_test_set(self):
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


def construir(context, dataset, params, epochs, model_path, wandb_run=None, descarte=None):
    aplicar_perda_focal(params["focal_alpha"], params["focal_gamma"])
    dataset.batch_size = int(params["batch_size"])
    model = RetinaModel.RetinaModel(
        context, tile_size=TILE_SIZE, architecture="retinanet",
        trainable_backbone_layers=int(params["trainable_layers"]),
        nms_thresh=float(params["nms_thresh"]),
    )
    ajustar_matcher(model, params["fg_iou"])
    experiment = RetinaExperimentTese(
        model, dataset, context, wandb_run=wandb_run, descarte=descarte,
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
        melhor = -math.inf
        try:
            model, experiment = construir(context, dataset, params, TRIAL_EPOCHS, caminho)
            for epoca in range(1, TRIAL_EPOCHS + 1):
                loss, _ = experiment.training_loop(dataset.dataloader())
                if epoca % VAL_EVERY and epoca != TRIAL_EPOCHS:
                    continue
                registros = experiment.inferir_particao(dataset.validation(), "validacao")
                metricas = experiment.metricas(registros, SCORE_THRESHOLD)
                valor, folgas = objetivo_tese(metricas)
                melhor = max(melhor, valor)
                trial.set_user_attr(f"epoca_{epoca}", {k: round(float(v), 4) for k, v in metricas.items()})
                log(
                    f"rtn/optuna > trial {trial.number} epoca {epoca}/{TRIAL_EPOCHS}: "
                    f"objetivo={valor:.4f} (melhor {melhor:.4f}); P={metricas['PRECISION']:.4f}, "
                    f"R={metricas['RECALL']:.4f}, F1={metricas['F1']:.4f}, AP={metricas['AP_TESE']:.4f}; "
                    f"folga: {formatar_folgas(folgas)}"
                )
                trial.report(valor, epoca)
                if trial.should_prune():
                    log(f"rtn/optuna > trial {trial.number} PODADO na epoca {epoca}")
                    raise optuna.TrialPruned()
            return melhor
        except torch.cuda.OutOfMemoryError as erro:
            log_erro(f"trial {trial.number} sem memoria na GPU; podado", erro)
            raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as erro:
            log_erro(f"trial {trial.number} falhou", erro)
            raise
        finally:
            trial.set_user_attr("minutos", round((time.time() - t0) / 60, 1))
            del model, experiment
            if os.path.exists(caminho):
                os.remove(caminho)
            liberar_gpu()

    return objetivo


def salvar_estudo(study, pasta):
    os.makedirs(pasta, exist_ok=True)
    try:
        study.trials_dataframe().to_csv(os.path.join(pasta, "trials.csv"), index=False)
    except Exception as erro:
        log_erro("nao consegui gravar trials.csv", erro)
    if any(t.state.name == "COMPLETE" for t in study.trials):
        with open(os.path.join(pasta, "best_params.json"), "w") as arquivo:
            json.dump({"trial": study.best_trial.number, "objetivo": "tese (1 + F1 se supera as 4)",
                       "valor": study.best_value, "params": study.best_params,
                       "trial_epochs": TRIAL_EPOCHS}, arquivo, indent=2)


def resumo_trial(study, trial, pasta):
    concluidos = [t for t in study.trials if t.state.name == "COMPLETE"]
    melhor = (f"melhor ate agora: trial {study.best_trial.number} = {study.best_value:.4f}"
              if concluidos else "nenhum trial concluido ainda")
    valor = "-" if trial.value is None else f"{trial.value:.4f}"
    supera = "SIM" if trial.value is not None and trial.value > 0 else "nao"
    log(f"rtn/optuna > trial {trial.number} FIM: estado={trial.state.name}, objetivo={valor} "
        f"(supera as 4 na validacao: {supera}), {trial.user_attrs.get('minutos', '?')} min; {melhor}")
    salvar_estudo(study, pasta)


if __name__ == "__main__":
    try:
        context = fl.init()
    except Exception:
        log_erro("fl.init() falhou — sem contexto nao ha log na plataforma")
        raise SystemExit(1)

    log(f"rtn/run > experimento de hiperparametros: {NOME_EXPERIMENTO}")
    log(f"rtn/run > alvo (media da tese, limiar 0,5): {REF_TESE}; folga exigida na validacao: {FOLGAS}")
    log("rtn/run > flautim inicializado", details=(
        f"experiment_id={context.experiment.id} path={context.filesystem.path} "
        f"output_path={context.filesystem.output_path}"))

    try:
        with etapa("importar o optuna", object="optuna"):
            optuna = garantir_pacote("optuna", "optuna>=4.0,<5")
        with etapa("importar o W&B", object="wandb"):
            wandb, modo = importar_wandb()
    except Exception as erro:
        log_erro("sem optuna nao ha busca", erro)
        raise SystemExit(1)

    try:
        with etapa("construcao do Dataset", object="dataset"):
            dataset = RetinaDataset.RetinaDataset(
                dataset_path=DATASET_PATH, tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP,
                crops_per_image=CROPS_PER_IMAGE, target_box_size=TARGET_BOX, scale=SCALE,
                max_images=MAX_IMAGES, batch_size=4, num_workers=NUM_WORKERS, seed=SEED,
            )
    except RetinaError as erro:
        log(f"rtn/run > ERRO FATAL no dataset: {erro}", object="dataset")
        raise SystemExit(1)
    except Exception as erro:
        log_erro("falha inesperada ao construir o dataset", erro)
        raise SystemExit(1)

    saida = context.filesystem.output_path
    pasta_optuna = os.path.join(saida, "optuna")
    pasta_sementes = os.path.join(saida, "sementes")
    pasta_wandb = os.path.join(saida, "wandb")
    for pasta in (pasta_optuna, pasta_sementes, pasta_wandb):
        os.makedirs(pasta, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Busca
    # ------------------------------------------------------------------
    log(f"rtn/optuna > busca: {N_TRIALS} trials x {TRIAL_EPOCHS} epocas (validacao a cada "
        f"{VAL_EVERY}), timeout={TIMEOUT_HOURS}h, objetivo = superar as 4 metricas da tese")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=NOME_EXPERIMENTO, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4),
    )
    for configuracao in CONFIGURACOES_INICIAIS:
        study.enqueue_trial(configuracao)

    with tempfile.TemporaryDirectory(prefix="rtn_supera_") as pasta_tmp:
        with etapa("busca de hiperparametros (Optuna)", object="optuna"):
            study.optimize(
                criar_objetivo(optuna, context, dataset, pasta_tmp),
                n_trials=N_TRIALS, timeout=TIMEOUT_HOURS * 3600,
                callbacks=[lambda s, t: resumo_trial(s, t, pasta_optuna)],
                catch=(Exception,), gc_after_trial=True,
            )

        concluidos = [t for t in study.trials if t.state.name == "COMPLETE"]
        estados = {}
        for t in study.trials:
            estados[t.state.name] = estados.get(t.state.name, 0) + 1
        log(f"rtn/optuna > busca encerrada: {len(study.trials)} trials, estados={estados}, "
            f"superam as 4 na validacao: {sum(1 for t in concluidos if t.value > 0)}")
        if not concluidos:
            log("rtn/optuna > ERRO: nenhum trial concluido; nada para retreinar")
            raise SystemExit(1)
        for posicao, t in enumerate(sorted(concluidos, key=lambda t: t.value, reverse=True)[:5], 1):
            log(f"rtn/optuna > top {posicao}: trial {t.number} objetivo={t.value:.4f} {json.dumps(t.params)}")
        try:
            importancias = optuna.importance.get_param_importances(study)
            log("rtn/optuna > importancia dos hiperparametros: "
                + ", ".join(f"{k}={v:.3f}" for k, v in importancias.items()))
        except Exception as erro:
            log(f"rtn/optuna > importancia nao calculada: {type(erro).__name__}: {erro}")

        melhores = {**FIXOS_BUSCA, **study.best_params}
        log(f"rtn/optuna > melhor configuracao (trial {study.best_trial.number}): {json.dumps(melhores)}")

        # --------------------------------------------------------------
        # 2. Retreino da melhor configuracao com varias sementes
        # --------------------------------------------------------------
        config = {"tile_size": TILE_SIZE, "target_box": TARGET_BOX, "epochs": EPOCHS,
                  "score_threshold": SCORE_THRESHOLD, "eval_iou": EVAL_IOU, **melhores}
        resultados, curvas = {}, {}
        descarte = os.path.join(pasta_tmp, "descarte.pt")
        for numero, semente in enumerate(SEMENTES, 1):
            ultima = numero == len(SEMENTES)
            caminho = (os.path.join(saida, "retinanet_bacilos.pt") if ultima
                       else os.path.join(pasta_tmp, f"semente_{semente}.pt"))
            run = abrir_run(wandb, modo, pasta_wandb, f"semente-{semente}",
                            {**config, "seed": semente}, "semente")
            t0 = time.time()
            model = experiment = None
            try:
                with etapa(f"semente {semente} ({numero}/{len(SEMENTES)})", object="experiment"):
                    semear(dataset, semente)
                    model, experiment = construir(context, dataset, melhores, EPOCHS, caminho,
                                                  wandb_run=run, descarte=descarte)
                    # So a ultima semente passa pelo experiment.run(), que marca
                    # o run como concluido na plataforma (mesmo esquema do
                    # retinanet-wandbSementes).
                    if ultima:
                        experiment.run(metrics={})
                    else:
                        for _ in range(EPOCHS):
                            experiment.training_loop(dataset.dataloader())
                            experiment.validation_loop(dataset.dataloader(validation=True))
                    metricas = experiment.evaluate_test_set()
                metricas["BEST_EPOCH"] = float(torch.load(caminho, map_location="cpu").get("epoca", -1))
                resultados[semente] = metricas
                curvas[semente] = experiment.curva_pr
                # Tambem no log, para a visualizacao ler direto do .md colado.
                log(f"rtn/curva_pr > semente {semente}: {json.dumps(experiment.curva_pr)}")
                _, folgas = objetivo_tese(metricas)
                log(
                    f"rtn/sementes > semente {semente}: "
                    + ", ".join(f"{k}={metricas[k]:.4f}" for k in RESUMO)
                    + f", melhor epoca={metricas['BEST_EPOCH']:.0f}, {(time.time() - t0) / 60:.1f} min; "
                    f"SUPERA AS 4 DA TESE: {'SIM' if supera_todas(metricas) else 'NAO'} "
                    f"({formatar_folgas(folgas)})",
                    object="test",
                )
            except Exception as erro:
                log_erro(f"semente {semente} falhou; seguindo com as demais", erro)
            finally:
                fechar_run(run)
                del model, experiment
                liberar_gpu()
            with open(os.path.join(pasta_sementes, "sementes.json"), "w") as arquivo:
                json.dump({"config": config, "resultados": resultados}, arquivo, indent=2)
            with open(os.path.join(pasta_sementes, "curvas_pr.json"), "w") as arquivo:
                json.dump(curvas, arquivo, indent=2)

    if not resultados:
        log("rtn/sementes > ERRO: nenhuma semente terminou")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 3. Resumo: quantas sementes superam as 4, e a media
    # ------------------------------------------------------------------
    resumo = {}
    for chave in RESUMO:
        valores = np.asarray([m[chave] for m in resultados.values()], dtype=np.float64)
        resumo[chave] = {"media": float(valores.mean()),
                         "desvio": float(valores.std(ddof=1)) if len(valores) > 1 else 0.0,
                         "min": float(valores.min()), "max": float(valores.max())}
    vencedoras = [s for s, m in resultados.items() if supera_todas(m)]
    media = {k: v["media"] for k, v in resumo.items()}
    log(
        f"rtn/sementes > RESUMO ({len(resultados)} sementes, limiar {SCORE_THRESHOLD}): "
        + ", ".join(f"{k}={v['media']:.4f} (±{v['desvio']:.4f})" for k, v in resumo.items()),
        object="test",
    )
    log(
        f"rtn/tese > sementes que superam as 4 metricas da tese: {len(vencedoras)}/{len(resultados)} "
        f"{vencedoras}; a MEDIA supera as 4: {'SIM' if supera_todas(media) else 'NAO'} "
        f"({formatar_folgas(objetivo_tese(media)[1])})",
        object="test",
    )
    log(f"rtn/tese > alvo (media da tese): P>{REF_TESE['PRECISION']:.4f}, R>{REF_TESE['RECALL']:.4f}, "
        f"F1>{REF_TESE['F1']:.4f}, AP>{REF_TESE['AP_TESE']:.4f}")
    with open(os.path.join(pasta_sementes, "resumo.json"), "w") as arquivo:
        json.dump({"config": config, "sementes": list(resultados), "resumo": resumo,
                   "sementes_que_superam_as_4": vencedoras, "alvo_tese": REF_TESE}, arquivo, indent=2)

    run = abrir_run(wandb, modo, pasta_wandb, "resumo", config, "resumo")
    if run is not None:
        try:
            for chave, valores in resumo.items():
                for estatistica, valor in valores.items():
                    run.summary[f"teste/{chave.lower()}_{estatistica}"] = valor
            run.summary["teste/sementes_que_superam_as_4"] = len(vencedoras)
        except Exception as erro:
            log(f"rtn/wandb > AVISO falha ao enviar o resumo: {type(erro).__name__}: {erro}")
        fechar_run(run)

    log("rtn/run > execucao finalizada")
