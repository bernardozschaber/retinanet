import math
from pathlib import Path

import numpy as np
import torch
from flautim.pytorch.centralized.Experiment import Experiment
from torchvision.ops import box_iou, nms

from RetinaDataset import etapa, log, log_erro


# Limiares usados pela AP da tese (np.arange(0.2, 0.9, 0.05) no
# average_precision_score do notebook do Thales).
LIMIARES_AP_TESE = np.round(np.arange(0.2, 0.9, 0.05), 2)


# ##################################################################
# Metricas
# ##################################################################
# Protocolo do notebook do Thales (hiperparametros_thales.ipynb, secao
# 7), para os numeros serem comparaveis aos da tese:
#   - casamento guloso predicao -> GT por ordem de score, cada predicao
#     fica com o GT ainda livre de maior IoU;
#   - acerto se IoU >= 0,1 (o bacilo e fino e a caixa anotada e
#     imprecisa; IoU 0,5 pune caixas corretas com borda diferente);
#   - precisao/sensibilidade/F1 no limiar de score 0,5;
#   - AP varrendo os limiares 0,20..0,85.
# A AP "VOC" (todos os pontos) com IoU 0,5 continua sendo reportada
# como AP50, a medida usual de deteccao.
def casar_guloso(sobreposicoes, limiar_iou):
    """Para predicoes ja em ordem de score: 1 se acertou, 0 se nao."""
    n_pred, n_real = sobreposicoes.shape
    acertos = np.zeros(n_pred, dtype=np.int8)
    if n_pred == 0 or n_real == 0:
        return acertos
    livres = np.ones(n_real, dtype=bool)
    for i in range(n_pred):
        candidatas = np.where(livres, sobreposicoes[i], -1.0)
        j = int(candidatas.argmax())
        if candidatas[j] >= limiar_iou:
            livres[j] = False
            acertos[i] = 1
    return acertos


def ap_voc(scores, acertos, total_real):
    """AP interpolada em todos os pontos da curva precisao-recall."""
    if total_real == 0 or len(scores) == 0:
        return 0.0
    ordem = np.argsort(-scores, kind="stable")
    acertos = acertos[ordem]
    verdadeiros = np.cumsum(acertos)
    falsos = np.cumsum(1 - acertos)
    recall = verdadeiros / total_real
    precisao = verdadeiros / np.maximum(verdadeiros + falsos, 1)
    curva_recall = np.concatenate(([0.0], recall, [recall[-1]]))
    curva_precisao = np.concatenate(([1.0], precisao, [0.0]))
    curva_precisao = np.maximum.accumulate(curva_precisao[::-1])[::-1]
    return float(np.sum(np.diff(curva_recall) * curva_precisao[1:]))


def prf_no_limiar(scores, acertos, total_real, limiar):
    manter = scores >= limiar
    verdadeiros = int(acertos[manter].sum())
    preditos = int(manter.sum())
    precisao = verdadeiros / preditos if preditos else 0.0
    recall = verdadeiros / total_real if total_real else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    return precisao, recall, f1


def ap_tese(scores, acertos, total_real, corte):
    """average_precision_score do notebook do Thales, sem sklearn.

    As predicoes sao cortadas em `corte` antes; os GT nao casados entram
    com score 0, entao para limiares abaixo do corte o ponto da curva e
    o mesmo do corte.
    """
    precisoes, recalls = [], []
    for limiar in LIMIARES_AP_TESE:
        p, r, _ = prf_no_limiar(scores, acertos, total_real, max(limiar, corte))
        precisoes.append(p)
        recalls.append(r)
    precisoes.append(1.0)
    recalls.append(0.0)
    precisoes, recalls = np.asarray(precisoes), np.asarray(recalls)
    return float(np.sum((recalls[:-1] - recalls[1:]) * precisoes[:-1]))


class RetinaExperiment(Experiment):
    """Treino e avaliacao do detector pelo ciclo centralizado da Flautim."""

    def __init__(self, model, dataset, context, **kwargs):
        super(RetinaExperiment, self).__init__(model, dataset, context, **kwargs)
        self.epochs = int(kwargs.get("epochs", 30))
        self.total_epochs = self.epochs
        self.score_threshold = float(kwargs.get("score_threshold", 0.5))
        self.eval_iou = float(kwargs.get("eval_iou", 0.1))
        self.merge_nms = float(kwargs.get("merge_nms", 0.4))
        self.warmup_steps = int(kwargs.get("warmup_steps", 300))
        self.grad_clip = float(kwargs.get("grad_clip", 10.0))
        self.model_path = Path(kwargs.get("model_path", "retinanet_bacilos.pt"))
        self.log_every_batches = max(1, int(kwargs.get("log_every_batches", 25)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.device.type == "cuda"
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(kwargs.get("learning_rate", 5e-3)),
            momentum=float(kwargs.get("momentum", 0.9)),
            weight_decay=float(kwargs.get("weight_decay", 1e-4)),
        )
        self.lr_base = self.optimizer.param_groups[0]["lr"]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.scheduler = None
        self.best_score = -1.0
        self.best_threshold = self.score_threshold
        self.current_epoch = 0
        self.completed_epochs = 0
        log("rtn/exp > experimento centralizado criado")
        log(f"rtn/exp > dispositivo selecionado: {self.device}")
        if self.device.type == "cuda":
            log(
                f"rtn/exp > gpu: {torch.cuda.get_device_name(0)}; "
                f"CUDA={torch.version.cuda}"
            )
        log(
            f"rtn/exp > treinamento: epochs={self.total_epochs}, "
            f"lr={self.lr_base}, momentum={self.optimizer.param_groups[0]['momentum']}, "
            f"weight_decay={self.optimizer.param_groups[0]['weight_decay']}, "
            f"warmup={self.warmup_steps} passos + cosseno, grad_clip={self.grad_clip}"
        )
        log(
            f"rtn/exp > avaliacao (protocolo da tese): IoU={self.eval_iou}, "
            f"limiar de score={self.score_threshold}, NMS entre ladrilhos="
            f"{self.merge_nms}"
        )
        log(f"rtn/exp > checkpoint: {self.model_path}")

    # --------------------------------------------------------------
    # O dataset entrega (ladrilhos, lista de caixas, nomes); o detector
    # do torchvision espera uma lista de tensores e uma lista de
    # dicionarios. Ladrilhos sem bacilo entram no lote como exemplos
    # negativos (alvo vazio): ensinam o fundo e seguram a precisao.
    # --------------------------------------------------------------
    def preparar_lote(self, images, targets):
        imagens, alvos = [], []
        for imagem, caixas in zip(images, targets):
            caixas = caixas.to(self.device).float().reshape(-1, 4)
            imagens.append(imagem.to(self.device, non_blocking=True))
            alvos.append(
                {
                    "boxes": caixas,
                    "labels": torch.ones(
                        (len(caixas),), dtype=torch.int64, device=self.device
                    ),
                }
            )
        return imagens, alvos

    def _criar_scheduler(self, lotes_por_epoca):
        passos_totais = max(1, self.total_epochs * lotes_por_epoca)
        aquecimento = min(self.warmup_steps, max(1, passos_totais // 10))

        def fator_lr(passo):
            if passo < aquecimento:
                return 0.001 + (1 - 0.001) * passo / aquecimento
            progresso = min(1.0, (passo - aquecimento) / max(1, passos_totais - aquecimento))
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progresso))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, fator_lr)
        log(
            f"rtn/treino > scheduler: {passos_totais} passos, aquecimento={aquecimento}, "
            f"cosseno ate {0.01 * self.lr_base:.2e}"
        )

    def training_loop(self, data_loader):
        self.current_epoch += 1
        log(
            f"rtn/treino > inicio da epoca {self.current_epoch}/{self.total_epochs}; "
            f"lotes={len(data_loader)}"
        )
        self.model.to(self.device)
        self.model.train()
        if self.scheduler is None:
            self._criar_scheduler(len(data_loader))

        somas = {}
        soma_loss = 0.0
        caixas_vistas = 0
        lotes_usados = 0
        lotes_nao_finitos = 0

        for numero_lote, (images, targets, filenames) in enumerate(data_loader, 1):
            if numero_lote == 1 and self.current_epoch == 1:
                log(
                    f"rtn/treino > primeiro lote: tensor={tuple(images.shape)}, "
                    f"caixas={[len(t) for t in targets]}"
                )

            imagens, alvos = self.preparar_lote(images, targets)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                perdas = self.model(imagens, alvos)
                loss = sum(perdas.values())

            if not torch.isfinite(loss):
                # Com AMP o detector as vezes estoura num lote; pular e
                # melhor do que contaminar os pesos ou abortar o run.
                lotes_nao_finitos += 1
                self.scheduler.step()
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            soma_loss += float(loss.detach())
            for chave, valor in perdas.items():
                somas[chave] = somas.get(chave, 0.0) + float(valor.detach())
            caixas_vistas += sum(len(alvo["boxes"]) for alvo in alvos)
            lotes_usados += 1
            if numero_lote % self.log_every_batches == 0:
                detalhes = ", ".join(f"{k}={float(v):.4f}" for k, v in perdas.items())
                log(
                    f"rtn/treino > epoca {self.current_epoch}/{self.total_epochs}, "
                    f"lote {numero_lote}/{len(data_loader)}: loss={float(loss):.4f} "
                    f"({detalhes}), lr={self.optimizer.param_groups[0]['lr']:.2e}",
                    plataforma=False,
                )

        if lotes_usados == 0:
            raise RuntimeError(
                f"A epoca {self.current_epoch} nao processou nenhum lote valido "
                f"({len(data_loader)} lote(s), {lotes_nao_finitos} com perda nao finita)."
            )

        loss_medio = soma_loss / lotes_usados
        metricas = {
            f"LOSS_{chave.upper()}": soma / lotes_usados for chave, soma in somas.items()
        }
        metricas["BOXES"] = float(caixas_vistas)
        log(
            f"rtn/treino > fim da epoca {self.current_epoch}/{self.total_epochs}: "
            f"loss_medio={loss_medio:.4f}, "
            + ", ".join(f"{k}={v:.4f}" for k, v in metricas.items())
            + f", lotes={lotes_usados}/{len(data_loader)} "
            f"(nao finitos={lotes_nao_finitos}), lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )
        if self.device.type == "cuda":
            log(
                f"rtn/treino > memoria CUDA: alocada="
                f"{torch.cuda.memory_allocated() / 1024**2:.0f} MiB, pico="
                f"{torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB",
                plataforma=False,
            )
        return float(loss_medio), metricas

    def validation_loop(self, data_loader):
        log(f"rtn/valid > inicio da validacao da epoca {self.current_epoch}")
        self.model.to(self.device)
        soma_loss = 0.0
        lotes_usados = 0
        with torch.no_grad():
            # A perda so existe em modo treino; sem gradiente (e com o
            # BatchNorm do backbone congelado) o forward nao altera nada.
            self.model.train()
            for images, targets, _ in data_loader:
                imagens, alvos = self.preparar_lote(images, targets)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    perdas = self.model(imagens, alvos)
                loss_lote = float(sum(perdas.values()))
                if math.isfinite(loss_lote):
                    soma_loss += loss_lote
                    lotes_usados += 1
        loss_validacao = soma_loss / max(1, lotes_usados)

        registros = self.inferir_particao(self.dataset.validation(), "validacao")
        metricas = self.metricas(registros, self.score_threshold)
        limiar, f1_limiar = self.melhor_limiar(registros)
        metricas["BEST_THRESHOLD"] = limiar
        metricas["F1_BEST_THRESHOLD"] = f1_limiar
        log(
            f"rtn/valid > resultado da epoca {self.current_epoch}: loss={loss_validacao:.4f}, "
            + self._formatar(metricas)
        )

        # Selecao pela AP no IoU da tese: nao depende de limiar de score.
        criterio = metricas["AP_VOC"]
        if criterio > self.best_score:
            anterior = self.best_score
            self.best_score = criterio
            self.best_threshold = limiar
            with etapa(f"salvar o checkpoint em {self.model_path}", object="checkpoint"):
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "modelo": self.model.state_dict(),
                        "limiar": limiar,
                        "epoca": self.current_epoch,
                        "metricas_validacao": metricas,
                    },
                    self.model_path,
                )
            log(
                f"rtn/ckpt > nova melhor AP (IoU {self.eval_iou}) na validacao: "
                f"{anterior:.4f} -> {self.best_score:.4f}; limiar de score "
                f"escolhido={limiar:.2f}"
            )
        else:
            log(
                f"rtn/ckpt > sem melhora: AP atual={criterio:.4f}, "
                f"melhor={self.best_score:.4f}"
            )

        self.completed_epochs += 1
        return float(loss_validacao), {k: float(v) for k, v in metricas.items()}

    # --------------------------------------------------------------
    # Inferencia em ladrilhos (estrategia de particao da tese, Exp. 5)
    # --------------------------------------------------------------
    @torch.no_grad()
    def detectar_imagem(self, particao, imagem):
        """Detecta numa imagem inteira, ladrilho por ladrilho.

        Caixas que encostam na borda interna de um ladrilho sao
        descartadas: o bacilo cortado ali aparece inteiro no ladrilho
        vizinho (a sobreposicao e maior que um bacilo), e a caixa
        parcial viraria um falso positivo que o NMS nem sempre remove.
        """
        self.model.eval()
        altura, largura = imagem.shape[:2]
        lado = particao.tile_size
        margem = 2.0
        ladrilhos = particao.ladrilhos(imagem)
        todas_caixas, todos_scores = [], []
        tamanho_lote = max(1, int(self.dataset.batch_size))
        for inicio in range(0, len(ladrilhos), tamanho_lote):
            grupo = ladrilhos[inicio:inicio + tamanho_lote]
            entradas = [tensor.to(self.device) for _, _, tensor in grupo]
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                saidas = self.model(entradas)
            for (x0, y0, _), saida in zip(grupo, saidas):
                caixas = saida["boxes"].float()
                scores = saida["scores"].float()
                if len(caixas) == 0:
                    continue
                corta = torch.zeros(len(caixas), dtype=torch.bool, device=caixas.device)
                if x0 > 0:
                    corta |= caixas[:, 0] <= margem
                if y0 > 0:
                    corta |= caixas[:, 1] <= margem
                if x0 + lado < largura:
                    corta |= caixas[:, 2] >= lado - margem
                if y0 + lado < altura:
                    corta |= caixas[:, 3] >= lado - margem
                caixas, scores = caixas[~corta], scores[~corta]
                caixas = caixas + torch.tensor(
                    [x0, y0, x0, y0], dtype=caixas.dtype, device=caixas.device
                )
                todas_caixas.append(caixas)
                todos_scores.append(scores)

        if not todas_caixas:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        caixas = torch.cat(todas_caixas)
        scores = torch.cat(todos_scores)
        caixas[:, [0, 2]] = caixas[:, [0, 2]].clamp(0, largura)
        caixas[:, [1, 3]] = caixas[:, [1, 3]].clamp(0, altura)
        manter = nms(caixas, scores, self.merge_nms)
        return caixas[manter].cpu().numpy(), scores[manter].cpu().numpy()

    @torch.no_grad()
    def inferir_particao(self, particao, nome_particao):
        """Para cada imagem: scores ordenados e acertos em IoU 0,1 e 0,5."""
        registros = []
        for numero, nome in enumerate(particao.nomes, 1):
            imagem, caixas_reais = particao.carregar(nome)
            caixas, scores = self.detectar_imagem(particao, imagem)
            ordem = np.argsort(-scores, kind="stable")
            caixas, scores = caixas[ordem], scores[ordem]
            if len(caixas) and len(caixas_reais):
                sobreposicoes = box_iou(
                    torch.from_numpy(caixas), torch.from_numpy(caixas_reais)
                ).numpy()
            else:
                sobreposicoes = np.zeros((len(caixas), len(caixas_reais)), np.float32)
            registro = {
                "nome": nome,
                "scores": scores,
                "total_real": len(caixas_reais),
                "acertos": casar_guloso(sobreposicoes, self.eval_iou),
                "acertos50": casar_guloso(sobreposicoes, 0.5),
            }
            registros.append(registro)
            acima = scores >= self.score_threshold
            log(
                f"rtn/aval > {nome_particao} {numero}/{len(particao.nomes)}: "
                f"arquivo={nome}, anotadas={len(caixas_reais)}, "
                f"predicoes>={self.score_threshold}={int(acima.sum())}, "
                f"acertos(IoU {self.eval_iou})={int(registro['acertos'][acima].sum())}",
                plataforma=False,
            )
        return registros

    @staticmethod
    def _juntar(registros, chave="acertos"):
        scores = np.concatenate([r["scores"] for r in registros]) if registros else np.zeros(0)
        acertos = np.concatenate([r[chave] for r in registros]) if registros else np.zeros(0)
        total = int(sum(r["total_real"] for r in registros))
        return scores.astype(np.float64), acertos.astype(np.int64), total

    def metricas(self, registros, limiar):
        scores, acertos, total = self._juntar(registros, "acertos")
        scores50, acertos50, _ = self._juntar(registros, "acertos50")
        precisao, recall, f1 = prf_no_limiar(scores, acertos, total, limiar)
        _, _, f1_50 = prf_no_limiar(scores50, acertos50, total, limiar)
        return {
            "PRECISION": precisao,
            "RECALL": recall,
            "F1": f1,
            "AP_TESE": ap_tese(scores, acertos, total, limiar),
            "AP_VOC": ap_voc(scores, acertos, total),
            "AP50": ap_voc(scores50, acertos50, total),
            "F1_IOU50": f1_50,
            "THRESHOLD": float(limiar),
            "PREDICTIONS": float((scores >= limiar).sum()),
            "GROUND_TRUTH": float(total),
        }

    def melhor_limiar(self, registros):
        """Limiar de score com maior F1 (IoU da tese) — escolhido na validacao."""
        scores, acertos, total = self._juntar(registros, "acertos")
        melhor, melhor_f1 = self.score_threshold, -1.0
        for limiar in np.round(np.arange(0.05, 0.96, 0.05), 2):
            _, _, f1 = prf_no_limiar(scores, acertos, total, limiar)
            if f1 > melhor_f1:
                melhor, melhor_f1 = float(limiar), f1
        return melhor, float(max(melhor_f1, 0.0))

    @staticmethod
    def _formatar(metricas):
        return ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metricas.items()
        )

    # --------------------------------------------------------------
    def carregar_melhor_modelo(self):
        with etapa(f"carregar o melhor modelo ({self.model_path})", object="checkpoint"):
            estado = torch.load(self.model_path, map_location=self.device)
            if isinstance(estado, dict) and "modelo" in estado:
                self.best_threshold = float(estado.get("limiar", self.best_threshold))
                log(
                    f"rtn/ckpt > checkpoint da epoca {estado.get('epoca')} com limiar "
                    f"de score {self.best_threshold:.2f}"
                )
                estado = estado["modelo"]
            self.model.load_state_dict(estado)
            self.model.to(self.device)
            self.model.eval()

    def evaluate_test_set(self):
        log("rtn/teste > iniciando a avaliacao final do conjunto de teste")
        if self.model_path.is_file():
            try:
                self.carregar_melhor_modelo()
            except Exception as erro:
                log_erro(
                    "nao consegui recarregar o checkpoint; avaliando com os "
                    "pesos atuais",
                    erro,
                )
        self.model.to(self.device)
        registros = self.inferir_particao(self.dataset.test(), "teste")

        # 1) limiar fixo da tese (0,5): comparavel diretamente ao Exp. 8.1
        metricas = self.metricas(registros, self.score_threshold)
        log(f"rtn/teste > limiar {self.score_threshold:.2f} (tese): " + self._formatar(metricas))

        # 2) limiar escolhido na validacao (sem olhar o teste)
        if abs(self.best_threshold - self.score_threshold) > 1e-9:
            ajustado = self.metricas(registros, self.best_threshold)
            log(
                f"rtn/teste > limiar {self.best_threshold:.2f} (escolhido na validacao): "
                + self._formatar(ajustado)
            )
            for chave in ("PRECISION", "RECALL", "F1", "AP_TESE"):
                metricas[f"{chave}_VAL_THRESHOLD"] = ajustado[chave]
            metricas["VAL_THRESHOLD"] = self.best_threshold

        log(
            "rtn/teste > alvo da tese (Exp. 8.1, Faster R-CNN + particao): "
            "PRECISION=0.8769, RECALL=0.9341, F1=0.9046, AP=0.8664"
        )
        metricas = {k: float(v) for k, v in metricas.items()}
        log(f"rtn/teste > metricas finais: {metricas}")
        return metricas
