from pathlib import Path
import contextlib
import os
import time
import traceback

import cv2
import flautim as fl
import numpy as np
import pandas as pd
import torch
from flautim.pytorch.Dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


# ##################################################################
# 0. LOG E DIAGNOSTICO
# ##################################################################
# Mesmo padrao do TBDataset.py: estas funcoes tambem sao importadas
# pelo RetinaExperiment.py e pelo RetinaModel.py (a plataforma copia
# os arquivos achatados no diretorio do experimento, entao "from
# RetinaDataset import log" funciona como um modulo comum comum).
#
# Tudo vai para o stdout do pod (que a Flautim captura no log do
# experimento) e, quando o fl.init() ja rodou, tambem para o log da
# plataforma. Linhas de progresso de dentro de loops usam
# plataforma=False para nao inundar o Mongo.


class RetinaError(RuntimeError):
    """Erro do experimento com mensagem ja explicativa.

    Existe para separar, no traceback, o que e falha conhecida e
    diagnosticada (caminho errado, CSV sem coluna, imagem ilegivel)
    do que e bug de verdade.
    """


def log(message, details="", object="", plataforma=True):
    """Escreve no stdout do pod e, se possivel, no log da Flautim."""
    linha = f"[RTN] {message}"
    if details:
        linha += f" | {details}"
    print(linha, flush=True)

    if not plataforma:
        return
    try:
        fl.log(message, details=str(details), object=str(object))
    except Exception:
        # Antes do fl.init(), ou rodando fora da plataforma: o stdout
        # acima ja basta, e um log que falha nao pode derrubar o run.
        pass


def log_erro(message, excecao=None, details=""):
    """Loga uma falha com tipo, mensagem e traceback completo."""
    if excecao is not None:
        details = f"{type(excecao).__name__}: {excecao}" + (f" | {details}" if details else "")
    log(f"ERRO: {message}", details=details)
    print(traceback.format_exc(), flush=True)


@contextlib.contextmanager
def etapa(nome, object=""):
    """Marca inicio/fim/falha de uma etapa, com o tempo gasto.

    Uso:
        with etapa("leitura do Annotation.csv"):
            ...

    Assim, no log do experimento, toda etapa aparece como um par
    "INICIO x" / "OK x" — e se algo estourar no meio, a ultima linha
    "INICIO" sem "OK" diz exatamente onde o run parou.
    """
    log(f"INICIO: {nome}", object=object)
    t0 = time.time()
    try:
        yield
    except Exception as ex:
        log_erro(f"FALHOU na etapa '{nome}' apos {time.time() - t0:.1f}s", ex)
        raise
    log(f"OK: {nome}", details=f"{time.time() - t0:.1f}s", object=object)


# ##################################################################
# 1. LOCALIZACAO DOS DADOS
# ##################################################################
# A plataforma monta o volume compartilhado em raizes diferentes
# conforme o cluster que pega o job: no k3s-prod ele aparece sob
# /home/flautim_workplace/..., no k8s-bcm o mesmo dado so existe sob
# /mnt/flautim_workplace/... (confirmado comparando TBlog.md, rodado
# no k3s-prod, com retina3log.md, rodado no k8s-bcm — a
# 'Error fetching logs' e o FileNotFoundError vinham do pod ter so
# /mnt montado). resolve_dataset_path tenta as duas raizes antes de
# desistir, para o experimento nao depender de qual cluster pegou o
# job.
def resolve_dataset_path(base):
    """Devolve o Path do dataset a partir de `base`.

    Aceita tanto `.../db_tb/Dataset` quanto `.../db_tb` — o
    Annotation.csv e o images/ sao procurados no proprio diretorio,
    num subdiretorio `Dataset/` e no diretorio pai. Tambem tenta a
    mesma arvore sob a raiz irma /home <-> /mnt, porque o cluster que
    executa o job muda onde o volume compartilhado aparece no pod.
    """
    base = os.path.abspath(os.path.expanduser(str(base)))
    log(f"rtn/dados > procurando o dataset a partir de: {base}")

    # Diagnostico do caminho: mostra ate onde a arvore existe. Se o
    # volume nao foi montado no pod (ou foi montado noutra raiz), isso
    # aponta o nivel exato que falta, em vez de so dizer "nao achei".
    partes = base.strip(os.sep).split(os.sep)
    acumulado = os.sep
    for parte in partes:
        acumulado = os.path.join(acumulado, parte)
        if os.path.isdir(acumulado):
            try:
                filhos = sorted(os.listdir(acumulado))
                amostra = ", ".join(filhos[:8]) + (" ..." if len(filhos) > 8 else "")
                log(
                    f"rtn/dados >   diretorio existe: {acumulado}",
                    details=f"{len(filhos)} item(ns): {amostra}",
                    plataforma=False,
                )
            except PermissionError as ex:
                log(
                    f"rtn/dados >   diretorio existe mas SEM PERMISSAO de leitura: "
                    f"{acumulado}",
                    details=str(ex),
                )
        else:
            log(f"rtn/dados >   NAO EXISTE a partir daqui: {acumulado}")
            break

    raizes = [base]
    for antiga, nova in (("/home/", "/mnt/"), ("/mnt/", "/home/")):
        if base.startswith(antiga):
            alternativa = nova + base[len(antiga):]
            if alternativa not in raizes:
                raizes.append(alternativa)

    tentativas = []
    for raiz in raizes:
        candidatos = [
            raiz,
            os.path.join(raiz, "Dataset"),
            os.path.dirname(raiz),
            os.path.join(os.path.dirname(raiz), "Dataset"),
        ]
        for candidato in dict.fromkeys(candidatos):
            csv_path = os.path.join(candidato, "Annotation.csv")
            images_dir = os.path.join(candidato, "images")
            tem_csv = os.path.isfile(csv_path)
            tem_imgs = os.path.isdir(images_dir)
            tentativas.append(
                f"{candidato} (csv={'sim' if tem_csv else 'nao'}, "
                f"images={'sim' if tem_imgs else 'nao'})"
            )
            if tem_csv and tem_imgs:
                log(
                    "rtn/dados > caminho do dataset encontrado",
                    details=f"annotation={csv_path} images={images_dir}",
                )
                return Path(candidato)
            log(f"rtn/dados >   candidato descartado: {tentativas[-1]}", plataforma=False)

    raise RetinaError(
        f"Nao encontrei 'Annotation.csv' + 'images/' em nenhum candidato. "
        f"Tentei: {tentativas}. Confira se o volume do dataset foi montado no "
        f"pod e ajuste RETINA_DATASET_PATH (valor atual: {base!r})."
    )


COLUNAS_OBRIGATORIAS = {"filename", "xmin", "ymin", "xmax", "ymax"}


def carregar_anotacoes(caminho_csv):
    """Le e valida o Annotation.csv usado pelo experimento RetinaNet."""
    caminho_csv = Path(caminho_csv)
    log(
        f"rtn/dados > lendo anotacoes em {caminho_csv} "
        f"({caminho_csv.stat().st_size} bytes)"
    )
    if caminho_csv.stat().st_size == 0:
        raise RetinaError(f"Arquivo de anotacoes vazio: {caminho_csv}")

    try:
        tabela = pd.read_csv(caminho_csv, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as erro:
        raise RetinaError(
            f"Arquivo de anotacoes sem cabecalho ou dados: {caminho_csv}"
        ) from erro

    tabela.columns = [str(coluna).strip().lower() for coluna in tabela.columns]
    log(
        f"rtn/dados > csv interpretado: {len(tabela)} linhas; "
        f"colunas={list(tabela.columns)}"
    )
    faltantes = COLUNAS_OBRIGATORIAS - set(tabela.columns)
    if faltantes:
        raise RetinaError(f"Colunas ausentes em {caminho_csv}: {sorted(faltantes)}")
    if tabela.empty:
        raise RetinaError(f"Arquivo de anotacoes sem registros: {caminho_csv}")

    tabela = tabela.copy()
    tabela["filename"] = tabela["filename"].astype(str).str.strip()
    tabela["filename"] = tabela["filename"].map(os.path.basename)
    if (tabela["filename"] == "").any():
        raise RetinaError("Existem registros sem filename no Annotation.csv.")

    colunas_coordenada = ["xmin", "ymin", "xmax", "ymax"]
    for coluna in colunas_coordenada:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    nao_numericas = ~np.isfinite(tabela[colunas_coordenada]).all(axis=1)
    if nao_numericas.any():
        log(
            f"rtn/dados > {int(nao_numericas.sum())} registro(s) com coordenada "
            f"nao numerica descartado(s)"
        )
        tabela = tabela[~nao_numericas]

    geometria_invalida = (tabela["xmax"] <= tabela["xmin"]) | (
        tabela["ymax"] <= tabela["ymin"]
    )
    if geometria_invalida.any():
        log(
            f"rtn/dados > {int(geometria_invalida.sum())} caixa(s) com largura ou "
            f"altura invalida descartada(s)"
        )
        tabela = tabela[~geometria_invalida]

    if tabela.empty:
        raise RetinaError(
            f"Nenhuma anotacao utilizavel sobrou em {caminho_csv} apos a limpeza."
        )

    if "label" in tabela.columns:
        tabela["label"] = tabela["label"].astype(str).str.strip().str.lower()
        log(
            f"rtn/dados > rotulos presentes: "
            f"{tabela['label'].value_counts().to_dict()}; o detector e de uma "
            f"classe, entao todos contam como objeto positivo"
        )

    sem_duplicatas = tabela.drop_duplicates(
        subset=["filename", "xmin", "ymin", "xmax", "ymax"]
    ).reset_index(drop=True)
    log(
        f"rtn/dados > anotacoes validas: {len(sem_duplicatas)} caixas em "
        f"{sem_duplicatas['filename'].nunique()} imagens; duplicatas removidas="
        f"{len(tabela) - len(sem_duplicatas)}"
    )
    return sem_duplicatas


def escolher_escala(anotacoes, lado_alvo=40.0, escala_fixa=None):
    """Fator de escala aplicado a todas as imagens antes de recortar.

    A tese (Exp. 5 e 8.1) mostrou que o ganho grande veio de nao
    encolher a lamina inteira ate caber na rede: os bacilos sao
    pequenos demais para isso. Aqui a imagem e reescalada de modo que
    o bacilo mediano fique com ~`lado_alvo` px de lado (tamanho em
    que as ancoras pequenas do detector funcionam bem) e depois e
    processada em ladrilhos, sem distorcer a proporcao.
    """
    caixas = anotacoes
    if "label" in anotacoes.columns and (anotacoes["label"] == "bacillus").any():
        caixas = anotacoes[anotacoes["label"] == "bacillus"]
    largura = (caixas["xmax"] - caixas["xmin"]).to_numpy(np.float64)
    altura = (caixas["ymax"] - caixas["ymin"]).to_numpy(np.float64)
    lado = np.sqrt(largura * altura)
    p10, p50, p90 = np.percentile(lado, [10, 50, 90])
    log(
        f"rtn/dados > tamanho das caixas (px originais, sqrt(area)): "
        f"p10={p10:.1f}, mediana={p50:.1f}, p90={p90:.1f}; "
        f"largura mediana={np.median(largura):.1f}, altura mediana={np.median(altura):.1f}"
    )
    if escala_fixa:
        escala = float(escala_fixa)
        origem = "fixada por RETINA_SCALE"
    else:
        escala = float(np.clip(lado_alvo / max(p50, 1.0), 0.25, 2.0))
        origem = f"automatica (bacilo mediano -> {lado_alvo:.0f} px)"
    escala = round(escala, 4)
    log(
        f"rtn/dados > escala das imagens: {escala} ({origem}); bacilo mediano "
        f"passa a ter ~{p50 * escala:.1f} px"
    )
    return escala


def repartir_imagens(
    anotacoes,
    caminhos_imagem,
    maximo_imagens=None,
    semente=42,
    fracoes=(0.70, 0.10, 0.20),
):
    """Split deterministico por imagem (70/10/20, como no Exp. 6 da tese)."""
    disponiveis = [
        nome
        for nome in anotacoes["filename"].drop_duplicates()
        if nome in caminhos_imagem
    ]
    ausentes = anotacoes["filename"].nunique() - len(disponiveis)
    log(
        f"rtn/dados > cruzamento anotacoes x imagens: encontradas="
        f"{len(disponiveis)}, ausentes={ausentes}"
    )
    if maximo_imagens is not None:
        if maximo_imagens < 3:
            raise RetinaError("maximo_imagens deve ser None ou pelo menos 3.")
        if len(disponiveis) < maximo_imagens:
            raise RetinaError(
                f"Foram solicitadas {maximo_imagens} imagens, mas somente "
                f"{len(disponiveis)} foram encontradas."
            )

    sorteio = np.random.default_rng(semente)
    sorteio.shuffle(disponiveis)
    selecionadas = (
        disponiveis if maximo_imagens is None else disponiveis[:maximo_imagens]
    )
    if len(selecionadas) < 3:
        raise RetinaError("Sao necessarias pelo menos tres imagens anotadas.")

    brutos = np.asarray(fracoes, dtype=np.float64) / sum(fracoes) * len(selecionadas)
    quantidades = np.floor(brutos).astype(int)
    folga = len(selecionadas) - quantidades.sum()
    for indice in np.argsort(-(brutos - quantidades))[:folga]:
        quantidades[indice] += 1
    for indice in np.where(quantidades == 0)[0]:
        doador = int(np.argmax(quantidades))
        quantidades[doador] -= 1
        quantidades[indice] = 1
    n_treino, n_validacao, _ = quantidades.tolist()

    particoes = {
        "train": selecionadas[:n_treino],
        "validation": selecionadas[n_treino:n_treino + n_validacao],
        "test": selecionadas[n_treino + n_validacao:],
    }
    log(
        f"rtn/dados > split sem sobreposicao: treino={len(particoes['train'])}, "
        f"validacao={len(particoes['validation'])}, teste={len(particoes['test'])}, "
        f"semente={semente}, limite={maximo_imagens or 'todas'}"
    )
    return particoes


def posicoes_ladrilho(comprimento, lado, sobreposicao):
    """Inicios dos ladrilhos ao longo de um eixo.

    Todo ladrilho tem exatamente `lado` px (o ultimo e puxado para
    dentro da imagem em vez de sair com borda), para que o transform
    interno do detector nunca precise reescalar a entrada.
    """
    if comprimento <= lado:
        return [0]
    passo = max(1, lado - sobreposicao)
    quantidade = int(np.ceil((comprimento - lado) / passo)) + 1
    return sorted({int(round(p)) for p in np.linspace(0, comprimento - lado, quantidade)})


class CacheDeImagens:
    """Imagens ja lidas e reescaladas, guardadas em RAM.

    Com num_workers=0 tudo roda no processo principal, entao um dict
    simples basta. Alem de poupar a decodificacao do JPEG a cada
    recorte, isso faz o aviso "Premature end of JPEG file" (que vem da
    libjpeg, e inofensivo) aparecer uma vez por imagem, e nao a cada
    epoca.
    """

    def __init__(self, escala, limite_bytes=8 * 1024**3):
        self.escala = float(escala)
        self.limite_bytes = int(limite_bytes)
        self.bytes_usados = 0
        self.imagens = {}
        self.tamanho_original = {}

    def ler(self, nome, caminho):
        imagem = self.imagens.get(nome)
        if imagem is not None:
            return imagem
        imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
        if imagem is None or imagem.size == 0:
            raise FileNotFoundError(f"Falha ao abrir a imagem: {caminho}")
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        self.tamanho_original[nome] = imagem.shape[:2]
        if self.escala != 1.0:
            interpolacao = cv2.INTER_AREA if self.escala < 1.0 else cv2.INTER_LINEAR
            imagem = cv2.resize(
                imagem,
                None,
                fx=self.escala,
                fy=self.escala,
                interpolation=interpolacao,
            )
        imagem = np.ascontiguousarray(imagem)
        if self.bytes_usados + imagem.nbytes <= self.limite_bytes:
            self.imagens[nome] = imagem
            self.bytes_usados += imagem.nbytes
        return imagem


def _cor_de_fundo(imagem):
    return [int(v) for v in np.median(imagem.reshape(-1, 3), axis=0)]


def recortar(imagem, caixas, x0, y0, lado, cor_fundo=None, visibilidade_minima=0.45):
    """Recorta um quadrado `lado` x `lado` em (x0, y0), com borda se faltar imagem."""
    altura, largura = imagem.shape[:2]
    x1, y1 = min(largura, x0 + lado), min(altura, y0 + lado)
    pedaco = imagem[max(0, y0):y1, max(0, x0):x1]
    if pedaco.shape[0] != lado or pedaco.shape[1] != lado:
        if cor_fundo is None:
            cor_fundo = _cor_de_fundo(imagem)
        fundo = np.empty((lado, lado, 3), dtype=imagem.dtype)
        fundo[:] = cor_fundo
        fundo[: pedaco.shape[0], : pedaco.shape[1]] = pedaco
        pedaco = fundo

    if len(caixas) == 0:
        return pedaco, np.zeros((0, 4), dtype=np.float32)
    deslocadas = caixas - np.asarray([x0, y0, x0, y0], dtype=np.float32)
    area = (deslocadas[:, 2] - deslocadas[:, 0]) * (deslocadas[:, 3] - deslocadas[:, 1])
    cortadas = deslocadas.copy()
    cortadas[:, [0, 2]] = cortadas[:, [0, 2]].clip(0, min(lado, x1 - x0))
    cortadas[:, [1, 3]] = cortadas[:, [1, 3]].clip(0, min(lado, y1 - y0))
    largura_c = cortadas[:, 2] - cortadas[:, 0]
    altura_c = cortadas[:, 3] - cortadas[:, 1]
    # Bacilo cortado pela borda do recorte: so vira alvo se a maior
    # parte dele estiver visivel; um pedacinho de bastonete rotulado
    # como objeto ensina mais ruido do que sinal.
    manter = (
        (largura_c >= 1.0)
        & (altura_c >= 1.0)
        & (largura_c * altura_c >= visibilidade_minima * np.maximum(area, 1e-6))
    )
    return pedaco, cortadas[manter].astype(np.float32)


def aumentar(imagem, caixas, rng):
    """Augmentation geometrica e de cor so com cv2 + numpy.

    Lamina de microscopio nao tem "lado de cima": giros de 90 graus e
    espelhamentos nos dois eixos sao todos validos. A variacao de cor
    imita a diferenca de coloracao (Ziehl-Neelsen) entre laminas.
    """
    lado = imagem.shape[0]
    caixas = caixas.copy()

    if rng.random() < 0.5:
        imagem = imagem[:, ::-1]
        caixas[:, [0, 2]] = lado - caixas[:, [2, 0]]
    if rng.random() < 0.5:
        imagem = imagem[::-1, :]
        caixas[:, [1, 3]] = lado - caixas[:, [3, 1]]
    if rng.random() < 0.5:
        # transposicao + espelho = giro de 90 graus (quadrado, entao lado igual)
        imagem = imagem.transpose(1, 0, 2)
        caixas = caixas[:, [1, 0, 3, 2]]
    imagem = np.ascontiguousarray(imagem)

    if rng.random() < 0.8:
        hsv = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        hsv[..., 2] *= rng.uniform(0.75, 1.25)
        imagem = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.3:
        contraste = rng.uniform(0.8, 1.2)
        media = imagem.mean()
        imagem = ((imagem.astype(np.float32) - media) * contraste + media).clip(0, 255)
        imagem = imagem.astype(np.uint8)
    if rng.random() < 0.1:
        imagem = cv2.GaussianBlur(imagem, (3, 3), 0)
    return imagem, caixas


class RetinaTorchDataset(TorchDataset):
    """Particao do dataset no formato de ladrilhos.

    - treino: a cada item sorteia um recorte de `tile_size` px da imagem
      reescalada (na maior parte das vezes centrado num bacilo, com
      jitter de escala) e aplica augmentation; cada imagem gera
      `crops_per_image` recortes por epoca.
    - validacao/teste: ladrilhos fixos em grade, com sobreposicao, para
      medir a perda. A deteccao em imagem inteira (usada nas metricas)
      sai de `carregar()` + `ladrilhos()`.
    """

    def __init__(
        self,
        nomes,
        anotacoes,
        caminhos_imagem,
        cache,
        tile_size=640,
        tile_overlap=128,
        crops_per_image=8,
        treinando=False,
        probabilidade_bacilo=0.8,
        seed=42,
    ):
        self.nomes = list(nomes)
        self.caminhos_imagem = caminhos_imagem
        self.cache = cache
        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)
        self.crops_per_image = max(1, int(crops_per_image))
        self.treinando = treinando
        self.probabilidade_bacilo = float(probabilidade_bacilo)
        self.rng = np.random.default_rng(seed)
        self.caixas_por_nome = {
            nome: grupo[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)
            for nome, grupo in anotacoes.groupby("filename", sort=False)
        }
        self._grade = None

    # ---------------------------------------------------------------
    def carregar(self, nome):
        """Imagem RGB uint8 ja reescalada e as caixas na mesma escala."""
        imagem = self.cache.ler(nome, self.caminhos_imagem[nome])
        caixas = self.caixas_por_nome.get(
            nome, np.zeros((0, 4), dtype=np.float32)
        ) * np.float32(self.cache.escala)
        altura, largura = imagem.shape[:2]
        caixas[:, [0, 2]] = caixas[:, [0, 2]].clip(0, largura)
        caixas[:, [1, 3]] = caixas[:, [1, 3]].clip(0, altura)
        validas = (caixas[:, 2] - caixas[:, 0] >= 1) & (caixas[:, 3] - caixas[:, 1] >= 1)
        return imagem, caixas[validas]

    def ladrilhos(self, imagem):
        """[(x0, y0, tensor 3xLxL em [0,1])] cobrindo a imagem inteira."""
        altura, largura = imagem.shape[:2]
        cor_fundo = _cor_de_fundo(imagem)
        saida = []
        for y0 in posicoes_ladrilho(altura, self.tile_size, self.tile_overlap):
            for x0 in posicoes_ladrilho(largura, self.tile_size, self.tile_overlap):
                pedaco, _ = recortar(
                    imagem, np.zeros((0, 4), np.float32), x0, y0,
                    self.tile_size, cor_fundo,
                )
                saida.append((x0, y0, self.para_tensor(pedaco)))
        return saida

    @staticmethod
    def para_tensor(imagem):
        # O detector do torchvision normaliza por media/desvio do
        # ImageNet dentro do proprio forward, entao aqui basta [0, 1].
        return torch.from_numpy(np.ascontiguousarray(imagem)).permute(2, 0, 1).float() / 255.0

    # ---------------------------------------------------------------
    def _grade_avaliacao(self):
        if self._grade is None:
            self._grade = []
            for nome in self.nomes:
                imagem = self.cache.ler(nome, self.caminhos_imagem[nome])
                altura, largura = imagem.shape[:2]
                for y0 in posicoes_ladrilho(altura, self.tile_size, self.tile_overlap):
                    for x0 in posicoes_ladrilho(largura, self.tile_size, self.tile_overlap):
                        self._grade.append((nome, x0, y0))
        return self._grade

    def __len__(self):
        if self.treinando:
            return len(self.nomes) * self.crops_per_image
        return len(self._grade_avaliacao())

    def _recorte_aleatorio(self, nome):
        imagem, caixas = self.carregar(nome)
        altura, largura = imagem.shape[:2]
        # jitter de escala: recorta um quadrado um pouco maior/menor e
        # devolve-o ao tamanho do ladrilho
        lado = int(round(self.tile_size * self.rng.uniform(0.8, 1.25)))
        if len(caixas) and self.rng.random() < self.probabilidade_bacilo:
            alvo = caixas[self.rng.integers(len(caixas))]
            cx = (alvo[0] + alvo[2]) / 2 + self.rng.uniform(-0.4, 0.4) * lado
            cy = (alvo[1] + alvo[3]) / 2 + self.rng.uniform(-0.4, 0.4) * lado
            x0, y0 = int(cx - lado / 2), int(cy - lado / 2)
        else:
            x0 = int(self.rng.integers(0, max(1, largura - lado + 1)))
            y0 = int(self.rng.integers(0, max(1, altura - lado + 1)))
        x0 = int(np.clip(x0, 0, max(0, largura - lado)))
        y0 = int(np.clip(y0, 0, max(0, altura - lado)))

        pedaco, caixas = recortar(imagem, caixas, x0, y0, lado)
        if lado != self.tile_size:
            fator = self.tile_size / lado
            pedaco = cv2.resize(
                pedaco, (self.tile_size, self.tile_size),
                interpolation=cv2.INTER_AREA if fator < 1 else cv2.INTER_LINEAR,
            )
            caixas = caixas * np.float32(fator)
        pedaco, caixas = aumentar(pedaco, caixas, self.rng)
        validas = (caixas[:, 2] - caixas[:, 0] >= 1) & (caixas[:, 3] - caixas[:, 1] >= 1)
        return pedaco, caixas[validas]

    def __getitem__(self, indice):
        if self.treinando:
            nome = self.nomes[indice % len(self.nomes)]
            pedaco, caixas = self._recorte_aleatorio(nome)
        else:
            nome, x0, y0 = self._grade_avaliacao()[indice]
            imagem, caixas = self.carregar(nome)
            pedaco, caixas = recortar(imagem, caixas, x0, y0, self.tile_size)
        caixas = torch.from_numpy(np.ascontiguousarray(caixas, dtype=np.float32)).reshape(-1, 4)
        return self.para_tensor(pedaco), caixas, nome


def juntar_lote(lote):
    """Os ladrilhos tem lado fixo, mas o numero de caixas varia."""
    imagens, caixas, nomes = zip(*lote)
    return torch.stack(imagens), list(caixas), list(nomes)


class RetinaDataset(Dataset):
    """Dataset Flautim para deteccao de bacilos de TB com RetinaNet."""

    # A GPU do cluster (Quadro RTX 6000, 24 GB) aguenta com folga lotes
    # de 4 ladrilhos 640x640 com AMP; acima de 8 o risco de OOM volta.
    BATCH_SIZE_PADRAO = 4
    BATCH_SIZE_MAXIMO = 8

    def __init__(
        self,
        dataset_path,
        tile_size=640,
        tile_overlap=128,
        crops_per_image=8,
        target_box_size=40.0,
        scale=None,
        max_images=None,
        seed=42,
        **kwargs,
    ):
        super(RetinaDataset, self).__init__("Bacilos-RetinaNet", **kwargs)

        # O Dataset base da Flautim usa batch_size=128 quando o valor nao
        # e passado. Com imagens grandes isso coloca o treino inteiro num
        # unico lote e estoura a GPU ("CUDA out of memory. Tried to
        # allocate 5.81 GiB", log de 24/09/2026) — o Experiment.run()
        # engole a excecao e o run termina como "Falha de execucao".
        pedido = kwargs.get("batch_size")
        self.batch_size = int(pedido) if pedido else self.BATCH_SIZE_PADRAO
        if self.batch_size > self.BATCH_SIZE_MAXIMO:
            log(
                f"rtn/dados > AVISO batch_size={self.batch_size} e grande demais para "
                f"ladrilhos de {tile_size}px; usando {self.BATCH_SIZE_MAXIMO}"
            )
            self.batch_size = self.BATCH_SIZE_MAXIMO

        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)
        self.crops_per_image = int(crops_per_image)
        log("rtn/dados > inicializando o dataset de bacilos")
        log(
            f"rtn/dados > configuracao: tile_size={self.tile_size}, "
            f"tile_overlap={self.tile_overlap}, crops_por_imagem={self.crops_per_image}, "
            f"batch_size={self.batch_size}, num_workers={self.num_workers}, "
            f"shuffle={self.shuffle}, semente={seed}, "
            f"max_images={max_images or 'todas'}"
        )

        with etapa("localizar o dataset no disco", object="dataset"):
            self.dataset_path = resolve_dataset_path(dataset_path)
        self.image_dir = self.dataset_path / "images"
        self.annotation_csv = self.dataset_path / "Annotation.csv"

        with etapa("indexar as imagens recursivamente", object="dataset"):
            self.indice_imagens = {}
            nomes_repetidos = 0
            for caminho in self.image_dir.rglob("*"):
                if caminho.is_file():
                    if caminho.name in self.indice_imagens:
                        nomes_repetidos += 1
                    else:
                        self.indice_imagens[caminho.name] = caminho
            log(
                f"rtn/dados > arquivos indexados: {len(self.indice_imagens)}; "
                f"nomes repetidos ignorados={nomes_repetidos}"
            )

        with etapa("ler o Annotation.csv", object="dataset"):
            self.anotacoes = carregar_anotacoes(self.annotation_csv)
            self.escala = escolher_escala(
                self.anotacoes, lado_alvo=target_box_size, escala_fixa=scale
            )
            self.cache = CacheDeImagens(self.escala)

        with etapa("validar e carregar as imagens anotadas", object="dataset"):
            candidatos = {}
            for nome in self.anotacoes["filename"].drop_duplicates():
                direto = self.image_dir / nome
                if direto.is_file():
                    candidatos[nome] = direto
                elif Path(nome).name in self.indice_imagens:
                    candidatos[nome] = self.indice_imagens[Path(nome).name]
            log(f"rtn/dados > conferindo a leitura de {len(candidatos)} imagens anotadas")

            # A leitura de validacao ja deixa a imagem reescalada no cache.
            legiveis = {}
            ilegiveis = 0
            for nome, caminho in candidatos.items():
                try:
                    self.cache.ler(nome, caminho)
                except FileNotFoundError:
                    ilegiveis += 1
                    try:
                        bytes_arquivo = caminho.stat().st_size
                    except OSError:
                        bytes_arquivo = -1
                    log(
                        f"rtn/dados > AVISO imagem corrompida ou ilegivel ignorada: "
                        f"{nome}; caminho={caminho}; bytes={bytes_arquivo}"
                    )
                else:
                    legiveis[nome] = caminho

            self.caminhos_imagem = legiveis
            tamanhos = list(self.cache.tamanho_original.values())
            if tamanhos:
                alturas, larguras = zip(*tamanhos)
                log(
                    f"rtn/dados > tamanho original das imagens: "
                    f"largura {min(larguras)}-{max(larguras)}, "
                    f"altura {min(alturas)}-{max(alturas)}"
                )
            log(
                f"rtn/dados > validacao das imagens concluida: legiveis="
                f"{len(self.caminhos_imagem)}, ignoradas={ilegiveis}; cache em RAM="
                f"{self.cache.bytes_usados / 1024**2:.0f} MiB "
                f"({len(self.cache.imagens)} imagens)"
            )
            if not self.caminhos_imagem:
                raise RetinaError(
                    f"Nenhuma imagem anotada pode ser lida em {self.image_dir}. "
                    f"Confira se o volume de imagens foi montado por completo."
                )

        with etapa("montar os splits treino/validacao/teste", object="dataset"):
            self.particoes = repartir_imagens(
                self.anotacoes,
                self.caminhos_imagem,
                maximo_imagens=max_images,
                semente=seed,
            )

            comuns = {
                "anotacoes": self.anotacoes,
                "caminhos_imagem": self.caminhos_imagem,
                "cache": self.cache,
                "tile_size": self.tile_size,
                "tile_overlap": self.tile_overlap,
                "crops_per_image": self.crops_per_image,
                "seed": seed,
            }
            self.particao_treino = RetinaTorchDataset(
                self.particoes["train"], treinando=True, **comuns
            )
            self.particao_validacao = RetinaTorchDataset(
                self.particoes["validation"], **comuns
            )
            self.particao_teste = RetinaTorchDataset(self.particoes["test"], **comuns)
            log(
                f"rtn/dados > particoes pytorch criadas: "
                f"treino={len(self.particoes['train'])} imagens "
                f"({len(self.particao_treino)} recortes/epoca), "
                f"validacao={len(self.particoes['validation'])} imagens "
                f"({len(self.particao_validacao)} ladrilhos), "
                f"teste={len(self.particoes['test'])} imagens "
                f"({len(self.particao_teste)} ladrilhos)"
            )

    def dataloader(self, validation=False, test=False):
        if test:
            particao, embaralhar, rotulo = self.particao_teste, False, "teste"
        elif validation:
            particao, embaralhar, rotulo = self.particao_validacao, False, "validacao"
        else:
            particao, embaralhar, rotulo = (
                self.particao_treino,
                self.shuffle,
                "treino",
            )

        log(
            f"rtn/dados > montando o dataloader de {rotulo}: "
            f"itens={len(particao)}, batch_size={self.batch_size}, "
            f"shuffle={embaralhar}, num_workers={self.num_workers}",
            plataforma=False,
        )
        return DataLoader(
            particao,
            batch_size=self.batch_size,
            shuffle=embaralhar,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=juntar_lote,
        )

    def train(self):
        return self.particao_treino

    def validation(self):
        return self.particao_validacao

    def test(self):
        return self.particao_teste

    def manifest(self):
        log("rtn/dados > gerando o manifesto dos splits")
        linhas = []
        for particao, nomes in self.particoes.items():
            for nome in nomes:
                linhas.append(
                    {
                        "split": particao,
                        "filename": nome,
                        "boxes": int((self.anotacoes["filename"] == nome).sum()),
                    }
                )
        manifesto = pd.DataFrame(linhas)
        log(
            f"rtn/dados > manifesto gerado: {len(manifesto)} imagens e "
            f"{manifesto['boxes'].sum()} caixas"
        )
        return manifesto
