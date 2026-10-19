"""Gera o painel de diagnostico dos experimentos RetinaNet (retinanet-visualizacao.html).

Le os logs completos (log.md) de cada pasta em retinanet-experimentos/,
sem dependencias alem da biblioteca padrao. Experimentos novos entram sozinhos:
basta a pasta ter o log completo.

Tambem reescreve a linha do tempo do README.md da raiz (entre as marcas
<!-- linha-do-tempo:inicio --> e <!-- linha-do-tempo:fim -->).

Uso:
    python3 gerar_visualizacao.py                 # grava retinanet-visualizacao.html
    python3 gerar_visualizacao.py --fragmento X   # tambem grava a versao sem <html>/<head>
                                                  # (formato de publicacao como Artifact)
"""
import ast
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXPERIMENTOS = RAIZ / "retinanet-experimentos"
SAIDA = Path(__file__).resolve().parent / "retinanet-visualizacao.html"

# RetinaNet da tese (Thales), Tabelas 22-25: (media, desvio) por dataset.
TESE = {
    "dataset-1": {"PRECISION": (0.9660, 0.0096), "RECALL": (0.4071, 0.0278), "F1": (0.5723, 0.0276), "AP_TESE": (0.3971, 0.0283)},
    "dataset-2": {"PRECISION": (0.9078, 0.0033), "RECALL": (0.5002, 0.0224), "F1": (0.6447, 0.0179), "AP_TESE": (0.4707, 0.0214)},
    "dataset-3": {"PRECISION": (0.9285, 0.0225), "RECALL": (0.5314, 0.0365), "F1": (0.6748, 0.0273), "AP_TESE": (0.5040, 0.0331)},
    "dataset-4": {"PRECISION": (0.9332, 0.0260), "RECALL": (0.4575, 0.1085), "F1": (0.6059, 0.0918), "AP_TESE": (0.4415, 0.1032)},
}
METRICAS = ["PRECISION", "RECALL", "F1", "AP_TESE"]
# Experimentos da 1a rodada (variacao de um hiperparametro por vez).
PRIMEIRA_RODADA = {"baseline", "altoLearningRate", "baixoLearningRate", "altoEpochs", "altoWeightDecay",
                   "altoImageSize", "baixoImageSize", "altoTargetBox", "baixoTargetBox", "baixoTrainableLayers"}

# Hiperparametros de cada treino: a base e o que cada experimento mudou.
# Experimentos que nao estao aqui (ex.: um Optuna novo) tem a configuracao
# lida do proprio log (linha "melhor configuracao" ou "rtn/run > configuracao").
HIPER_BASE = {"image_size": 640, "target_box": 40, "batch_size": 4, "learning_rate": 5e-3,
              "momentum": 0.9, "weight_decay": 1e-4, "warmup_steps": 300, "grad_clip": 10.0, "epochs": 30,
              "trainable_layers": 5, "focal_alpha": 0.25, "focal_gamma": 2.0, "fg_iou": 0.5,
              "nms_thresh": 0.4, "merge_nms": 0.4}
TRIAL_12 = {"learning_rate": 0.0026159960682001554, "momentum": 0.9185585731063599,
            "weight_decay": 0.0002902988766898052, "grad_clip": 5.0, "focal_alpha": 0.5, "fg_iou": 0.4, "merge_nms": 0.3}
TRIAL_1_FOCAL = {"batch_size": 8, "learning_rate": 0.0025210814666085596, "momentum": 0.9465500014486941,
                 "weight_decay": 0.00027586986748117065, "grad_clip": 2.5, "focal_alpha": 0.6, "focal_gamma": 1.5,
                 "nms_thresh": 0.3, "merge_nms": 0.25}
HIPER_MUDANCA = {
    "baseline": {},
    "altoLearningRate": {"learning_rate": 1e-2},
    "baixoLearningRate": {"learning_rate": 2.5e-3},
    "altoEpochs": {"epochs": 40},
    "altoWeightDecay": {"weight_decay": 2e-4},
    "altoImageSize": {"image_size": 768},
    "baixoImageSize": {"image_size": 512},
    "altoTargetBox": {"target_box": 48},
    "baixoTargetBox": {"target_box": 32},
    "baixoTrainableLayers": {"trainable_layers": 3},
    "wandb": {"batch_size": 8, "learning_rate": 1e-2, "epochs": 40},
    "optuna": TRIAL_12,            # trial 12 da 1a busca
    "wandbSementes": TRIAL_12,     # trial 12, sementes 42-46
    "optunaFocalF1": TRIAL_1_FOCAL,         # trial 1 da 2a busca
    "wandbSementesFocalF1": TRIAL_1_FOCAL,  # trial 1, sementes 42-46
    "wandbSementesFocalF1Alpha055": {**TRIAL_1_FOCAL, "focal_alpha": 0.55},  # trial 1 com alpha 0,55, sementes 47-56
}
HIPER_FIXOS = "tile_overlap 128 · crops_per_image 8 · score_threshold 0,5 · evaluation_iou 0,1 · RetinaNet R50-FPN v2 com pesos COCO"


def hiper_do_log(eventos):
    """Configuracao escrita no log, para experimentos fora de HIPER_MUDANCA."""
    achado = {}
    for _, msg in eventos:
        m = re.search(r"melhor configuracao \(trial \d+\): (\{.*\})", msg)
        if m:
            achado = json.loads(m.group(1))
        m = re.search(r"rtn/run > configuracao: (\{.*\})", msg)
        if m and not achado:
            try:
                achado = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                pass
    if "tile_size" in achado:
        achado["image_size"] = achado["tile_size"]
    return {k: achado[k] for k in HIPER_BASE if k in achado}


# Linha do tempo: o que cada rodada ensinou. Datas e horas de inicio/fim
# vem das linhas "experimento de hiperparametros" e "execucao finalizada"
# dos logs; os numeros sao os do teste no limiar 0,5. "experimentos" liga o
# marco as pastas: se nenhuma tem log ainda, o marco aparece como pendente.
LINHA_DO_TEMPO = [
    {"data": "07/10/2026", "hora": "12:16–12:44", "titulo": "Baseline",
     "experimentos": ["baseline"],
     "numeros": "P 0,9537 · R 0,6042 · F1 0,7398 · AP 0,5963 · 27 min de treino",
     "licao": "A configuração base já supera a média do RetinaNet da tese nas quatro métricas. A estratégia de "
              "ladrilhos de 640 px, com a imagem reescalada para o bacilo mediano ter ~40 px, resolve o problema dos "
              "bacilos pequenos demais. O ponto fraco é a sensibilidade: o modelo acerta quase tudo o que marca, "
              "mas deixa 40% dos bacilos sem marcar."},
    {"data": "08/10/2026", "hora": "08:45–11:04", "titulo": "1ª rodada: um hiperparâmetro por vez",
     "experimentos": ["altoLearningRate", "baixoLearningRate", "altoEpochs", "altoWeightDecay", "altoImageSize",
                      "baixoImageSize", "altoTargetBox", "baixoTargetBox", "baixoTrainableLayers"],
     "numeros": "melhor: altoLearningRate, F1 0,7748 · AP 0,6431 · 8 de 10 configurações superam as 4 médias",
     "licao": "Mexer no learning rate foi o que mais rendeu. Ladrilho e escala fora de 640 px / 40 px pioraram, e "
              "congelar camadas ou dobrar o weight decay também. 40 épocas ganharam só +0,005 de F1 por 31% a mais "
              "de tempo. O baixoTargetBox desabou (sensibilidade 0,22): bacilos a 32 px ficam pequenos demais para "
              "as âncoras. A pista mais importante: com o limiar escolhido na validação o F1 chegava a ~0,86, então "
              "o modelo já detectava bem e o problema era a calibração dos scores."},
    {"data": "10/10/2026", "hora": "", "titulo": "Decisão: comparar só no limiar 0,5",
     "experimentos": [],
     "numeros": "alvo: P > 0,9339 · R > 0,4741 · F1 > 0,6244 · AP > 0,4533",
     "licao": "O limiar escolhido na validação dá números melhores, mas muda a régua. Para a comparação com a "
              "tese ser justa, tudo passou a ser medido no limiar 0,5, e as análises com limiar menor saíram. "
              "A partir daqui, melhorar significa calibrar o modelo para o 0,5."},
    {"data": "10/10/2026", "hora": "15:29–18:15", "titulo": "1ª busca com Optuna",
     "experimentos": ["optuna"],
     "numeros": "P 0,8571 · R 0,8697 · F1 0,8634 · AP 0,8386 · busca 140 min + treino 27 min",
     "licao": "O alpha da perda focal é a alavanca da calibração. Os 5 melhores trials usaram alpha 0,5, o maior "
              "valor permitido, contra 0,25 do padrão, e grad_clip 5. A sensibilidade saltou de 0,60 para 0,87 e o "
              "limiar ideal na validação subiu de 0,35 para 0,55: o 0,5 da tese virou praticamente o corte certo. "
              "O preço: a precisão caiu para 0,857, abaixo dos 0,934 da tese, e o AP50 caiu de 0,795 para 0,764, "
              "com caixas um pouco menos justas (fg_iou 0,4)."},
    {"data": "10/10/2026", "hora": "18:16–18:50", "titulo": "Treino com Weights & Biases",
     "experimentos": ["wandb"],
     "numeros": "P 0,9807 · R 0,4967 · F1 0,6595 · AP 0,4910 · 34 min de treino",
     "licao": "Somar duas melhorias isoladas (lr 1e-2 e 40 épocas) com lote 8 não somou os ganhos: o F1 ficou "
              "abaixo do baseline. O melhor checkpoint saiu na época 13 de 40, então mais épocas não ajudaram. "
              "O W&B rodou offline por falta de chave, o que não atrapalha o treino: os gráficos só aparecem "
              "depois de um wandb sync."},
    {"data": "10/10/2026", "hora": "19:35–22:06", "titulo": "2ª busca: focal loss e F1 no 0,5",
     "experimentos": ["optunaFocalF1"],
     "numeros": "P 0,8281 · R 0,9104 · F1 0,8673 · AP 0,8593 · busca 125 min, 12 de 20 trials podados",
     "licao": "O alpha virou o hiperparâmetro mais importante da busca (importância 0,33). O gamma pesou sete "
              "vezes menos (0,05): 1,5 e 2,0 deram resultados parecidos, e o vencedor usou 1,5, que suprime menos "
              "os exemplos fáceis e, junto com alpha 0,6, tende a empurrar os scores para cima. O alpha 0,75 não apareceu "
              "entre os 5 melhores, então o ótimo fica entre 0,5 e 0,6. Lote 8 apareceu em todos os 5 melhores. "
              "Com alpha 0,6 a calibração passou um pouco do ponto: o limiar ideal na validação foi 0,60 e a "
              "precisão caiu para 0,83."},
    {"data": "10/10/2026", "hora": "22:00–00:11 de 11/10", "titulo": "Cinco sementes da melhor configuração",
     "experimentos": ["wandbSementes"],
     "numeros": "F1 0,8553 ± 0,0091 · AP 0,8066 ± 0,0323 · P 0,8787 ± 0,0288 · R 0,8352 ± 0,0388",
     "licao": "A semente 42 repetiu exatamente o resultado da 1ª busca, prova de que o treino é reprodutível. Mas "
              "ela foi a melhor das cinco: o 0,8634 era otimista, e a média real é 0,855. O AP varia 3,5 vezes mais "
              "que o F1 entre sementes, então diferenças de AP abaixo de ~0,03 numa semente só são ruído. Mesmo "
              "assim, o F1 fica 0,18 acima do melhor dataset da tese, mais de seis desvios de distância."},
    {"data": "11/10/2026", "hora": "22:54–00:56 de 12/10", "titulo": "Cinco sementes da 2ª busca",
     "experimentos": ["wandbSementesFocalF1"],
     "numeros": "F1 0,8643 ± 0,0037 · AP 0,8414 ± 0,0229 · P 0,8459 ± 0,0172 · R 0,8844 ± 0,0256 · 24,5 min por semente",
     "licao": "A semente 42 repetiu F1 0,8673, então a configuração foi aplicada como deveria. Desta vez o ganho é real: "
              "+0,009 de F1 sobre o wandbSementes, com p = 0,032 num teste de permutação exato entre as 5 + 5 sementes, e "
              "+0,049 de sensibilidade (p = 0,040). O AP subiu 0,035, mas sem significância (p = 0,087). O desvio do F1 caiu "
              "para menos da metade (0,0037 contra 0,0091): alpha 0,6 com lote 8 é a configuração de maior F1 e a mais estável "
              "do projeto. O preço continua sendo a precisão, 0,846, abaixo dos 0,934 da tese."},
    {"data": "11/10/2026", "hora": "22:54–02:03 de 12/10", "titulo": "Superar as quatro médias ao mesmo tempo, com folga",
     "experimentos": ["optunaSuperaTese"],
     "numeros": "P 0,9629 ± 0,0067 · R 0,5261 ± 0,0388 · F1 0,6799 ± 0,0337 · AP 0,5176 ± 0,0382 · 3 de 3 sementes superam as 4",
     "licao": "Objetivo cumprido: as três sementes e a média superam as quatro métricas da tese, com +0,029 de folga na "
              "precisão. A busca voltou ao alpha 0,25 com lr ~0,011, praticamente o altoLearningRate da 1ª rodada: os três "
              "trials que passaram a tese na validação usaram alpha 0,25 ou 0,35, e learning rate e weight decay foram os "
              "hiperparâmetros mais importantes (0,41 e 0,29). Mas o F1 ficou em 0,68, abaixo dos 0,77 do altoLearningRate. "
              "O motivo é a escolha do checkpoint: ela exigia precisão de pelo menos 0,944 na validação, onde a precisão sai "
              "menor que no teste, e acabou ficando com épocas conservadoras, de sensibilidade baixa. As últimas épocas "
              "tinham F1 maior na validação (0,69 a 0,75). A curva precisão × sensibilidade do teste aponta para o mesmo "
              "lado: no limiar 0,45, as três sementes ainda superariam a tese em precisão e sensibilidade, com F1 de 0,73 "
              "a 0,79. Como isso foi visto no teste, é uma pista e não um resultado. Superar a tese nas quatro métricas é "
              "robusto; fazer isso com F1 alto pede uma folga menor na validação."},
    {"data": "12/10/2026", "hora": "16:33–18:49", "titulo": "Quatro critérios de checkpoint no mesmo treino",
     "experimentos": ["wandbSementesSuperaTese"],
     "numeros": "P 0,9605 ± 0,0095 · R 0,5401 ± 0,0364 · F1 0,6909 ± 0,0311 · AP 0,5308 ± 0,0359 · 5 de 5 sementes superam as 4",
     "licao": "Cinco sementes, cinco vezes acima da tese nas quatro métricas, com +0,027 de folga na precisão. A pergunta "
              "era se tirar a folga de 0,01 do checkpoint subia o F1. Subiu pouco: 0,6863 com folga e 0,6909 sem, dentro "
              "do desvio entre sementes (0,03). O critério folga_001 repete o optunaSuperaTese (0,6799), então aquele "
              "resultado se confirmou com mais sementes, e a folga não era o gargalo. O gargalo é escolher a época pelo "
              "máximo de uma validação pequena e ruidosa: numa mesma semente, a sensibilidade da validação foi de 0,14 na "
              "época 13 para 0,71 na 14, e a época escolhida variou de 7 a 26 entre as sementes. A última época, sem "
              "seleção nenhuma, teve o maior F1 (0,7036) e também superou a tese em 5 de 5; no fim do cosseno a taxa de "
              "aprendizado é baixa e os checkpoints se estabilizam. Os quatro critérios ficaram entre 0,686 e 0,704, "
              "sem diferença clara nesta configuração."},
    {"data": "12/10/2026", "hora": "16:51–19:07", "titulo": "Alpha 0,30 com o limiar fixo em 0,5",
     "experimentos": ["wandbSementesAlpha030"],
     "numeros": "P 0,9603 ± 0,0120 · R 0,5524 ± 0,0348 · F1 0,7007 ± 0,0266 · AP 0,5423 ± 0,0334 · 5 de 5 sementes superam as 4",
     "licao": "Com o critério principal, o alpha 0,30 não se distingue do 0,25: +0,010 de F1 (p = 0,59 num teste de "
              "permutação exato entre as 5 + 5 sementes) e a mesma precisão, 0,9603 contra 0,9605. O efeito do alpha "
              "aparece quando se tira a seleção de época. Na última época o alpha 0,30 chega a F1 0,7537 ± 0,0138, contra "
              "0,7036 do alpha 0,25 (+0,050, p = 0,016), com sensibilidade 0,6251 (+0,069, p = 0,024) e precisão 0,9496, "
              "ainda +0,016 acima da tese nas cinco sementes. É o maior F1 entre as configurações que superam as quatro "
              "métricas em várias sementes, e o desvio é a metade do critério principal. Escolher a época pelo AP_VOC é "
              "arriscado com alpha 0,30: o F1 médio é 0,7214, mas só 2 de 5 sementes superam a tese (duas perdem na "
              "precisão, uma na sensibilidade), com desvio de 0,0775. A ressalva é que a última época era um critério "
              "secundário e foi a melhor de quatro vistas no teste, então o ganho precisa ser confirmado num experimento "
              "que a declare como principal antes de rodar."},
    {"data": "14/10/2026", "hora": "17:44–21:59", "titulo": "Alpha 0,55 e a última época declarada antes",
     "experimentos": ["wandbSementesFocalF1Alpha055"],
     "numeros": "P 0,8752 ± 0,0100 · R 0,8458 ± 0,0135 · F1 0,8601 ± 0,0050 · AP 0,8109 ± 0,0114 · 10 sementes, 25,5 min cada",
     "licao": "Baixar o alpha de 0,6 para 0,55 fez o que se esperava com a precisão e não mudou o F1. Comparando o mesmo "
              "critério do wandbSementesFocalF1 (AP_VOC na validação), a precisão subiu 0,027 (0,8726 contra 0,8459, "
              "p = 0,003 num teste de permutação exato entre as 10 + 5 sementes) e a sensibilidade caiu 0,036 (p = 0,008). "
              "O F1 ficou em 0,8601 contra 0,8643 (p = 0,25), e o AP caiu 0,028 (p = 0,014). A troca entre precisão e "
              "sensibilidade foi quase de um para um, então o alpha 0,55 só move o ponto de operação na mesma curva, e a "
              "precisão continua 0,059 abaixo da tese: 0 de 10 sementes superam as quatro métricas. A última época, agora "
              "declarada antes de rodar, empatou com o AP_VOC (F1 0,8601 nos dois, p = 0,98 pareado nas mesmas sementes) e "
              "ficou um pouco acima do F1 da validação (0,8573, p = 0,10), com o menor desvio dos três (0,0050). Como a "
              "validação desta configuração é estável, escolher a época quase não importa; o ganho de 0,05 do alpha 0,30 "
              "veio de uma validação ruidosa. A última época é o critério mais simples e não perdeu para os outros em "
              "nenhum dos três experimentos que a mediram."},
]


# Linhas no formato do log da plataforma: "10/10/2026, 22:26:30: <mensagem>".
# Cada linha aparece repetida no export; o formato da plataforma (com data)
# e o mais completo, e a deduplicacao e pela linha inteira.
LINHA = re.compile(r"^\s*(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}): (.*)$")
METRICA = re.compile(r"([A-Z][A-Z0-9_]+)=([0-9.]+)")
METRICA_DP = re.compile(r"([A-Z][A-Z0-9_]+)=([0-9.]+) \(±([0-9.]+)\)")
CRITERIO_RESUMO = re.compile(r"rtn/criterio > RESUMO criterio=(\w+)( \(principal\))?: (.*?); "
                             r"sementes que superam as 4: (\d+)/(\d+); a media supera as 4: (\w+)")


def linhas_plataforma(texto):
    vistas = set()
    for bruta in texto.splitlines():
        m = LINHA.match(bruta)
        if m and bruta.strip() not in vistas:
            vistas.add(bruta.strip())
            yield datetime.strptime(m.group(1), "%d/%m/%Y, %H:%M:%S"), m.group(2)


def ler_experimento(pasta):
    log_md = pasta / "log.md"
    if not log_md.exists():
        return None  # experimento ainda sem log (ex.: rodando na plataforma)
    texto = log_md.read_text(errors="replace")
    eventos = sorted(linhas_plataforma(texto), key=lambda e: e[0])

    testes, epocas, tempos_treino, curvas = [], [], [], []
    criterios, escolhidas = [], {}
    busca = None
    for _, msg in eventos:
        if "rtn/teste > limiar 0.50 (tese):" in msg:
            testes.append({k: float(v) for k, v in METRICA.findall(msg.split("(tese):", 1)[1])})
        elif "rtn/valid > resultado da epoca" in msg:
            n = int(re.search(r"epoca (\d+):", msg).group(1))
            valores = {k: float(v) for k, v in METRICA.findall(msg)}
            if not epocas or n <= epocas[-1][-1]["epoca"]:
                epocas.append([])  # nova semente (ou novo treino) recomeca a contagem
            epocas[-1].append({"epoca": n, **{k: valores.get(k) for k in METRICAS}})
        elif re.match(r"OK: (treino|semente)", msg):
            t = re.search(r"([0-9.]+)s\s*$", msg)
            if t:
                tempos_treino.append(float(t.group(1)) / 60)
        elif msg.startswith("OK: busca"):
            t = re.search(r"([0-9.]+)s\s*$", msg)
            busca = float(t.group(1)) / 60 if t else None
        elif msg.startswith("rtn/criterio > RESUMO"):
            m = CRITERIO_RESUMO.match(msg)
            if m:
                criterios.append({
                    "nome": m.group(1), "principal": bool(m.group(2)),
                    "media": {k: float(v) for k, v, _ in METRICA_DP.findall(m.group(3))},
                    "desvio": {k: float(d) for k, _, d in METRICA_DP.findall(m.group(3))},
                    "superam": int(m.group(4)), "sementes": int(m.group(5)), "media_supera": m.group(6) == "SIM",
                })
        elif msg.startswith("rtn/criterio > semente"):
            m = re.match(r"rtn/criterio > semente \d+ criterio=(\w+)(?: \(principal\))? epoca=(\d+):", msg)
            if m:
                escolhidas.setdefault(m.group(1), []).append(int(m.group(2)))
        elif msg.startswith("rtn/curva_pr >"):
            try:
                curvas.append(json.loads(msg.split(": ", 1)[1].rsplit(" - ", 1)[0]))
            except (ValueError, IndexError):
                pass
    if not testes:
        return None
    ordem = ["sem_folga", "folga_001", "ap_voc", "f1_val", "ultima"]
    criterios.sort(key=lambda c: ordem.index(c["nome"]) if c["nome"] in ordem else len(ordem))
    for c in criterios:
        c["epocas"] = escolhidas.get(c["nome"], [])
    nome = pasta.name.removeprefix("retinanet-")
    if nome in HIPER_MUDANCA:
        hiper, origem = {**HIPER_BASE, **HIPER_MUDANCA[nome]}, "tabela"
    else:
        lidos = hiper_do_log(eventos)
        hiper, origem = {**HIPER_BASE, **lidos}, ("log" if lidos else "desconhecida")
    return {
        "hiper": hiper,
        "hiper_origem": origem,
        "nome": nome,
        "rodada": "1ª rodada" if nome in PRIMEIRA_RODADA else "buscas e sementes",
        "sementes": len(testes) > 1,
        "testes": testes,
        "epocas": epocas,
        "tempo_treino": statistics.mean(tempos_treino) if tempos_treino else None,
        "tempo_busca": busca,
        "curvas": curvas,
        "criterios": criterios,
    }


def resumir(valores):
    v = [x for x in valores if x is not None]
    return {
        "media": statistics.mean(v),
        "desvio": statistics.stdev(v) if len(v) > 1 else 0.0,
        "min": min(v),
        "max": max(v),
        "n": len(v),
    }


def montar_dados():
    experimentos = [e for p in sorted(EXPERIMENTOS.iterdir()) if p.is_dir() and (e := ler_experimento(p))]
    ref = {k: statistics.mean(d[k][0] for d in TESE.values()) for k in METRICAS}
    for e in experimentos:
        e["resumo"] = {k: resumir([t.get(k) for t in e["testes"]])
                       for k in METRICAS + ["AP_VOC", "AP50", "PREDICTIONS", "GROUND_TRUTH"]}
        medias = {k: e["resumo"][k]["media"] for k in METRICAS}
        e["supera_media"] = all(medias[k] > ref[k] for k in METRICAS)
        e["rodadas_que_superam"] = sum(all(t[k] > ref[k] for k in METRICAS) for t in e["testes"])
    experimentos.sort(key=lambda e: e["resumo"]["F1"]["media"], reverse=True)
    return {
        "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "tese": {ds: {k: {"media": v[0], "desvio": v[1]} for k, v in d.items()} for ds, d in TESE.items()},
        "ref": ref,
        "hiper_base": HIPER_BASE,
        "linha_do_tempo": LINHA_DO_TEMPO,
        "hiper_fixos": HIPER_FIXOS,
        "experimentos": experimentos,
    }


README = RAIZ / "README.md"
MARCA_INICIO = "<!-- linha-do-tempo:inicio -->"
MARCA_FIM = "<!-- linha-do-tempo:fim -->"


def linha_do_tempo_markdown(dados):
    """A mesma linha do tempo do painel, em Markdown, para o README."""
    com_log = {e["nome"] for e in dados["experimentos"]}
    blocos = []
    for m in dados["linha_do_tempo"]:
        if not m["experimentos"]:
            estado, rotulo = "decisão", "Por quê"
        elif any(n in com_log for n in m["experimentos"]):
            estado, rotulo = "log auditado", "O que ensinou"
        else:
            estado, rotulo = "aguardando log", "Por que foi criado"
        hora = f" ({m['hora']})" if m["hora"] else ""
        exps = " · ".join(f"`{n}`" for n in m["experimentos"])
        # "\\" no fim da linha e a quebra de linha do GitHub dentro de um item de lista.
        linhas = [f"- **{m['data']}{hora} · {m['titulo']}** · _{estado}_"]
        if exps:
            linhas.append(f"  {exps}")
        linhas += [f"  `{m['numeros']}`", f"  **{rotulo}:** {m['licao']}"]
        blocos.append(" \\\n".join(linhas))
    return "\n\n".join(blocos)


def atualizar_readme(dados):
    """Reescreve o trecho entre as marcas do README; o resto fica intacto."""
    if not README.exists():
        return
    texto = README.read_text()
    if MARCA_INICIO not in texto or MARCA_FIM not in texto:
        print(f"aviso: {README} sem as marcas da linha do tempo; nada alterado")
        return
    antes, resto = texto.split(MARCA_INICIO, 1)
    _, depois = resto.split(MARCA_FIM, 1)
    README.write_text(f"{antes}{MARCA_INICIO}\n{linha_do_tempo_markdown(dados)}\n{MARCA_FIM}{depois}")
    print(f"ok {README} (linha do tempo)")


def main():
    dados = montar_dados()
    modelo = (Path(__file__).resolve().parent / "modelo.html").read_text()
    pagina = modelo.replace("/*__DADOS__*/null", json.dumps(dados, ensure_ascii=False))
    SAIDA.write_text(
        "<!doctype html>\n<html lang=\"pt-BR\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "</head>\n<body>\n" + pagina + "\n</body>\n</html>\n"
    )
    print(f"ok {SAIDA} ({len(dados['experimentos'])} experimentos)")
    atualizar_readme(dados)
    if "--fragmento" in sys.argv:
        destino = Path(sys.argv[sys.argv.index("--fragmento") + 1])
        destino.write_text(pagina)
        print(f"ok {destino}")


if __name__ == "__main__":
    main()
