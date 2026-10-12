"""Gera o painel de diagnostico dos experimentos RetinaNet (retinanet-visualizacao.html).

Le os logs completos (resultados-*.md) de cada pasta em retinanet-experimentos/,
sem dependencias alem da biblioteca padrao. Experimentos novos entram sozinhos:
basta a pasta ter o log completo.

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
    {"data": "11/10/2026", "hora": "22:00 (10/10)–00:11", "titulo": "Cinco sementes da melhor configuração",
     "experimentos": ["wandbSementes"],
     "numeros": "F1 0,8553 ± 0,0091 · AP 0,8066 ± 0,0323 · P 0,8787 ± 0,0288 · R 0,8352 ± 0,0388",
     "licao": "A semente 42 repetiu exatamente o resultado da 1ª busca, prova de que o treino é reprodutível. Mas "
              "ela foi a melhor das cinco: o 0,8634 era otimista, e a média real é 0,855. O AP varia 3,5 vezes mais "
              "que o F1 entre sementes, então diferenças de AP abaixo de ~0,03 numa semente só são ruído. Mesmo "
              "assim, o F1 fica 0,18 acima do melhor dataset da tese, mais de seis desvios de distância."},
    {"data": "11/10/2026", "hora": "", "titulo": "Confirmar a 2ª busca com sementes",
     "experimentos": ["wandbSementesFocalF1"],
     "numeros": "referência: wandbSementes, F1 0,8553 ± 0,0091",
     "licao": "O ganho da 2ª busca sobre a 1ª (+0,004 de F1, +0,021 de AP) é menor que o desvio entre sementes. "
              "Só a média de cinco sementes diz se alpha 0,6 com lote 8 é melhor de fato. A semente 42 deve repetir "
              "F1 0,8673, o que confirma que a configuração foi aplicada."},
    {"data": "11/10/2026", "hora": "", "titulo": "Superar as quatro médias ao mesmo tempo, com folga",
     "experimentos": ["optunaSuperaTese"],
     "numeros": "9 de 14 experimentos já superam as 4 médias; os do Optuna falham só na precisão",
     "licao": "Otimizar só o F1 levou a busca a trocar precisão por sensibilidade até a precisão cair abaixo da "
              "tese. A busca seguinte exige as quatro métricas acima da tese com folga de 0,01 e só então maximiza "
              "o F1, explorando alpha entre 0,25 e 0,5, a faixa entre a precisão alta da 1ª rodada e a "
              "sensibilidade alta do Optuna."},
]


# Linhas no formato do log da plataforma: "10/10/2026, 22:26:30: <mensagem>".
# Cada linha aparece repetida no export; o formato da plataforma (com data)
# e o mais completo, e a deduplicacao e pela linha inteira.
LINHA = re.compile(r"^\s*(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2}): (.*)$")
METRICA = re.compile(r"([A-Z][A-Z0-9_]+)=([0-9.]+)")


def linhas_plataforma(texto):
    vistas = set()
    for bruta in texto.splitlines():
        m = LINHA.match(bruta)
        if m and bruta.strip() not in vistas:
            vistas.add(bruta.strip())
            yield datetime.strptime(m.group(1), "%d/%m/%Y, %H:%M:%S"), m.group(2)


def ler_experimento(pasta):
    arquivos = list(pasta.glob("resultados-*.md"))
    if not arquivos:
        return None
    texto = max(arquivos, key=lambda a: a.stat().st_size).read_text(errors="replace")
    eventos = sorted(linhas_plataforma(texto), key=lambda e: e[0])

    testes, epocas, tempos_treino, curvas = [], [], [], []
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
        elif msg.startswith("rtn/curva_pr >"):
            try:
                curvas.append(json.loads(msg.split(": ", 1)[1].rsplit(" - ", 1)[0]))
            except (ValueError, IndexError):
                pass
    if not testes:
        return None
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
    if "--fragmento" in sys.argv:
        destino = Path(sys.argv[sys.argv.index("--fragmento") + 1])
        destino.write_text(pagina)
        print(f"ok {destino}")


if __name__ == "__main__":
    main()
