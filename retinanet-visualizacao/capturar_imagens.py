"""Tira as imagens do painel usadas no README (pasta imagens/).

O GitHub mostra o README, mas nao renderiza o HTML do painel; estas capturas
sao a forma de o painel aparecer la. Rode depois de gerar_visualizacao.py.

Precisa do Playwright com Chromium (fora da biblioteca padrao):
    pip install playwright && playwright install chromium
    python3 capturar_imagens.py

Para usar um Chromium ja instalado, passe o caminho em RETINA_CHROMIUM.
"""
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

PASTA = Path(__file__).resolve().parent
PAINEL = PASTA / "retinanet-visualizacao.html"
SAIDA = PASTA / "imagens"

# (seletor da secao no painel, nome do arquivo)
SECOES = [
    ("header.topo", "01-resumo"),
    ("#superacao", "02-folga-sobre-a-tese"),
    ("#pr", "03-precisao-sensibilidade"),
    ("#faixas", "04-candlestick"),
    ("#matriz", "05-matriz-de-confusao"),
    ("#aprendizado", "06-curva-de-aprendizado"),
]


def main():
    SAIDA.mkdir(exist_ok=True)
    with sync_playwright() as p:
        executavel = os.getenv("RETINA_CHROMIUM")
        navegador = p.chromium.launch(executable_path=executavel) if executavel else p.chromium.launch()
        pagina = navegador.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1.5, color_scheme="light")
        erros = []
        pagina.on("pageerror", lambda e: erros.append(str(e)))
        pagina.goto(PAINEL.as_uri(), wait_until="networkidle", timeout=90000)
        # Plotly desenha depois do carregamento; espera todos os graficos.
        pagina.wait_for_function("window.Plotly && document.querySelectorAll('.js-plotly-plot').length >= 9", timeout=60000)
        pagina.wait_for_timeout(1500)
        for seletor, nome in SECOES:
            pagina.locator(seletor).screenshot(path=str(SAIDA / f"{nome}.png"))
            print("ok", SAIDA / f"{nome}.png")
        navegador.close()
    if erros:
        raise SystemExit(f"erros de JavaScript no painel: {erros}")


if __name__ == "__main__":
    main()
