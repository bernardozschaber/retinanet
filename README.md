<table style="margin: auto; background-color: white;">
  <tr>
    <td style="background-color: white;">
      <img src='https://drive.google.com/uc?export=view&id=1lgflViz1uefcvVW1iI57haB4M1bKsZtp' alt="FutureLab" width="200" />
    </td>
    <td style="background-color: white;">
      <img src='https://drive.google.com/uc?export=view&id=1R6PphT9jmd2vikODFPf6cW54QtZ29o2a' alt="Flautim" width="200" />
    </td>
    <td style="background-color: white;">
      <img src='https://drive.google.com/uc?export=view&id=1lgflViz1uefcvVW1iI57haB4M1bKsZtp' alt="FutureLab" width="200" />
    </td>
    <td style="background-color: white;">
      <img src='https://drive.google.com/uc?export=view&id=1R6PphT9jmd2vikODFPf6cW54QtZ29o2a' alt="Flautim" width="200" />
    </td>
    <td style="background-color: white;">
      <img src='https://drive.google.com/uc?export=view&id=1lgflViz1uefcvVW1iI57haB4M1bKsZtp' alt="FutureLab" width="200" />
    </td>
    <td style="background-color: white;">
      <img src='https://drive.google.com/uc?export=view&id=1R6PphT9jmd2vikODFPf6cW54QtZ29o2a' alt="Flautim" width="200" />
    </td>
    <td style="background-color: white;">
      <img src='https://drive.google.com/uc?export=view&id=1lgflViz1uefcvVW1iI57haB4M1bKsZtp' alt="FutureLab" width="200" />
    </td>
  </tr>
</table>

# RetinaNet para detecção de bacilos de Koch

Experimentos de hiperparâmetros com o RetinaNet R50-FPN v2 para detectar bacilos da tuberculose em lâminas coradas por Ziehl-Neelsen, executados na **[Plataforma Flautim](http://flautim.futurelab.dcc.ufmg.br/)** do FutureLab (DCC/UFMG). O objetivo é superar o RetinaNet da tese de Thales Carvalho no mesmo protocolo de avaliação, e entender por que cada mudança ajuda ou atrapalha.

## 🎯 Objetivo

O ponto de partida foi a tabela da tese (`resultados-thales-todos-modelos.md`). Na média dos quatro datasets dela, o RetinaNet chega a:

| Precisão | Sensibilidade | F-measure | AP |
|---|---|---|---|
| 0,9339 | 0,4741 | 0,6244 | 0,4533 |

**Atalhos:** [🧭 Painel de diagnóstico](https://bernardozschaber.me/retinanet/retinanet-visualizacao/retinanet-visualizacao.html) · [🏆 Resultados](#-resultados-até-agora) · [🕒 Linha do tempo](#-linha-do-tempo) · [📚 Tutoriais da Flautim](#-tutoriais-da-flautim)

---

## 🏗️ Arquitetura

O detector existe uma vez só, em `retinanet-base/`, e cada experimento é só um `run.py` que monta as três classes da base com outros hiperparâmetros. A separação vem da plataforma Flautim, que executa os experimentos: ela copia os arquivos para uma pasta só, sem subpastas, e roda `python3 run.py` dali. Por isso os imports são planos (`import RetinaDataset`), o ponto de entrada sempre se chama `run.py`, e um experimento novo nunca copia o detector.

Os logs ficam no repositório de propósito. Eles são a fonte de verdade: o painel e os resultados deste README saem deles, com data e hora de cada treino.

```
retinanet-base/                 detector: dataset em ladrilhos, modelo e ciclo de treino/avaliação
retinanet-experimentos/
  retinanet-<variação>/         um experimento: run.py, o .md do que mudou e o log da Flautim
                                  1ª rodada: baseline e 9 variações de um hiperparâmetro
                                  optuna*: buscas automáticas; wandb*: treinos acompanhados no W&B
                                  *Sementes: a mesma configuração com várias sementes (média ± desvio)
retinanet-visualizacao/         painel gerado a partir dos logs, imagens do README e scripts
resultados-thales-todos-modelos.md   tabelas da tese usadas como referência
```

### 🧩 Como o projeto se encaixa na Flautim

Todo projeto Flautim estende a interface de programação da plataforma, a `flautim_api`. A figura, dos [tutoriais da Flautim](https://github.com/FutureLab-DCC/flautim_tutoriais), mostra o esquema. Aqui ele vira: `RetinaDataset` estende `Dataset`, `RetinaModel` estende `Model`, `RetinaExperiment` estende o `Experiment` centralizado, e o `run.py` de cada experimento usa o `common` para iniciar a plataforma e registrar o log.

<div align="center">
  <img src='https://drive.google.com/uc?export=view&id=1QOI4jWrwS979xhW_wlGzkPa2bMA-giuc' alt="Interface de programação da plataforma Flautim" width="800" />
</div>

### 📐 Diagrama de classes

As três classes da base estendem as classes da Flautim. Os experimentos que precisam de algo a mais (enviar métricas ao W&B, escolher o checkpoint pelo critério da tese) estendem o `RetinaExperiment` dentro do próprio `run.py`, sem tocar na base.

```mermaid
classDiagram
    direction TB
    class FlautimDataset {
        <<Flautim>>
    }
    class FlautimModel {
        <<Flautim>>
    }
    class FlautimExperiment {
        <<Flautim>>
        +run(metrics)
    }
    class TorchDataset {
        <<PyTorch>>
    }
    class RetinaDataset {
        +batch_size
        +tile_size
        +escala
        +particoes
        +dataloader(validation, test)
        +train()
        +validation()
        +test()
        +manifest()
    }
    class RetinaTorchDataset {
        +nomes
        +tile_size
        +crops_per_image
        +treinando
        +carregar(nome)
        +ladrilhos(imagem)
        +#95;#95;getitem#95;#95;(indice)
    }
    class CacheDeImagens {
        +escala
        +imagens
        +ler(nome, caminho)
    }
    class RetinaModel {
        +detector
        +architecture
        +tile_size
        +TAMANHOS_ANCORA
        +forward(images, targets)
        +save()
    }
    class RetinaExperiment {
        +epochs
        +optimizer
        +score_threshold
        +eval_iou
        +best_score
        +model_path
        +training_loop(data_loader)
        +validation_loop(data_loader)
        +detectar_imagem(particao, imagem)
        +inferir_particao(particao, nome)
        +metricas(registros, limiar)
        +melhor_limiar(registros)
        +evaluate_test_set()
    }
    class Metricas {
        <<module>>
        +casar_guloso(sobreposicoes, limiar_iou)
        +prf_no_limiar(scores, acertos, total, limiar)
        +ap_tese(scores, acertos, total, corte)
        +ap_voc(scores, acertos, total)
    }
    class RetinaExperimentWandb {
        +wandb_run
        +training_loop(data_loader)
        +validation_loop(data_loader)
        +evaluate_test_set()
    }
    class RetinaExperimentTese {
        +melhor_obj
        +curva_pr
        +validation_loop(data_loader)
        +evaluate_test_set()
    }

    FlautimDataset <|-- RetinaDataset
    TorchDataset <|-- RetinaTorchDataset
    FlautimModel <|-- RetinaModel
    FlautimExperiment <|-- RetinaExperiment
    RetinaExperiment <|-- RetinaExperimentWandb : run.py dos experimentos wandb*
    RetinaExperiment <|-- RetinaExperimentTese : run.py do optunaSuperaTese
    RetinaDataset "1" *-- "3" RetinaTorchDataset : treino, validação, teste
    RetinaTorchDataset "*" --> "1" CacheDeImagens : compartilham
    RetinaExperiment --> RetinaModel
    RetinaExperiment --> RetinaDataset
    RetinaExperiment ..> Metricas : usa
```

### 🔄 Diagrama de sequência (um experimento na Flautim)

```mermaid
sequenceDiagram
    actor Pesquisador
    participant Flautim
    participant Run as run.py
    participant Dataset as RetinaDataset
    participant Modelo as RetinaModel
    participant Exp as RetinaExperiment

    Pesquisador->>Flautim: envia retinanet-base/*.py + run.py (+ requirements.txt)
    Flautim->>Run: pip install -r requirements.txt, python3 run.py
    Run->>Flautim: fl.init()
    Run->>Dataset: constrói (Annotation.csv, escala para bacilo de ~40 px, cache em RAM, split 139/20/40)
    opt experimentos optuna*
        loop cada trial
            Run->>Modelo: constrói com os hiperparâmetros do trial
            Run->>Exp: training_loop() e inferir_particao(validação), época a época
            Exp-->>Run: objetivo na validação (poda os trials ruins)
        end
    end
    Run->>Modelo: constrói (pesos COCO, âncoras de 16 px, cabeça de 2 classes)
    Run->>Exp: run()
    loop cada época
        Exp->>Dataset: recortes de 640 px com augmentation
        Exp->>Exp: training_loop() (SGD, AMP, aquecimento + cosseno)
        Exp->>Exp: validation_loop() (ladrilhos, métricas, checkpoint)
    end
    Run->>Exp: evaluate_test_set()
    Exp-->>Flautim: métricas no limiar 0,5, no log (rtn/teste)
    Pesquisador->>Pesquisador: cola o log completo em resultados-*.md
```

### 🧱 Diagrama de componentes

```mermaid
flowchart LR
    subgraph Flautim[Plataforma Flautim]
        Vol[(Volume do dataset<br/>Annotation.csv + images/)] --> Treino[run.py + retinanet-base<br/>Quadro RTX 6000]
    end
    Treino -->|log completo| Logs[resultados-*.md<br/>retinanet-experimentos/]
    Logs --> Gerador[gerar_visualizacao.py]
    Modelo[modelo.html] --> Gerador
    Gerador --> Painel[retinanet-visualizacao.html]
    Gerador -->|linha do tempo| Readme[README.md]
    Painel --> Captura[capturar_imagens.py<br/>Playwright]
    Captura -->|imagens/| Readme
    Painel --> Pages[GitHub Pages<br/>bernardozschaber.me/retinanet]
    Treino -. opcional .-> WandB[(Weights & Biases)]
```

---

## 🔬 O problema e a estratégia

As 200 lâminas do `Annotation.csv` têm até 2560 × 2557 px e 3145 caixas anotadas (2941 bacilos e 204 aglomerados, tratados como uma classe só). O bacilo mediano tem ~72 px de lado. Encolher a lâmina inteira para caber na rede deixaria os bacilos com poucos pixels, então o `RetinaDataset` faz o contrário:

1. reescala cada lâmina para o bacilo mediano ficar com ~40 px (fator 0,5533 neste dataset);
2. treina com recortes de 640 × 640 px, a maioria centrados num bacilo, com espelhamentos, rotações e variação de cor da coloração;
3. avalia a lâmina inteira em ladrilhos com sobreposição e junta as detecções com NMS.

O split é por lâmina, com semente fixa: 139 de treino, 20 de validação e 40 de teste (614 bacilos). Uma das 200 imagens é ilegível e fica de fora.

---

## 📏 Protocolo de avaliação

Para os números serem comparáveis com a tese, o `RetinaExperiment` reproduz o notebook do Thales:

- uma detecção acerta se tem IoU ≥ 0,1 com um bacilo ainda não casado, em ordem de score (o bacilo é fino e a anotação é imprecisa, então IoU 0,5 puniria caixas corretas);
- Precisão, Sensibilidade e F-measure são medidas no limiar de score **0,5**;
- o AP (`AP_TESE`) varre os limiares de 0,20 a 0,85.

A comparação com a tese é sempre no limiar 0,5. O limiar escolhido na validação dá números melhores, mas muda a régua. O log também traz AP_VOC e as métricas com IoU 0,5 (AP50, F1_IOU50), que a tese não reporta.

---

## 🚀 Rodar um experimento na Flautim

Envie para a plataforma, juntos:

- os três módulos de `retinanet-base/`;
- o `run.py` da pasta do experimento;
- o `requirements.txt` da pasta, se existir. Ele substitui o da base, que é propositalmente vazio.

O dataset é lido de `/home/flautim_workplace/datasets/db_tb/Dataset/` (ou da mesma árvore sob `/mnt`, conforme o cluster). Quase todo hiperparâmetro pode ser trocado por variável de ambiente sem editar código, por exemplo `RETINA_LR`, `RETINA_EPOCHS`, `RETINA_BATCH_SIZE` e `RETINA_SEEDS`; cada `run.py` lista as suas no topo.

Não acrescente pacotes ao `requirements.txt` sem travar `numpy==1.26.4`. A imagem da plataforma (`experiments-gpu:0.2`, Python 3.12) foi compilada contra o NumPy 1.x, e um pacote que puxe o NumPy 2 quebra torch, cv2 e pycocotools de uma vez. Foi o que derrubou um treino em 17/09/2026, e é por isso que o dataset não usa albumentations.

Um treino de 30 épocas leva de 20 a 36 minutos numa Quadro RTX 6000 e usa menos de 5 GB de VRAM. As buscas com Optuna levam de 2 a 3 horas.

---

## 🧪 Criar um experimento novo

1. Copie a pasta do experimento mais parecido com o que você quer testar e renomeie no padrão `retinanet-<variacaoEmCamelCase>`.
2. Mude os valores padrão no `run.py` e diga no `.md` o que mudou e por quê.
3. Depois de rodar, cole o **log completo** da plataforma em `resultados-retinanet-<nome>.md`. Só as últimas linhas não bastam: o tempo de treino e as curvas por época estão no meio do log.
4. Rode `python3 gerar_visualizacao.py` em `retinanet-visualizacao/`. O experimento entra sozinho no painel e na linha do tempo deste README. Para atualizar as imagens do README, rode também `python3 capturar_imagens.py`.

Os experimentos com Optuna e com sementes ajustam a perda focal (`focal_alpha`, `focal_gamma`) e o matcher de âncoras (`fg_iou`) dentro do próprio `run.py`, sem tocar no `RetinaModel.py`. Copie essas funções junto se o seu experimento precisar delas.

---

## 📊 Painel de diagnóstico

O painel junta todos os experimentos numa página: **[abrir o painel](https://bernardozschaber.me/retinanet/retinanet-visualizacao/retinanet-visualizacao.html)**. Ele também abre direto do arquivo `retinanet-visualizacao/retinanet-visualizacao.html`, e funciona nos temas claro e escuro.

As cores seguem a coloração de Ziehl-Neelsen, o método padrão do diagnóstico da tuberculose: **fucsina**, a cor em que o bacilo fica, marca este trabalho; **azul de metileno**, a contracoloração do fundo da lâmina, marca a tese.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="retinanet-visualizacao/imagens/01-resumo-escuro.png">
  <img src="retinanet-visualizacao/imagens/01-resumo-claro.png" alt="Resumo do painel: 10 de 16 experimentos superam as quatro médias da tese">
</picture>

**Folga sobre a tese.** Cada experimento contra cada uma das quatro métricas, como diferença para a média do RetinaNet da tese. É a resposta direta à pergunta do projeto: quem supera a tese, em quê e por quanto. A marca "4/4" indica quem supera as quatro ao mesmo tempo.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="retinanet-visualizacao/imagens/02-folga-sobre-a-tese-escuro.png">
  <img src="retinanet-visualizacao/imagens/02-folga-sobre-a-tese-claro.png" alt="Mapa de calor da folga de cada experimento sobre a média da tese">
</picture>

**Ponto de operação.** Precisão contra sensibilidade no limiar 0,5. A área sombreada é onde as duas superam a tese. O gráfico mostra a troca que os experimentos com Optuna fizeram: mais sensibilidade, à custa de precisão abaixo da tese.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="retinanet-visualizacao/imagens/03-precisao-sensibilidade-escuro.png">
  <img src="retinanet-visualizacao/imagens/03-precisao-sensibilidade-claro.png" alt="Precisão × sensibilidade de cada execução no limiar 0,5">
</picture>

**Mínimo, média e máximo.** Um candlestick por grupo: os 4 datasets da tese, as 10 configurações da 1ª rodada e cada experimento com várias sementes. O pavio vai do mínimo ao máximo e o corpo cobre média ± desvio, então dá para comparar a variação, e não só a média.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="retinanet-visualizacao/imagens/04-candlestick-escuro.png">
  <img src="retinanet-visualizacao/imagens/04-candlestick-claro.png" alt="Candlestick de cada métrica por grupo de execuções">
</picture>

**Matriz de confusão da detecção.** Verdadeiros positivos, falsos positivos e bacilos perdidos de cada execução. Em detecção de objetos não existe verdadeiro negativo contável, porque o fundo da lâmina não é uma lista de objetos; por isso a curva ROC não se aplica, e o painel usa a curva precisão × sensibilidade no lugar dela.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="retinanet-visualizacao/imagens/05-matriz-de-confusao-escuro.png">
  <img src="retinanet-visualizacao/imagens/05-matriz-de-confusao-claro.png" alt="Matriz de confusão da detecção para as cinco sementes">
</picture>

**Curva precisão × sensibilidade no teste.** A curva completa de cada semente do optunaSuperaTese, com o limiar indo de 0,05 a 0,95 e o ponto do limiar 0,5 marcado, ao lado dos pontos da tese. Mostra quanto de sensibilidade o modelo ganharia baixando o limiar, e onde ele deixaria de superar a tese em precisão.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="retinanet-visualizacao/imagens/07-curva-pr-teste-escuro.png">
  <img src="retinanet-visualizacao/imagens/07-curva-pr-teste-claro.png" alt="Curva precisão × sensibilidade no teste para as sementes do optunaSuperaTese">
</picture>

**Curva de aprendizado.** A métrica da validação época a época, com a faixa entre a melhor e a pior semente. Mostra quando o treino estabiliza e quanto as sementes divergem.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="retinanet-visualizacao/imagens/06-curva-de-aprendizado-escuro.png">
  <img src="retinanet-visualizacao/imagens/06-curva-de-aprendizado-claro.png" alt="Curva de aprendizado na validação">
</picture>

O painel traz ainda o custo em tempo de treino, a tabela completa, os hiperparâmetros de cada treino e a linha do tempo que está no fim deste README.

### 🔁 Atualizar o painel

`retinanet-visualizacao/gerar_visualizacao.py` lê os logs e gera o painel, sem nenhum pacote além da biblioteca padrão do Python. O mesmo script reescreve a linha do tempo deste README. O visual fica em `modelo.html`: para mudar o painel, edite o modelo e gere de novo, porque o HTML gerado é sobrescrito a cada execução. As imagens acima saem de `capturar_imagens.py`, que precisa do Playwright com Chromium. Cada uma tem duas versões com fundo transparente, uma do tema claro e outra do tema escuro do painel, e o GitHub mostra a que combina com o tema de quem lê.

---

## 🏆 Resultados até agora

No teste, limiar 0,5. As linhas com ± são médias de várias sementes, que é a forma justa de comparar com a tese.

| Execução | Precisão | Sensibilidade | F-measure | AP | Supera as 4 médias? |
|---|---|---|---|---|---|
| Tese, RetinaNet (média dos 4 datasets) | 0,9339 | 0,4741 | 0,6244 | 0,4533 | referência |
| Baseline | 0,9537 | 0,6042 | 0,7398 | 0,5963 | sim |
| altoLearningRate (1ª rodada) | 0,9355 | 0,6612 | 0,7748 | 0,6431 | sim, por +0,0016 na precisão |
| **optunaSuperaTese** (3 sementes) | **0,9629 ± 0,0067** | 0,5261 ± 0,0388 | 0,6799 ± 0,0337 | 0,5176 ± 0,0382 | **sim, nas 3 sementes** |
| wandbSementes (5 sementes) | 0,8787 ± 0,0288 | 0,8352 ± 0,0388 | 0,8553 ± 0,0091 | 0,8066 ± 0,0323 | não (precisão) |
| **wandbSementesFocalF1** (5 sementes) | 0,8459 ± 0,0172 | **0,8844 ± 0,0256** | **0,8643 ± 0,0037** | **0,8414 ± 0,0229** | não (precisão) |

Os achados principais:

- **Superar a tese nas quatro métricas ao mesmo tempo é robusto.** O optunaSuperaTese faz isso nas três sementes e na média, com +0,029 de folga na precisão, a métrica que sempre foi o limite. Dos 16 experimentos rodados até 12/10/2026, 10 superam as quatro médias.
- **Existe uma troca entre precisão e F1, e o alpha da perda focal escolhe o ponto.** Com alpha 0,25 o modelo marca pouco e quase sempre certo, e supera a tese em tudo; com alpha 0,5 a 0,6 o F1 sobe para ~0,86, mas a precisão cai abaixo da tese. O gamma pesa sete vezes menos que o alpha.
- **O maior F1 do projeto é estável e estatisticamente confirmado.** O wandbSementesFocalF1 tem F1 0,8643 ± 0,0037, acima do wandbSementes com p = 0,032 num teste de permutação exato entre as 5 + 5 sementes.
- **Uma semente só engana.** Na primeira repetição com sementes, o resultado de uma execução isolada (F1 0,8634) era o melhor de cinco, e a média real foi 0,855. Diferenças de AP abaixo de ~0,03 entre execuções de uma semente são ruído.
- **A escolha do checkpoint importa tanto quanto os hiperparâmetros.** O optunaSuperaTese tem quase a mesma configuração do altoLearningRate, mas escolheu o checkpoint exigindo precisão de pelo menos 0,944 na validação. Isso garantiu a precisão no teste à custa de sensibilidade: o F1 ficou em 0,68 (média de três sementes), contra 0,77 do altoLearningRate (uma semente). A curva precisão × sensibilidade do teste reforça isso: no limiar 0,45, as três sementes ainda passariam a tese em precisão e sensibilidade, com F1 de 0,73 a 0,79. Por ter sido observado no teste, isso serve de pista para o próximo experimento, não de resultado.

O próximo passo natural é juntar as duas coisas: a configuração de alpha 0,25 do optunaSuperaTese com uma folga menor (ou nenhuma) na escolha do checkpoint, para tentar superar a tese nas quatro métricas com F1 perto de 0,77.


---

## 🕒 Linha do tempo

O que cada rodada ensinou, em ordem cronológica. Os horários de início e fim saem dos logs, e os números são os do teste no limiar 0,5. Este trecho é gerado por `gerar_visualizacao.py`; edite a lista `LINHA_DO_TEMPO` no script, não o texto abaixo.

<!-- linha-do-tempo:inicio -->
- **07/10/2026 (12:16–12:44) · Baseline** · _log auditado_ \
  `baseline` \
  `P 0,9537 · R 0,6042 · F1 0,7398 · AP 0,5963 · 27 min de treino` \
  **O que ensinou:** A configuração base já supera a média do RetinaNet da tese nas quatro métricas. A estratégia de ladrilhos de 640 px, com a imagem reescalada para o bacilo mediano ter ~40 px, resolve o problema dos bacilos pequenos demais. O ponto fraco é a sensibilidade: o modelo acerta quase tudo o que marca, mas deixa 40% dos bacilos sem marcar.

- **08/10/2026 (08:45–11:04) · 1ª rodada: um hiperparâmetro por vez** · _log auditado_ \
  `altoLearningRate` · `baixoLearningRate` · `altoEpochs` · `altoWeightDecay` · `altoImageSize` · `baixoImageSize` · `altoTargetBox` · `baixoTargetBox` · `baixoTrainableLayers` \
  `melhor: altoLearningRate, F1 0,7748 · AP 0,6431 · 8 de 10 configurações superam as 4 médias` \
  **O que ensinou:** Mexer no learning rate foi o que mais rendeu. Ladrilho e escala fora de 640 px / 40 px pioraram, e congelar camadas ou dobrar o weight decay também. 40 épocas ganharam só +0,005 de F1 por 31% a mais de tempo. O baixoTargetBox desabou (sensibilidade 0,22): bacilos a 32 px ficam pequenos demais para as âncoras. A pista mais importante: com o limiar escolhido na validação o F1 chegava a ~0,86, então o modelo já detectava bem e o problema era a calibração dos scores.

- **10/10/2026 · Decisão: comparar só no limiar 0,5** · _decisão_ \
  `alvo: P > 0,9339 · R > 0,4741 · F1 > 0,6244 · AP > 0,4533` \
  **Por quê:** O limiar escolhido na validação dá números melhores, mas muda a régua. Para a comparação com a tese ser justa, tudo passou a ser medido no limiar 0,5, e as análises com limiar menor saíram. A partir daqui, melhorar significa calibrar o modelo para o 0,5.

- **10/10/2026 (15:29–18:15) · 1ª busca com Optuna** · _log auditado_ \
  `optuna` \
  `P 0,8571 · R 0,8697 · F1 0,8634 · AP 0,8386 · busca 140 min + treino 27 min` \
  **O que ensinou:** O alpha da perda focal é a alavanca da calibração. Os 5 melhores trials usaram alpha 0,5, o maior valor permitido, contra 0,25 do padrão, e grad_clip 5. A sensibilidade saltou de 0,60 para 0,87 e o limiar ideal na validação subiu de 0,35 para 0,55: o 0,5 da tese virou praticamente o corte certo. O preço: a precisão caiu para 0,857, abaixo dos 0,934 da tese, e o AP50 caiu de 0,795 para 0,764, com caixas um pouco menos justas (fg_iou 0,4).

- **10/10/2026 (18:16–18:50) · Treino com Weights & Biases** · _log auditado_ \
  `wandb` \
  `P 0,9807 · R 0,4967 · F1 0,6595 · AP 0,4910 · 34 min de treino` \
  **O que ensinou:** Somar duas melhorias isoladas (lr 1e-2 e 40 épocas) com lote 8 não somou os ganhos: o F1 ficou abaixo do baseline. O melhor checkpoint saiu na época 13 de 40, então mais épocas não ajudaram. O W&B rodou offline por falta de chave, o que não atrapalha o treino: os gráficos só aparecem depois de um wandb sync.

- **10/10/2026 (19:35–22:06) · 2ª busca: focal loss e F1 no 0,5** · _log auditado_ \
  `optunaFocalF1` \
  `P 0,8281 · R 0,9104 · F1 0,8673 · AP 0,8593 · busca 125 min, 12 de 20 trials podados` \
  **O que ensinou:** O alpha virou o hiperparâmetro mais importante da busca (importância 0,33). O gamma pesou sete vezes menos (0,05): 1,5 e 2,0 deram resultados parecidos, e o vencedor usou 1,5, que suprime menos os exemplos fáceis e, junto com alpha 0,6, tende a empurrar os scores para cima. O alpha 0,75 não apareceu entre os 5 melhores, então o ótimo fica entre 0,5 e 0,6. Lote 8 apareceu em todos os 5 melhores. Com alpha 0,6 a calibração passou um pouco do ponto: o limiar ideal na validação foi 0,60 e a precisão caiu para 0,83.

- **10/10/2026 (22:00–00:11 de 11/10) · Cinco sementes da melhor configuração** · _log auditado_ \
  `wandbSementes` \
  `F1 0,8553 ± 0,0091 · AP 0,8066 ± 0,0323 · P 0,8787 ± 0,0288 · R 0,8352 ± 0,0388` \
  **O que ensinou:** A semente 42 repetiu exatamente o resultado da 1ª busca, prova de que o treino é reprodutível. Mas ela foi a melhor das cinco: o 0,8634 era otimista, e a média real é 0,855. O AP varia 3,5 vezes mais que o F1 entre sementes, então diferenças de AP abaixo de ~0,03 numa semente só são ruído. Mesmo assim, o F1 fica 0,18 acima do melhor dataset da tese, mais de seis desvios de distância.

- **11/10/2026 (22:54–00:56 de 12/10) · Cinco sementes da 2ª busca** · _log auditado_ \
  `wandbSementesFocalF1` \
  `F1 0,8643 ± 0,0037 · AP 0,8414 ± 0,0229 · P 0,8459 ± 0,0172 · R 0,8844 ± 0,0256 · 24,5 min por semente` \
  **O que ensinou:** A semente 42 repetiu F1 0,8673, então a configuração foi aplicada como deveria. Desta vez o ganho é real: +0,009 de F1 sobre o wandbSementes, com p = 0,032 num teste de permutação exato entre as 5 + 5 sementes, e +0,049 de sensibilidade (p = 0,040). O AP subiu 0,035, mas sem significância (p = 0,087). O desvio do F1 caiu para menos da metade (0,0037 contra 0,0091): alpha 0,6 com lote 8 é a configuração de maior F1 e a mais estável do projeto. O preço continua sendo a precisão, 0,846, abaixo dos 0,934 da tese.

- **11/10/2026 (22:54–02:03 de 12/10) · Superar as quatro médias ao mesmo tempo, com folga** · _log auditado_ \
  `optunaSuperaTese` \
  `P 0,9629 ± 0,0067 · R 0,5261 ± 0,0388 · F1 0,6799 ± 0,0337 · AP 0,5176 ± 0,0382 · 3 de 3 sementes superam as 4` \
  **O que ensinou:** Objetivo cumprido: as três sementes e a média superam as quatro métricas da tese, com +0,029 de folga na precisão. A busca voltou ao alpha 0,25 com lr ~0,011, praticamente o altoLearningRate da 1ª rodada: os três trials que passaram a tese na validação usaram alpha 0,25 ou 0,35, e learning rate e weight decay foram os hiperparâmetros mais importantes (0,41 e 0,29). Mas o F1 ficou em 0,68, abaixo dos 0,77 do altoLearningRate. O motivo é a escolha do checkpoint: ela exigia precisão de pelo menos 0,944 na validação, onde a precisão sai menor que no teste, e acabou ficando com épocas conservadoras, de sensibilidade baixa. As últimas épocas tinham F1 maior na validação (0,69 a 0,75). A curva precisão × sensibilidade do teste aponta para o mesmo lado: no limiar 0,45, as três sementes ainda superariam a tese em precisão e sensibilidade, com F1 de 0,73 a 0,79. Como isso foi visto no teste, é uma pista e não um resultado. Superar a tese nas quatro métricas é robusto; fazer isso com F1 alto pede uma folga menor na validação.
<!-- linha-do-tempo:fim -->

---

## 📚 Tutoriais da Flautim

Este projeto segue a estrutura ensinada nos tutoriais oficiais da plataforma. Para aprender a montar um experimento do zero:

- 📚 [Tutorial 1](https://github.com/FutureLab-DCC/flautim_tutoriais/blob/main/TUTORIAL_1.ipynb) - Notebook com exemplo de classificação usando o dataset IRIS.
- 📚 [Tutorial 2](https://github.com/FutureLab-DCC/flautim_tutoriais/blob/main/TUTORIAL_2.ipynb) - Notebook com exemplo de regressão usando o dataset BOSTON.
- 📚 [Tutorial 3](https://github.com/FutureLab-DCC/flautim_tutoriais/blob/main/TUTORIAL_3.ipynb) - Notebook com exemplo de classificação de imagens usando o dataset MNIST.
- 📚 [Tutorial 4](https://github.com/FutureLab-DCC/flautim_tutoriais/blob/main/TUTORIAL_4.ipynb) - Notebook com exemplo de classificação de imagens usando o dataset FMNIST, com seleção de clientes em ambiente federado.
- 🎥 [Tutorial em vídeo](https://drive.google.com/file/d/1eC0CnIzZ-pKLTM6M9QvOmdZlBghjyL9J/view?usp=drive_link) - Como rodar o experimento do Tutorial 1 na plataforma, passo a passo.

## 📞 Fale com a Flautim

Dúvidas sobre a plataforma ou interesse em acessá-la: 💬 flautim@dcc.ufmg.br
