# retinanet-wandbSementesFocalF1Alpha055

Informações sobre o experimento

image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
batch_size: 8  
learning_rate: 2,5e-3  
momentum: 0.947  
weight_decay: 2,8e-4  
warmup_steps: 300  
grad_clip: 2.5  
epochs: 30  
trainable_layers: 5  
focal_alpha: 0.55 (retinanet-wandbSementesFocalF1: 0.6)  
focal_gamma: 1.5  
fg_iou (matcher): 0.5  
nms_threshold: 0.3  
merge_nms: 0.25  
score_threshold: 0.5  
evaluation_iou: 0.1  
architecture: retinanet  
sementes de treino: 47, 48, 49, 50, 51, 52, 53, 54, 55, 56 (split de teste fixo, semente 42)  

**Configuração:** a do trial 1 do retinanet-optunaFocalF1, igual à do retinanet-wandbSementesFocalF1 (F1 0,8643 ± 0,0037 em 5 sementes), com um hiperparâmetro trocado: **focal_alpha 0,55** em vez de 0,6.  
**Por que alpha 0,55:** o focal_alpha foi o hiperparâmetro mais importante da busca (0,33), e os 5 melhores trials usaram 0,6. Mas com 0,6 a calibração passou do ponto: precisão 0,846 contra sensibilidade 0,884, e o limiar ideal na validação foi 0,60, não 0,5. Com alpha 0,5 (retinanet-wandbSementes, que também mudava outros hiperparâmetros) a precisão foi maior (0,879) e o F1 menor (0,855). O 0,55 fica entre os dois. O esperado é a precisão subir e o F1 ficar parecido ou um pouco maior; não há garantia de que o F1 suba.  
**O que mais muda:** as sementes (10 novas, 47 a 56, porque as 42 a 46 já foram usadas) e a **escolha do checkpoint**.  
**Critérios de checkpoint, todos medidos só na validação** (cada semente treina uma vez e guarda os três):  
- **ultima, o PRINCIPAL, declarado antes de rodar:** a última época (30). Só ele entra nas tabelas e no painel.  
- ap_voc: maior AP_VOC na validação. É o critério que o wandbSementesFocalF1 usou. Roda nos mesmos treinos que `ultima`, então `ultima` × `ap_voc` isola o critério (pareado, mesmas sementes). `ap_voc` aqui contra o wandbSementesFocalF1 isola o alpha, mas com sementes diferentes (10 × 5), então a comparação é mais fraca.  
- f1_val: maior F1 na validação, no limiar 0,5. É o objetivo da busca que achou esta configuração.  

Os dois secundários vão para o log como análise (linhas `rtn/criterio >`), com o resultado no teste de cada um.  
**Foco em F1, como no optunaFocalF1:** nenhum critério exige precisão acima da tese. Os três são de F1 ou AP, e as métricas esperadas são as do optunaFocalF1 (precisão ~0,83 a 0,85, sensibilidade ~0,88 a 0,91, F1 ~0,86 a 0,87). Neste experimento a linha "supera as 4 médias da tese" sai "não", como no optunaFocalF1, e isso é esperado. Se você preferir o F1 da validação como critério principal, é só trocar `PRINCIPAL` e `SECUNDARIOS` no `run.py`: os três checkpoints já são guardados.  
**Por que a última época:** no wandbSementesSuperaTese e no wandbSementesAlpha030 ela teve o maior F1 (0,7036 e 0,7537), e escolher a época pelo máximo da validação (20 lâminas, ruidosa) premiou épocas de sorte. Era um critério secundário visto no teste, então ela precisa ser confirmada num experimento que a declare antes. Este é esse experimento.  
**O que esperar:** a validação deste modelo é bem mais estável que a dos experimentos de alpha 0,25 e 0,30 (F1 de validação plano em ~0,83 depois da época 5), então a diferença entre os critérios deve ser pequena. Nesse caso o ganho do experimento está nas sementes: com as 5 do wandbSementesFocalF1, a configuração passa a ter 15, e o `ap_voc` das 10 novas mede quanto o alpha 0,55 muda o resultado.  
**Mantido igual ao retinanet-wandbSementesFocalF1:** código de treino, split, número de épocas e protocolo de avaliação (limiar 0,5, IoU 0,1).  
**Acompanhamento:** W&B, grupo `retinanet-wandbSementesFocalF1Alpha055` (offline sem `WANDB_API_KEY`).  
**Saídas em `output_path/sementes/`:** `criterios.json` (resultado de cada critério por semente), `resumo.json` e `curvas_pr.json`.  
**Duração estimada:** ~26 min por semente, ~4,3 h no total. As sementes podem ser trocadas pela variável `RETINA_SEEDS`.

**Variação:** Repete a melhor configuração do optunaFocalF1 com focal_alpha 0,55, dez sementes novas e checkpoint pela última época, declarada antes de rodar. Perguntas: baixar o alpha de 0,6 para 0,55 devolve precisão sem perder F1? A última época melhora o F1 em relação ao critério original (AP_VOC na validação), como melhorou no alpha 0,30?
