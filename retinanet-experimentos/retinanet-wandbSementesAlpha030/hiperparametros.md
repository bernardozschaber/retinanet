# retinanet-wandbSementesAlpha030

Informações sobre o experimento

image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
batch_size: 4  
learning_rate: 1,08e-2  
momentum: 0.886  
weight_decay: 1,18e-4  
warmup_steps: 300  
grad_clip: 10  
epochs: 30  
trainable_layers: 5  
focal_alpha: 0,30  
focal_gamma: 2.0  
fg_iou (matcher): 0.5  
nms_threshold: 0.4  
merge_nms: 0.4  
score_threshold: 0.5  
evaluation_iou: 0.1  
architecture: retinanet  
sementes de treino: 42, 43, 44, 45, 46 (split de teste fixo, semente 42)  

**Configuração:** a do trial 6 do retinanet-optunaSuperaTese (3 de 3 sementes acima da tese nas 4 métricas, F1 0,6799 ± 0,0337), com focal_alpha 0,30.  
**Critérios de checkpoint, todos medidos só na validação** (cada semente treina uma vez e guarda os quatro):  
- **sem_folga, o PRINCIPAL, declarado antes de rodar:** a época tem de superar a tese nas 4 métricas da validação; entre essas, a de maior F1. Só ele entra nas tabelas e no painel.  
- folga_001: o mesmo com folga de 0,01, o critério do optunaSuperaTese.  
- ap_voc: maior AP_VOC na validação, o critério da 1ª rodada.  
- ultima: a última época.  

Os três secundários vão para o log como análise (linhas `rtn/criterio >`), com o resultado no teste de cada um e quantas sementes superam a tese.  
**Acompanhamento:** W&B, grupo `retinanet-wandbSementesAlpha030` (offline sem `WANDB_API_KEY`).  
**Saídas em `output_path/sementes/`:** `criterios.json` (resultado de cada critério por semente), `resumo.json` e `curvas_pr.json`.  
**Duração estimada:** ~26 min por semente, ~2,3 h no total. Pode rodar em paralelo com o outro experimento desta rodada.

**Variação:** Igual ao retinanet-wandbSementesSuperaTese, com focal_alpha 0,30. Pergunta: subir um pouco o alpha empurra os scores para cima e ganha sensibilidade, mantendo a precisão acima da tese com o limiar fixo em 0,5? É a forma legítima de testar a pista do limiar 0,45, que foi vista no teste.
