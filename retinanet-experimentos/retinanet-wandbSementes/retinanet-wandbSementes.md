# retinanet-wandbSementes

Informações sobre o experimento

image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
batch_size: 4 (retinanet-wandb: 8)  
learning_rate: 2,6e-3 (retinanet-wandb: 1e-2)  
momentum: 0.919 (retinanet-wandb: 0.9)  
weight_decay: 2,9e-4 (retinanet-wandb: 1e-4)  
warmup_steps: 300  
grad_clip: 5 (retinanet-wandb: 10)  
epochs: 30 (retinanet-wandb: 40)  
trainable_layers: 5  
focal_alpha: 0.5 (retinanet-wandb: 0.25)  
focal_gamma: 2.0  
fg_iou (matcher): 0.4 (retinanet-wandb: 0.5)  
nms_threshold: 0.4  
merge_nms: 0.3 (retinanet-wandb: 0.4)  
score_threshold: 0.5  
evaluation_iou: 0.1  
architecture: retinanet  
sementes de treino: 42, 43, 44, 45, 46 (split de teste fixo, semente 42)  

**Configuração:** a do trial 12 do retinanet-optuna (F1 0,8634 e AP 0,8386 no teste, limiar 0,5).  
**Acompanhamento:** Weights & Biases, projeto `retinanet-bacilos`, grupo `retinanet-wandbSementes`: um run por semente e um run "resumo". Sem `WANDB_API_KEY`, roda em modo offline em `output_path/wandb/`.  
**Saídas em `output_path/sementes/`:** `sementes.json` (atualizado a cada semente), `sementes.csv` e `resumo.json` (média ± desvio amostral, mínimo e máximo).  
**Duração estimada:** ~27 min por semente, ~2,3 h no total.  
Variáveis de ambiente: `RETINA_SEEDS` (ex.: `42,43,44`), `RETINA_SPLIT_SEED`, `RETINA_EPOCHS`, e cada hiperparâmetro (`RETINA_FOCAL_ALPHA`, `RETINA_FG_IOU` etc.).

**Variação:** Repete a melhor configuração do Optuna com 5 sementes, para reportar média ± desvio como a tese, em vez de um número de uma rodada só.
