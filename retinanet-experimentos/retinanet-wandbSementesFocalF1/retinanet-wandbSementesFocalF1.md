# retinanet-wandbSementesFocalF1

Informações sobre o experimento

image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
batch_size: 8 (retinanet-wandbSementes: 4)  
learning_rate: 2,5e-3 (retinanet-wandbSementes: 2,6e-3)  
momentum: 0.947 (retinanet-wandbSementes: 0.919)  
weight_decay: 2,8e-4 (retinanet-wandbSementes: 2,9e-4)  
warmup_steps: 300  
grad_clip: 2.5 (retinanet-wandbSementes: 5)  
epochs: 30  
trainable_layers: 5  
focal_alpha: 0.6 (retinanet-wandbSementes: 0.5)  
focal_gamma: 1.5 (retinanet-wandbSementes: 2.0)  
fg_iou (matcher): 0.5 (retinanet-wandbSementes: 0.4)  
nms_threshold: 0.3 (retinanet-wandbSementes: 0.4)  
merge_nms: 0.25 (retinanet-wandbSementes: 0.3)  
score_threshold: 0.5  
evaluation_iou: 0.1  
architecture: retinanet  
sementes de treino: 42, 43, 44, 45, 46 (split de teste fixo, semente 42)  

**Configuração:** a do trial 1 do retinanet-optunaFocalF1 (F1 0,8673 e AP 0,8593 no teste, com uma semente só).  
**Mantido igual ao retinanet-wandbSementes:** código, sementes, split, número de épocas, seleção do checkpoint e protocolo de avaliação. Só os hiperparâmetros mudam, para a comparação média × média ser direta (referência: F1 0,8553 ± 0,0091, AP 0,8066 ± 0,0323).  
**Checagem de corretude:** a semente 42 deve reproduzir o retreino final do retinanet-optunaFocalF1 (F1 0,8673, AP 0,8593), assim como no wandbSementes a semente 42 reproduziu o retinanet-optuna.  
**Acompanhamento:** Weights & Biases, projeto `retinanet-bacilos`, grupo `retinanet-wandbSementesFocalF1`. Sem `WANDB_API_KEY`, roda em modo offline em `output_path/wandb/`.  
**Saídas em `output_path/sementes/`:** `sementes.json`, `sementes.csv` e `resumo.json`.  
**Duração estimada:** ~25 min por semente, ~2,1 h no total.

**Variação:** Repete com 5 sementes a melhor configuração da 2ª busca do Optuna, para confirmar se o ganho de focal_alpha 0,6 + batch 8 sobre o retinanet-wandbSementes é real ou sorte de semente.
