# retinanet-wandb

Informações sobre o experimento

image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
batch_size: 8 (base: 4)  
learning_rate: 1e-2 (base: 5e-3)  
momentum: 0.9  
weight_decay: 1e-4  
warmup_steps: 300  
grad_clip: 10.0  
epochs: 40 (base: 30)  
trainable_layers: 5  
score_threshold: 0.5  
nms_threshold: 0.4  
evaluation_iou: 0.1  
architecture: retinanet  

**Acompanhamento:** Weights & Biases, projeto `retinanet-bacilos`. A chave vai em `WANDB_API_KEY` (variável de ambiente ou constante no topo do run.py). Sem chave ou sem internet, o run fica em modo offline em `output_path/wandb/`, e dá para enviar depois com `wandb sync`.  
**Métricas enviadas por época:** perdas de treino (total, classificação, regressão), lr, pico de VRAM, tempo; métricas de validação (P/R/F1, AP_TESE, AP_VOC, AP50, melhor limiar). No fim, as métricas de teste ficam no summary, junto com uma tabela que compara este run com o RetinaNet da tese.

**Variação:** Junta as duas variações que melhoraram o F1 no limiar 0,5 (altoLearningRate + altoEpochs) e dobra o lote, já que sobrava VRAM.
