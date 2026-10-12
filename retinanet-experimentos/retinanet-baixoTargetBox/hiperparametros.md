# retinanet-baixoTargetBox

Informações sobre o experimento

image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 32 (base: 40)  
batch_size: 4  
learning_rate: 5e-3  
momentum: 0.9  
weight_decay: 1e-4  
warmup_steps: 300  
grad_clip: 10.0  
epochs: 30  
trainable_layers: 5  
score_threshold: 0.5  
nms_threshold: 0.4  
evaluation_iou: 0.1  
architecture: retinanet  

**Variação:** Imagens reescaladas para o bacilo mediano ter ~32 px (objetos menores).
