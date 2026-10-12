# retinanet-optunaSuperaTese

Informações sobre o experimento

**Alvo (média do RetinaNet da tese, datasets 1–4, limiar 0,5), superar as 4 ao mesmo tempo:**  
Precisão > 0,9339 · Sensibilidade > 0,4741 · F-measure > 0,6244 · AP > 0,4533  

**Fixos:**  
image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
warmup_steps: 300  
trainable_layers: 5  
epochs (retreino final): 30  
score_threshold: 0.5  
evaluation_iou: 0.1  
architecture: retinanet  

**Espaço de busca (Optuna, TPE multivariado + MedianPruner):**  
focal_alpha: 0.25 / 0.3 / 0.35 / 0.4 / 0.45 / 0.5 (alavanca de calibração: 0,25 → precisão alta; 0,5–0,6 → precisão abaixo da tese)  
focal_gamma: 1.5 / 2.0  
learning_rate: 2e-3 a 1,2e-2 (log)  
momentum: 0.88 a 0.95  
weight_decay: 8e-5 a 4e-4 (log)  
grad_clip: 2.5 / 5 / 10  
batch_size: 4 / 8  
fg_iou (matcher): 0.4 / 0.45 / 0.5  
nms_threshold: 0.3 / 0.4  
merge_nms: 0.25 / 0.3 / 0.4  

**Objetivo (na validação):** se as 4 métricas passam a média da tese com folga de +0,01, vale 1 + F1; senão, vale −(soma do que falta). Qualquer configuração que supera as 4 vale mais do que qualquer uma que não supera; entre as que superam, ganha a de maior F1.  
**Primeiros trials:** altoLearningRate (supera as 4 no teste), trial 12 da 1ª busca com α 0,4 e trial 1 da 2ª busca com α 0,4.  
**Retreino final:** a melhor configuração é treinada com as sementes 42, 43 e 44 (split fixo). O checkpoint de cada semente é escolhido pelo mesmo critério da busca, e não pela AP_VOC. O log diz quantas sementes superam as 4 e se a média supera.  
**Orçamento:** 16 trials × 12 épocas (timeout de 5 h) + 3 sementes × 30 épocas. Total estimado: ~3,5–4 h.  
**Acompanhamento:** W&B, grupo `retinanet-optunaSuperaTese` (offline sem `WANDB_API_KEY`).  
**Saídas:** `output_path/optuna/` (`trials.csv`, `best_params.json`) e `output_path/sementes/` (`sementes.json`, `resumo.json` e `curvas_pr.json`, que traz a curva precisão × sensibilidade do teste e os TP/FP/FN por limiar).  
Variáveis de ambiente: `RETINA_SEEDS`, `RETINA_FOLGA`, `RETINA_FOLGA_PRECISAO`, `RETINA_OPTUNA_TRIALS`, `RETINA_TRIAL_EPOCHS`, `RETINA_OPTUNA_TIMEOUT_HOURS`, `RETINA_EPOCHS`.

**Variação:** Em vez de maximizar o F1 (que levou o Optuna a trocar precisão por sensibilidade e cair abaixo da tese na precisão), busca o equilíbrio em que as 4 métricas superam a média da tese ao mesmo tempo, com várias sementes.
