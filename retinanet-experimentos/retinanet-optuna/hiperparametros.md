# retinanet-optuna

Informações sobre o experimento

**Fixos (iguais ao baseline):**  
image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
score_threshold: 0.5  
evaluation_iou: 0.1  
architecture: retinanet  

**Espaço de busca (Optuna, TPE multivariado + MedianPruner):**  
learning_rate: 2e-3 a 2e-2 (log)  
momentum: 0.85 a 0.95  
weight_decay: 3e-5 a 3e-4 (log)  
warmup_steps: 100 a 600 (passo 100)  
grad_clip: 5 / 10 / 20  
batch_size: 4 / 8 (mais VRAM)  
trainable_layers: 3 a 5  
focal_alpha: 0.25 / 0.35 / 0.5 (novo; padrão 0.25)  
focal_gamma: 1.0 / 1.5 / 2.0 (novo; padrão 2.0)  
fg_iou (matcher): 0.4 / 0.5 (novo; bg = fg − 0.1)  
nms_threshold (modelo): 0.3 / 0.4 / 0.5  
merge_nms (entre ladrilhos): 0.3 / 0.4 / 0.5  

**Orçamento:** 16 trials × 12 épocas (validação a cada 2 épocas), timeout de 10 h. Os 2 primeiros trials são o baseline e o altoLearningRate.  
**Objetivo:** F1 no melhor limiar de score, na validação (o AP_VOC fica saturado em ~0,92 em todos os runs e não separa as configurações).  
**Próxima rodada:** o padrão do `run.py` passou a ser F1 no limiar 0,5 (`RETINA_OPTUNA_OBJECTIVE=F1`); esta rodada usou o F1 no melhor limiar.  
**Retreino final:** a melhor configuração é retreinada com 30 épocas via `experiment.run()` e avaliada no teste, como os outros experimentos.  
**Saídas em `output_path/optuna/`:** `trials.csv`, `best_params.json` (atualizados a cada trial) e `test_metrics.json`.  

Tudo pode ser mudado por variável de ambiente: `RETINA_OPTUNA_TRIALS`, `RETINA_TRIAL_EPOCHS`, `RETINA_OPTUNA_TIMEOUT_HOURS`, `RETINA_TRIAL_VAL_EVERY`, `RETINA_OPTUNA_OBJECTIVE`, `RETINA_EPOCHS`, `RETINA_FINAL_RETRAIN=0` (só a busca).

**Variação:** Busca automática de hiperparâmetros. Além de lr/wd/momentum, inclui a perda focal e o IoU do matcher, porque no limiar 0,5 o modelo erra por calibração (precisão ~0,95, sensibilidade ~0,60).
