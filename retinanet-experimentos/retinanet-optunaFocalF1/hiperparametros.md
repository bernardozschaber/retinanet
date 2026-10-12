# retinanet-optunaFocalF1

Informações sobre o experimento

**Fixos (iguais ao baseline):**  
image_size: 640  
tile_overlap: 128  
crops_per_image: 8  
target_box: 40  
warmup_steps: 300 (era buscado na 1ª busca; importância 0,004)  
trainable_layers: 5 (era buscado na 1ª busca; importância 0,006)  
score_threshold: 0.5  
evaluation_iou: 0.1  
architecture: retinanet  

**Espaço de busca (Optuna, TPE multivariado + MedianPruner), em volta do trial 12 do retinanet-optuna:**  
learning_rate: 1,5e-3 a 6e-3 (log) (1ª busca: 2e-3 a 2e-2; trial 12: 2,6e-3)  
momentum: 0.88 a 0.95 (1ª: 0.85 a 0.95; trial 12: 0.919)  
weight_decay: 1e-4 a 4e-4 (log) (1ª: 3e-5 a 3e-4; trial 12: 2,9e-4)  
grad_clip: 2.5 / 5 / 10 (1ª: 5 / 10 / 20; trial 12: 5)  
batch_size: 4 / 8  
focal_alpha: 0.5 / 0.6 / 0.75 (1ª: 0.25 / 0.35 / 0.5; os 5 melhores trials usaram 0.5)  
focal_gamma: 1.5 / 2.0 / 2.5 (1ª: 1.0 / 1.5 / 2.0; trial 12: 2.0)  
fg_iou (matcher): 0.4 / 0.45 / 0.5 (trial 12: 0.4)  
nms_threshold (modelo): 0.3 / 0.4 / 0.5  
merge_nms (entre ladrilhos): 0.25 / 0.3 / 0.4 (1ª: 0.3 / 0.4 / 0.5; trial 12: 0.3)  

**Orçamento:** 20 trials × 12 épocas (validação a cada 2 épocas), timeout de 10 h. O 1º trial é o trial 12 da 1ª busca. A 1ª busca levou ~9 min por trial e 2,3 h no total; esta deve levar ~3,3 h (menos, se houver poda), mais ~27 min de retreino final.  
**Objetivo:** F1 no limiar 0,5 da tese, na validação (a 1ª busca usou F1 no melhor limiar).  
**Retreino final:** a melhor configuração é retreinada com 30 épocas via `experiment.run()` e avaliada no teste.  
**Saídas em `output_path/optuna/`:** `trials.csv`, `best_params.json` e `test_metrics.json`.

**Variação:** 2ª busca, mais estreita. Testa se a perda focal acima de 0,5 calibra ainda melhor os scores no limiar 0,5, e otimiza diretamente a métrica comparada com a tese.
