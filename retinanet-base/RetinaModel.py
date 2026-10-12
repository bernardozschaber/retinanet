import os
from functools import partial

import torch
from flautim.pytorch.Model import Model
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from RetinaDataset import etapa, log


class RetinaModel(Model):
    """Detector de bacilos (RetinaNet R50-FPN v2) no formato da Flautim.

    O detector vem do torchvision: ele ja traz a perda, o casamento de
    ancoras e o NMS da inferencia, e normaliza as imagens dentro do
    proprio forward. Tres ajustes em relacao aos pesos do COCO:

    - a cabeca de classificacao e trocada (91 classes -> fundo + bacilo);
    - as ancoras comecam em 16 px em vez de 32: o dataset reescala as
      imagens para o bacilo mediano ter ~40 px, e com a ancora minima
      de 32 os bacilos menores quase nunca casavam com ancora nenhuma;
    - min_size/max_size iguais ao lado do ladrilho, para o transform
      interno nunca reescalar a entrada (o dataset ja entrega na escala
      certa).

    `architecture="fasterrcnn"` troca o RetinaNet pelo Faster R-CNN R50-FPN
    v2 — o metodo com o qual a tese chegou ao alvo (Exp. 8.1) — mantendo
    as mesmas ancoras pequenas, entrada e interface.
    """

    TAMANHOS_ANCORA = (16, 32, 64, 128, 256)

    def __init__(
        self,
        context,
        num_classes=2,
        pretrained=True,
        tile_size=640,
        architecture="retinanet",
        trainable_backbone_layers=5,
        score_thresh=0.05,
        nms_thresh=0.4,
        detections_per_img=300,
        **kwargs,
    ):
        architecture = str(architecture).lower()
        if architecture not in ("retinanet", "fasterrcnn"):
            raise ValueError(f"arquitetura desconhecida: {architecture}")
        log(f"rtn/modelo > iniciando a construcao do {architecture} R50-FPN v2")
        log(f"rtn/modelo > backbone pre-treinado COCO: {pretrained}")
        super(RetinaModel, self).__init__(
            context, name="Bacilos-RetinaNet-R50-FPN", version=2, id=1, **kwargs
        )
        self.architecture = architecture
        self.num_classes = int(num_classes)
        self.tile_size = int(tile_size)

        comuns = dict(
            min_size=self.tile_size,
            max_size=self.tile_size,
            trainable_backbone_layers=int(trainable_backbone_layers),
        )
        with etapa("carregar o detector e o backbone ResNet-50", object="modelo"):
            if architecture == "retinanet":
                pesos = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
                self.detector = retinanet_resnet50_fpn_v2(
                    weights=pesos,
                    score_thresh=float(score_thresh),
                    nms_thresh=float(nms_thresh),
                    detections_per_img=int(detections_per_img),
                    **comuns,
                )
            else:
                pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
                self.detector = fasterrcnn_resnet50_fpn_v2(
                    weights=pesos,
                    box_score_thresh=float(score_thresh),
                    box_nms_thresh=float(nms_thresh),
                    box_detections_per_img=int(detections_per_img),
                    **comuns,
                )

        with etapa("ajustar ancoras e cabeca de classificacao", object="modelo"):
            if architecture == "retinanet":
                # 3 escalas x 3 proporcoes por nivel = 9 ancoras, o mesmo
                # numero dos pesos do COCO, entao a cabeca de regressao
                # pre-treinada continua servindo.
                tamanhos = tuple(
                    (s, int(round(s * 2 ** (1 / 3))), int(round(s * 2 ** (2 / 3))))
                    for s in self.TAMANHOS_ANCORA
                )
                self.detector.anchor_generator = AnchorGenerator(
                    tamanhos, ((0.5, 1.0, 2.0),) * len(tamanhos)
                )
                canais = self.detector.backbone.out_channels
                ancoras = self.detector.anchor_generator.num_anchors_per_location()[0]
                self.detector.head.classification_head = RetinaNetClassificationHead(
                    canais,
                    ancoras,
                    self.num_classes,
                    norm_layer=partial(nn.GroupNorm, 32),
                )
            else:
                tamanhos = tuple((s,) for s in self.TAMANHOS_ANCORA)
                self.detector.rpn.anchor_generator = AnchorGenerator(
                    tamanhos, ((0.5, 1.0, 2.0),) * len(tamanhos)
                )
                entrada = self.detector.roi_heads.box_predictor.cls_score.in_features
                self.detector.roi_heads.box_predictor = FastRCNNPredictor(
                    entrada, self.num_classes
                )

        total = sum(parametro.numel() for parametro in self.parameters())
        treinaveis = sum(
            parametro.numel()
            for parametro in self.parameters()
            if parametro.requires_grad
        )
        log(
            f"rtn/modelo > ancoras: {tamanhos}; classes={self.num_classes} "
            f"(inclui o fundo); camadas treinaveis do backbone={trainable_backbone_layers}"
        )
        log(
            f"rtn/modelo > entrada fixa em {self.tile_size}x{self.tile_size}; "
            f"score_thresh={score_thresh}, nms_thresh={nms_thresh}, "
            f"detections_per_img={detections_per_img}"
        )
        log(
            f"rtn/modelo > parametros: total={total:,}, treinaveis={treinaveis:,}"
        )
        log("rtn/modelo > construcao concluida")
        self._forma_registrada = False

    def save(self):
        # O Experiment.run() da Flautim chama model.save() ao fim do treino
        # e grava em <path>/models/, diretorio que nao existe no pod: sem
        # isto o torch.save estoura e o run inteiro e marcado como erro,
        # mesmo com o treino concluido.
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        torch.save(self.state_dict(), self.file)

    def forward(self, images, targets=None):
        if not self._forma_registrada:
            amostra = images[0] if isinstance(images, (list, tuple)) else images
            log(
                f"rtn/modelo > primeiro forward: entrada={tuple(amostra.shape)}, "
                f"modo={'treino' if targets is not None else 'inferencia'}"
            )
            self._forma_registrada = True
        if targets is not None:
            # modo treino: dicionario de perdas
            return self.detector(images, targets)
        # modo inferencia: uma predicao (boxes/scores/labels) por imagem
        return self.detector(images)
