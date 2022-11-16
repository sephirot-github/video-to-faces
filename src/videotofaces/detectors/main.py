from enum import Enum, auto

import torch

from .retina import RetinaNet_TorchVision, RetinaFace_Biubug6, RetinaFace_BBT
from .yolo import YOLOv3


class Detector():

    def __init__(self, modelnum, device=None):
        if device is None:
            dv = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            dv = device

        architecture = modelnum.name.split('_')[0]

        print('Initializing %s model for object detection' % architecture)

        if architecture == 'RetinaNet':
            self.model = RetinaNet_TorchVision(dv)
        elif architecture == 'RetinaFace':
            source, variation = modelnum.name.lower().split('_')[1:]
            if source == 'biubug6':
                self.model = RetinaFace_Biubug6(variation == 'mobilenet', variation, dv)
            elif source == 'bbt':
                self.model = RetinaFace_BBT(variation == 'resnet50', variation, dv)
        elif architecture == 'YOLOv3':
            dataset = modelnum.name.lower().split('_')[1]
            n = 80 if dataset == 'coco' else 1
            self.model = YOLOv3(dataset, dv, num_classes=n)
        
        self.model.eval()

    def __call__(self, imgs):
        with torch.inference_mode():
            return self.model(imgs)


class detmodels(Enum):
    RetinaNet = auto()
    RetinaFace_Biubug6_MobileNet = auto()
    RetinaFace_Biubug6_ResNet50 = auto()
    RetinaFace_BBT_ResNet50_Mixed = auto()
    RetinaFace_BBT_ResNet50_Wider = auto()
    RetinaFace_BBT_ResNet50_iCartoon = auto()
    RetinaFace_BBT_ResNet152_Mixed = auto()
    YOLOv3_Anime = auto()
    YOLOv3_Wider = auto()
    YOLOv3_COCO = auto()