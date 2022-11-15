import torch

from .retina import RetinaNet_TorchVision, RetinaFace_Biubug6, RetinaFace_BBT


class Detector():

    variations = [
        'net_torchvision_resnet50_coco',
        'face_biubug6_mobilenet', 'face_biubug6_resnet50',
        'face_bbt_resnet152_mixed', 'face_bbt_resnet50_mixed',
        'face_bbt_resnet50_wider', 'face_bbt_resnet50_icartoon'
    ]

    def __init__(self, architecture, source, variation=None, dataset=None, device=None):
        if device is None:
            dv = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            dv = device
        
        print('Initializing %s model for object detection' % architecture)

        if architecture == 'RetinaNet':
            self.model = RetinaNet_TorchVision(dv)
        
        elif architecture == 'RetinaFace':
            if source == 'Biubug6':
                self.model = RetinaFace_Biubug6(variation == 'MobileNet', variation.lower(), dv)
            elif variation.startswith('BBT_'):
                self.model = RetinaFace_BBT(variation, dv)
            else:
                raise ValueError('Unknown variation for architecture RetinaFace')
        
        else:
            raise ValueError('Unknown architecture')
        
        self.model.eval()

    def __call__(self, imgs):
        with torch.inference_mode():
            return self.model(imgs)