from .main import video_to_faces
from .detectors.retinaface import RetinaFaceDetector
from .detectors.yolo3 import YOLOv3Detector, YOLOv3DetectorAnime
from .detectors import MTCNNDetector
#from .encoders import MobileFaceNetEncoder, IncepResEncoder, IResNetEncoder
from .encoders import VitEncoderAnime
from .utils.eval_det import eval_det_wider, eval_det_fddb, eval_det_icartoon