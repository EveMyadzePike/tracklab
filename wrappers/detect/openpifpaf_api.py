from pbtrack.core.detector import Detector
from pbtrack.datastruct.detections import Detection
from pbtrack.utils.coordinates import kp_to_bbox_w_threshold

import sys
from hydra.utils import to_absolute_path
sys.path.append(to_absolute_path("plugins/detect/openpifpaf/src"))
import openpifpaf

class OpenPifPaf(Detector):
    def __init__(self, cfg, device):
        assert cfg.checkpoint or cfg.train, "Either a checkpoint or train must be declared"
        self.cfg = cfg
        self.device = device
        self.id = 0
        
        if cfg.checkpoint:
            self.predictor = openpifpaf.Predictor(cfg.checkpoint)
            self.predictor.model.to(device)
        
    def preprocess(self, metadata):
        return str(metadata.file_path)
    
    def process(self, preprocessed_batch, metadatas):
        detections = []
        processed = self.predictor.images(preprocessed_batch)
        for ((predictions, _, _), (_, metadata)) in zip(processed, metadatas.iterrows()):
            for prediction in predictions:
                detections.append(
                    Detection(
                        image_id = metadata.id,
                        id = self.id,
                        keypoints_xyc = prediction.data,
                        bbox = kp_to_bbox_w_threshold(prediction.data, vis_threshold=0.05),
                    )  # type: ignore
                )
                self.id += 1
        return detections

    def train(self):
        saved_argv = sys.argv
        sys.argv += [f'--{str(k)}={str(v)}' for k, v in self.cfg.train.items()]
        self.cfg.checkpoint = openpifpaf.train.main()
        sys.argv = saved_argv
        self.predictor = openpifpaf.Predictor(self.cfg.checkpoint)
        self.predictor.model.to(self.device)