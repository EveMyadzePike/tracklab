from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
from tracklab.utils.cv2 import cv2_load_image, crop_bbox_ltwh
from tracklab.utils.easyocr import bbox_easyocr_to_image_ltwh

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.openmmlab import get_checkpoint

from collections import Counter


import logging


log = logging.getLogger(__name__)


class VotingTrackletJerseyNumber(VideoLevelModule):
    
    input_columns = []
    output_columns = ["jn_tracklet"]
    
    def __init__(self, cfg, device, tracking_dataset=None):
        pass

    def select_best_jersey_number(self, jersey_numbers, jn_confidences):
        
        confidence_sum = {}
        
        # Iterate through the predictions to calculate the total confidence for each jersey number
        for jn, conf in zip(jersey_numbers, jn_confidences):
            if jn not in confidence_sum:
                confidence_sum[jn] = 0
            confidence_sum[jn] += conf
        
        # Find the jersey number with the maximum total confidence
        if len(confidence_sum) == 0:
            return None
        max_confidence_jn = max(confidence_sum, key=confidence_sum.get)
        return max_confidence_jn
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        detections["jn_tracklet"] = [np.nan] * len(detections)
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            jersey_numbers = tracklet.jersey_number
            jn_confidences = tracklet.jn_confidence
            tracklet_jn = [self.select_best_jersey_number(jersey_numbers, jn_confidences)] * len(tracklet)            
            detections.loc[tracklet.index, "jn_tracklet"] = tracklet_jn
            
        return detections
