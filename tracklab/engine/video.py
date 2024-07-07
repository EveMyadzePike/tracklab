import platform
from functools import partial
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from lightning import Fabric

from tracklab.engine import TrackingEngine
from tracklab.engine.engine import merge_dataframes
from tracklab.pipeline import Pipeline

import logging

log = logging.getLogger(__name__)


class VideoOnlineTrackingEngine:
    def __init__(
        self,
        modules: Pipeline,
        filename: str,
        target_fps: int,
        tracker_state,
        num_workers: int,
        callbacks: "Dict[Callback]" = None,
    ):
        # super().__init__()
        self.module_names = [module.name for module in modules]
        callbacks = list(callbacks.values()) if callbacks is not None else []

        self.fabric = Fabric(callbacks=callbacks)
        self.callback = partial(self.fabric.call, engine=self)
        self.num_workers = num_workers
        self.video_filename = filename
        self.target_fps = target_fps
        self.tracker_state = tracker_state
        self.img_metadatas = tracker_state.image_metadatas
        self.video_metadatas = tracker_state.video_metadatas
        self.models = {model.name: model for model in modules}
        self.datapipes = {}
        self.dataloaders = {}
        for model_name, model in self.models.items():
            self.datapipes[model_name] = getattr(model, "datapipe", None)
            self.dataloaders[model_name] = getattr(model, "dataloader", lambda **kwargs: ...)(engine=self)

    #the online method has its own track_dataset function
    def track_dataset(self):
        """Run tracking on complete dataset."""
        self.callback("on_dataset_track_start")
        self.callback(
            "on_video_loop_start",
            video_metadata=pd.Series(name=self.video_filename),
            video_idx=0,
            index=0,
        )
        detections = self.video_loop()
        self.callback(
            "on_video_loop_end",
            video_metadata=pd.Series(name=self.video_filename),
            video_idx=0,
            detections=detections,
        )
        self.callback("on_dataset_track_end")

    def video_loop(self):
        #will have 3 models that I can reset if possible
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()

        #online means cv2
        video_filename = int(self.video_filename) if str(self.video_filename).isnumeric() else str(self.video_filename)

        # create video capture object from webcam or video path
        video_cap = cv2.VideoCapture(video_filename)
        fps = video_cap.get(cv2.CAP_PROP_FPS) #get the frames per secomd
        frame_modulo = fps // self.target_fps #not sure what target is?
        assert video_cap.isOpened(), f"Error opening video stream or file {video_filename}"
        if platform.system() == "Linux":
            cv2.namedWindow(str(self.video_filename), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # cv2.resizeWindow(str(self.video_filename))

        model_names = self.module_names
        # print('in offline.py, model_names: ', model_names)
        frame_idx = -1
        detections = pd.DataFrame() #start with empty dataframe
        while video_cap.isOpened():
            frame_idx += 1 #this is a number
            ret, frame = video_cap.read() #read in 1st frame
            if frame_idx % frame_modulo != 0:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert the frame to RGB
            if not ret:
                break #if nothing read, stop

            #start collecting the metadata
            #id and frame are the same and name are the same
            metadata = pd.Series({"id": frame_idx, "frame": frame_idx,
                                  "video_id": video_filename}, name=frame_idx)

            #passes the data frame to the callback function named below
            self.callback("on_image_loop_start",
                          image_metadata=metadata, image_idx=frame_idx, index=frame_idx)

            #3 model names, so detections gets passed to the default step 3 times and the detections get 
            #updated and returned to the model
            
            for model_name in model_names:
                #so 1st model = yolov8
                model = self.models[model_name]
                if len(detections) > 0: #meaning your dataframe is not empty
                    # dets is a filtered data frame, after applying mask, so that you only see the
                    #the detctions that correspond to the current frame
                    dets = detections[detections.image_id == frame_idx]
                else:
                    dets = pd.DataFrame() #if dataframe is empty, no need to filter, it just stays empty
                if model.level == "video":
                    raise "Video-level not supported for online video tracking" #online means pass one frame at a time
                    #video level is offline

                #This is the heart of it
                elif model.level == "image":
                    #pass the image (RGB), the filtered detetcions, and the metadata to the appropriate models preprocess function
                    #The image, current detections (dets), and metadata for the current frame are passed to the preprocess method 
                    #of the model. This method prepares the data for the model's processing.
                    batch = model.preprocess(image=image, detections=dets, metadata=metadata)

                    #The preprocessed data is then collated into a batch format suitable for the model. 
                    #The collate_fn method is used for this, which is a static method of the model's class. 
                    #The batch is packed with its index (frame_idx).
                    batch = type(model).collate_fn([(frame_idx, batch)])

                    #The default_step method is called with the collated batch, model name, existing detections, and metadata. 
                    #This method handles the core processing logic for the model.
                    detections = self.default_step(batch, model_name, detections, metadata)
                elif model.level == "detection":
                    for idx, detection in dets.iterrows():
                        batch = model.preprocess(image=image, detection=detection, metadata=metadata)
                        batch = type(model).collate_fn([(detection.name, batch)])
                        detections = self.default_step(batch, model_name, detections, metadata)

            self.callback("on_image_loop_end",
                          image_metadata=metadata, image=image,
                          image_idx=frame_idx, detections=detections)

        return detections

    #There are different tasks, so the model for the task gets called
    def default_step(self, batch: Any, task: str, detections: pd.DataFrame, metadata, **kwargs):

        #The model corresponding to the given task (model_name) is retrieved.
        model = self.models[task] #this could be yolo or bpbreid

        #A callback is called to mark the start of the processing step.
        self.callback(f"on_module_step_start", task=task, batch=batch)
        idxs, batch = batch
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        if model.level == "image":
            log.info(f"step : {idxs} {self.img_metadatas.index}")
            #Metadata for the current frame is packed into a DataFrame.
            batch_metadatas = pd.DataFrame([metadata])

            #If there are existing detections, they are filtered to include only those for the current frame.
            if len(detections) > 0:
                batch_input_detections = detections.loc[
                    np.isin(detections.image_id, batch_metadatas.index)
                ]
            else:
                batch_input_detections = detections

            #The model's process method is called with the batch, filtered detections, and metadata. 
            #This method performs the actual detection processing.
            batch_detections = self.models[task].process(
                batch,
                batch_input_detections,
                batch_metadatas)
        else:
            batch_detections = detections.loc[idxs]
            batch_detections = self.models[task].process(
                batch=batch,
                detections=batch_detections,
                metadatas=None,
                **kwargs,
            )
        #The new detections from the current processing step are merged with the existing detections DataFrame.
        detections = merge_dataframes(detections, batch_detections)

        #A callback is called to mark the end of the processing step.
        self.callback(
            f"on_module_step_end", task=task, batch=batch, detections=detections
        )
        return detections

