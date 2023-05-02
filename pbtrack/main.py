import os
import torch
import hydra
from hydra.utils import instantiate
from pbtrack.datastruct import TrackerState
from pbtrack.pipeline import Pipeline
from pbtrack.utils import wandb

import logging

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )  # FIXME : why are we using too much file descriptors ?


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # For Hydra and Slurm compatibility
    set_sharing_strategy()  # Do not touch

    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO support Mac chips
    log.info(f"Using device: '{device}'.")

    wandb.init(cfg)

    # Initiate all the instances
    tracking_dataset = instantiate(cfg.dataset)
    detect_multi_model = instantiate(cfg.detect_multiple, device=device)
    detect_single_model = instantiate(cfg.detect_single, device=device)
    reid_model = instantiate(
        cfg.reid,
        tracking_dataset=tracking_dataset,
        device=device,
        model_detect=None,  # FIXME
    )
    track_model = instantiate(cfg.track, device=device)
    pipeline = Pipeline(
        models=[detect_multi_model, detect_single_model, reid_model, track_model]
    )

    evaluator = instantiate(cfg.eval)

    # FIXME, je pense qu'il faut repenser l'entrainement dans ce script
    # On peut pas entrainer 3 modèles dans un script quoi qu'il arrive
    # Il faut que l'entrainement soit fait dans un script propre à la librairie
    if cfg.train_detect:
        log.info("Training detection model.")
        detect_multi_model.train()

    if cfg.train_reid:
        log.info("Training reid model.")
        reid_model.train()

    if cfg.test_tracking:
        log.info(f"Starting tracking operation on {cfg.eval.test_set} set.")
        if cfg.eval.test_set == "train":
            tracking_set = tracking_dataset.train_set
        elif cfg.eval.test_set == "val":
            tracking_set = tracking_dataset.val_set
        else:
            tracking_set = tracking_dataset.test_set
        tracker_state = TrackerState(tracking_set, modules=pipeline, **cfg.state)
        # Run tracking and visualization
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )
        tracking_engine.track_dataset()
        # Evaluation
        if cfg.dataset.nframes == -1:
            if tracker_state.gt.detections is not None:
                log.info("Starting evaluation.")
                evaluator.run(tracker_state)
            else:
                log.warning(
                    "Skipping evaluation because there's no ground truth detection."
                )
        else:
            log.warning(
                "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
                "to -1)"
            )

        if tracker_state.save_file is not None:
            log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    return 0


if __name__ == "__main__":
    main()