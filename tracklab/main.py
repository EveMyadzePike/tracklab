import os
import rich.logging
import torch
import hydra #for config file management
import warnings
import logging

from tracklab.utils import monkeypatch_hydra, \
    progress  # needed to avoid complex hydra stacktraces when errors occur in "instantiate(...)"
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb


os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")



#add this decorator so that hydra knows to run the configuration before loading main function
#add config path and config name, so the decorator below points to the tracklab/configs/config.yaml
@hydra.main(version_base=None, config_path="pkg://tracklab.configs", config_name="config")
def main(cfg): #hydra loads the cfg file info and puts the data into the cfg object
    device = init_environment(cfg)

    # Instantiate all modules, using hydra.utils.instantiate, passes config info to create instance of a class with that config
    tracking_dataset = instantiate(cfg.dataset)
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)

    modules = []
    #pipeline is bbox detector, reid, tracker
    if cfg.pipeline is not None:
        for name in cfg.pipeline: #iterate through list
            module = cfg.modules[name]
            inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)
    #instantiates each module, so now have list of modules: yolov8, bpbreid, oc_sort

    #The Pipeline is a list of modules
    pipeline = Pipeline(models=modules)
    #instantiating the class causes the pipeline to list 
    #log.info("Pipeline: " + " -> ".join(model.name for model in self.models))


    #For the modules, these are all false or not set -- see yolov8.yaml, bpbreid.yaml, oc_sort.yaml
    # Train tracking modules
    for module in modules:
        if module.training_enabled:
            module.train()

    # Test tracking
    if cfg.test_tracking: #This is true
        log.info(f"Starting tracking operation on {cfg.dataset.eval_set} set.")


        #see soccer_net_gs.yaml for eval_set aka the valid set
        # Init tracker state and tracking engine
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]

        #create a datastuct named TrackerState, see tracker_state.py
        tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)

        #This gets called by Pipeline/module.py when Pipeline.validate is called
        #log.info(f"Pipeline has been validated")

        #create instance of class
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )

        #called by engine.py, can call track_dataset only because this is an instance of tracking_engine
        # Run tracking and visualization
        tracking_engine.track_dataset()

        # Evaluation
        evaluate(cfg, evaluator, tracker_state)


        #state is actually no save
        # Save tracker state
        if tracker_state.save_file is not None:
            log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    close_enviroment()

    return 0


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )


def init_environment(cfg):
    # For Hydra and Slurm compatibility
    progress.use_rich = cfg.use_rich #True in cfg file
    set_sharing_strategy()  # Do not touch, see definition above
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: '{device}'.")
    wandb.init(cfg) #weights and biases

    #seems to be just logging info
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_rich:
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.INFO)
    return device


def close_enviroment():
    wandb.finish()


def evaluate(cfg, evaluator, tracker_state):
    #eval_trackiong is True
    if cfg.get("eval_tracking", True) and cfg.dataset.nframes == -1:
        log.info("Starting evaluation.")
        evaluator.run(tracker_state)
    elif cfg.get("eval_tracking", True) == False:
        log.warning("Skipping evaluation because 'eval_tracking' was set to False.")
    else:
        log.warning(
            "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
            "to -1)"
        )


if __name__ == "__main__":
    main()
