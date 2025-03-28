# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
import copy
import time
import torch
import logging
import datetime
import pstats
import cProfile
import numpy as np
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_vision import cv2_annotations_dir, giraffe_count_coverage_df_dir, metadata_path_processed
from configs.config_vision import init_model_dir, init_model_config_yaml_dir
from configs.config_vision import giraffe_count, data_random_sample, small_dataset_serials, experiment_keyname
from utils.helpers_matching import print_memory_usage, log_to_file, restore_stdout, load_data_dirs
from utils.utils_vision import load_metadata, merge_metadata_count_giraffes, choose_metadata_subset 
from utils.utils_vision import build_data_splits, save_image_grid_from_dataloader
from utils.utils_vision import set_up_data_splits


device = "cuda" if torch.cuda.is_available() else "cpu"

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
           
def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.ResizeShortestEdge(short_edge_length=(640, 640), max_size=1333),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict

def train_model(metadata, output_dir, init_model_config_yaml_dir, init_model_dir, visualize=None):
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_dir
    cfg.merge_from_file(init_model_config_yaml_dir)
    cfg.MODEL.WEIGHTS = init_model_dir
    cfg.DATASETS.TRAIN = ("giraffe_torso_train",)
    cfg.DATASETS.TEST = ("giraffe_torso_val",)
    cfg.TEST.EVAL_PERIOD = 50
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 8 # This is the real "batch size" commonly known to deep learning people
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 12 #128 # The "RoIHead batch size"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # only has one class (torso)
    cfg.MODEL.DEVICE = device

    # linear warm up and cosine annealing
    cfg.SOLVER.STEPS = [] # do not decay learning rate
    cfg.SOLVER.BASE_LR = 8e-4
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_ITERS = int(0.2*cfg.SOLVER.MAX_ITER)

    if visualize:
        # visualize data from dataloader to double check augmentation impact
        dataloader = build_detection_train_loader(cfg, shuffle=False)
        dataloader = build_detection_train_loader(cfg, mapper=custom_mapper, shuffle=False)
        save_image_grid_from_dataloader(dataloader, metadata, cfg.OUTPUT_DIR, rows=20, cols=5)
    
    # save config data
    cfg_file_path = os.path.join(output_dir, "config.yaml")
    with open(cfg_file_path, "w") as f:
        f.write(cfg.dump())
    
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
def main():

    # Call the profiling function
    profiler = cProfile.Profile()
    profiler.enable()
    
    # set up directories
    root_dir, _ = load_data_dirs()
    root_output_dir = os.path.join(root_dir, 'object_detection_output_dir')    
    output_dir = os.path.join(root_output_dir, 'models_' + experiment_keyname)
    os.makedirs(output_dir, exist_ok=True)
    
    cv2_annotations_dir_full = os.path.join(root_dir, cv2_annotations_dir)
    metadata_path_processed_full = os.path.join(root_dir, metadata_path_processed)
    giraffe_count_coverage_df_dir_full = os.path.join(root_dir, giraffe_count_coverage_df_dir)
    init_model_config_yaml_dir_full = os.path.join(root_dir, init_model_config_yaml_dir)
    init_model_dir_full = os.path.join(root_dir, init_model_dir)
    
    # Set up logging files
    log_file_std_output, log_file_err_output = log_to_file(output_dir, 'torso_model_trainig', subdir='')    
    
    # Load metadata and select a subset if needed
    metadata_df = load_metadata(metadata_path_processed_full)
    metadata_df = merge_metadata_count_giraffes(metadata_df, giraffe_count_coverage_df_dir_full)
    metadata_df = choose_metadata_subset(metadata_df, giraffe_count, data_random_sample, small_dataset_serials)
    train_df, val_df, test_df = build_data_splits(metadata_df)
    
    # Set up data splits in catalog
    DatasetCatalog, MetadataCatalog = set_up_data_splits(train_df, val_df, test_df, cv2_annotations_dir_full, root_dir)

    # Train model
    train_model(MetadataCatalog.get("giraffe_torso_train"), output_dir, init_model_config_yaml_dir_full, init_model_dir_full, visualize=False)

    # Print memory usage
    print_memory_usage()
    
    # Disabling the profiling function
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
    # Restore stdout 
    restore_stdout(log_file_std_output, log_file_err_output)

if __name__ == "__main__":
    main()