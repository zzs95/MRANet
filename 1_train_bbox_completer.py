import shutil
from copy import deepcopy
import logging
import os
import random

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.object_detector.bbox_completer import BboxCompleter, prepare_bbox
from models.plotter import plot_gt_and_pred_bboxes_to_tensorboard
from path_datasets_and_weights import path_full_dataset, path_runs
# from models.losses import compute_intersection_and_union_area_per_class
from dataset.mimic.create_object_detector_dataloader import get_data_loaders
from utils.utils import write_config, seed_everything
from configs.bbox_completer_config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

seed_everything(SEED)

def get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken):

    model.eval()
    val_loss = 0.0
    num_images = 0

    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(val_dl)):
            # "targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
            # "boxes" maps to a tensor of shape [29 x 4] and "labels" maps to a tensor of shape [29]
            # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 29] (starting at 1, since 0 is background)
            images, targets = batch.values()
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            batch_size = len(targets)
            num_images += batch_size
            bboxes_masked, bboxes_gt, label_gt, bboxes_max_length = prepare_bbox(targets['bbox'], targets['label'], if_mask=True)

            # class_detected is a tensor of shape [batch_size x 29]
            class_detected = torch.ones_like(label_gt)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, detections = model(bboxes_masked, bboxes_gt, label_gt)
                loss = loss.mean()
            val_loss += loss.item() * batch_size
            detection_d = {}
            detection_d['top_region_boxes'] = detections.reshape(-1, 29, 4) * bboxes_max_length
            if batch_num == 0:
                plot_gt_and_pred_bboxes_to_tensorboard(writer, overall_steps_taken, images, detection_d, targets, class_detected, num_images_to_plot=2)
    val_loss /= len(val_dl)
    return val_loss


def log_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain loss: {train_loss:.3f}")
    log.info(f"\tVal loss: {val_loss:.3f}")


def train_model(
    model,
    train_dl,
    val_dl,
    optimizer,
    scaler,
    lr_scheduler,
    epochs,
    weights_folder_path,
    writer
):
    lowest_val_loss = np.inf
    # the best_model_state is the one where the val loss is the lowest overall
    best_model_state = None

    overall_steps_taken = 0  # for logging to tensorboard

    for epoch in range(epochs):
        log.info(f"Training epoch {epoch}!")

        train_loss = 0.0
        steps_taken = 0
        for num_batch, batch in tqdm(enumerate(train_dl)):
            # batch is a dict with keys "images" and "targets"
            # _, targets = batch.values()
            targets = batch['targets']
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            targets = {'bbox':[t['boxes'] for t in targets], 'label':[ t['labels'] for t in targets]}
            batch_size = len(targets)
            bboxes_masked, bboxes_gt, label_gt, _ = prepare_bbox(targets['bbox'], targets['label'], if_mask=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(bboxes_masked, bboxes_gt, label_gt)
            loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item() * batch_size
            steps_taken += 1
            overall_steps_taken += 1

            # evaluate every k steps and also at the end of an epoch
            if steps_taken >= EVALUATE_EVERY_K_STEPS or (num_batch + 1) == len(train_dl):
                log.info(f"Evaluating at step {overall_steps_taken}!")

                # normalize the train loss by steps_taken
                train_loss /= steps_taken

                val_loss = get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken)

                writer.add_scalars("_loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)

                current_lr = float(optimizer.param_groups[0]["lr"])
                writer.add_scalar("lr", current_lr, overall_steps_taken)

                log.info(f"Metrics evaluated at step {overall_steps_taken}!")

                # set the model back to training
                model.train()

                # decrease lr if val loss has not decreased after certain number of evaluations
                lr_scheduler.step(val_loss)

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_epoch = epoch
                    best_model_save_path = os.path.join(
                        weights_folder_path, f"val_loss_{lowest_val_loss:.3f}_epoch_{epoch}.pth"
                    )
                    best_model_state = deepcopy(model.state_dict())

                # log to console at the end of an epoch
                if (num_batch + 1) == len(train_dl):
                    log_stats_to_console(train_loss, val_loss, epoch)

                # reset values
                train_loss = 0.0
                steps_taken = 0

        # save the current best model weights at the end of each epoch
        torch.save(best_model_state, best_model_save_path)

    log.info("Finished training!")
    log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None

def get_model():
    model = BboxCompleter()
    model.to(device, non_blocking=True)
    model.train()
    return model

def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path = os.path.join(path_runs, 'object_detector', f"run_{RUN}")
    weights_folder_path = os.path.join(run_folder_path, "weights")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")

    if os.path.exists(run_folder_path):
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        log.error("Delete the folder or change the run number.")
        shutil.rmtree(run_folder_path)
        # return None

    os.mkdir(run_folder_path)
    os.mkdir(weights_folder_path)
    os.mkdir(tensorboard_folder_path)

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_parameters = {
        "RUN": RUN,
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
        "BATCH_SIZE": BATCH_SIZE,
        "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "EVALUATE_EVERY_K_STEPS": EVALUATE_EVERY_K_STEPS,
        "PATIENCE_LR_SCHEDULER": PATIENCE_LR_SCHEDULER,
        "THRESHOLD_LR_SCHEDULER": THRESHOLD_LR_SCHEDULER,
        "FACTOR_LR_SCHEDULER": FACTOR_LR_SCHEDULER,
        "COOLDOWN_LR_SCHEDULER": COOLDOWN_LR_SCHEDULER
    }

    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    return weights_folder_path, tensorboard_folder_path, config_file_path, config_parameters


def main():
    weights_folder_path, tensorboard_folder_path, config_file_path, config_parameters = create_run_folder()
    train_loader, val_loader = get_data_loaders(load_img=False, path_full_dataset_a=path_dataset, SEED=SEED, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, IMAGE_INPUT_SIZE=IMAGE_INPUT_SIZE,
                                                PERCENTAGE_OF_TRAIN_SET_TO_USE=PERCENTAGE_OF_TRAIN_SET_TO_USE, PERCENTAGE_OF_VAL_SET_TO_USE=PERCENTAGE_OF_VAL_SET_TO_USE)
    log.info(f"Train: {len(train_loader.dataset)} images")
    log.info(f"Val: {len(val_loader.dataset)} images")
    config_parameters["TRAIN NUM IMAGES"] = len(train_loader.dataset)
    config_parameters["VAL NUM IMAGES"] = len(val_loader.dataset)
    write_config(config_file_path, config_parameters)

    model = get_model()
    if MULTI_GPU:
        model = nn.DataParallel(model)
    model.train()

    scaler = torch.cuda.amp.GradScaler()

    opt = AdamW(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=FACTOR_LR_SCHEDULER, patience=PATIENCE_LR_SCHEDULER, threshold=THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)
    writer = SummaryWriter(log_dir=tensorboard_folder_path)
    log.info("\nStarting training!\n")

    train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        optimizer=opt,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        epochs=EPOCHS,
        weights_folder_path=weights_folder_path,
        writer=writer
    )


if __name__ == "__main__":
    main()
