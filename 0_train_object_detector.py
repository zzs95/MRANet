import shutil
from copy import deepcopy
import logging
import os
import random

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dataset.constants import ANATOMICAL_REGIONS

from models.object_detector.object_detector import ObjectDetector
from models.plotter import plot_gt_and_pred_bboxes_to_tensorboard
from path_datasets_and_weights import path_full_dataset, path_runs
from models.losses import compute_intersection_and_union_area_per_class
from dataset.mimic.create_object_detector_dataloader import get_data_loaders
from utils.utils import write_config, seed_everything
from configs.object_detector_config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

seed_everything(SEED)

def get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken):
    """
    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.
        writer (tensorboardX.SummaryWriter.writer): Writer used to plot gt and predicted bboxes of first couple of image in val set
        overall_steps_taken: for tensorboard

    Returns:
        val_loss (float): val loss for val set
        avg_num_detected_classes_per_image (float): since it's possible that certain classes/regions of all 29 regions are not detected in an image,
        this metric counts how many classes are detected on average for an image. Ideally, this number should be 29.0
        avg_detections_per_class (list[float]): this metric counts how many times a class was detected in an image on average. E.g. if the value is 1.0,
        then the class was detected in all images of the val set
        avg_iou_per_class (list[float]): average IoU per class computed over all images in val set
    """
    # PyTorch implementation only return losses in train mode, and only detections in eval mode
    # see https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn/65347721#65347721
    # my model is modified to return losses, detections and class_detected in eval mode
    # see forward method of object detector class for more information
    model.eval()

    val_loss = 0.0

    num_images = 0

    # tensor for accumulating the number of times a class is detected over all images (will be divided by num_images at the end of get average)
    sum_class_detected = torch.zeros(29, device=device)

    # tensor for accumulating the intersection area of each class (will be divided by union area of each class at the end of get the IoU for each class)
    sum_intersection_area_per_class = torch.zeros(29, device=device)

    # tensor for accumulating the union area of each class (will divide the intersection area of each class at the end of get the IoU for each class)
    sum_union_area_per_class = torch.zeros(29, device=device)

    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(val_dl)):
            # "targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
            # "boxes" maps to a tensor of shape [29 x 4] and "labels" maps to a tensor of shape [29]
            # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 29] (starting at 1, since 0 is background)
            images, targets = batch.values()

            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            # detections is a dict with keys "top_region_boxes" and "top_scores"
            # "top_region_boxes" maps to a tensor of shape [batch_size x 29 x 4]
            # "top_scores" maps to a tensor of shape [batch_size x 29]

            # class_detected is a tensor of shape [batch_size x 29]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict, detections, class_detected = model(images, targets)

            # sum up all 4 losses
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item() * batch_size

            # sum up detections for each class
            sum_class_detected += torch.sum(class_detected, dim=0)

            # compute intersection and union area for each class and add them to the sum
            intersection_area_per_class, union_area_per_class = compute_intersection_and_union_area_per_class(detections, targets, class_detected)
            sum_intersection_area_per_class += intersection_area_per_class
            sum_union_area_per_class += union_area_per_class

            if batch_num == 0:
                plot_gt_and_pred_bboxes_to_tensorboard(writer, overall_steps_taken, images, detections, targets, class_detected, num_images_to_plot=2)

    val_loss /= len(val_dl)
    avg_num_detected_classes_per_image = torch.sum(sum_class_detected / num_images).item()
    avg_detections_per_class = (sum_class_detected / num_images).tolist()
    avg_iou_per_class = (sum_intersection_area_per_class / sum_union_area_per_class).tolist()

    return val_loss, avg_num_detected_classes_per_image, avg_detections_per_class, avg_iou_per_class


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
    """
    Train a model on train set and evaluate on validation set.
    Saves best model w.r.t. val loss.

    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
    train_dl: torch.utils.data.Dataloder
        The train dataloader to train on.
    val_dl: torch.utils.data.Dataloder
        The val dataloader to validate on.
    optimizer: Optimizer
        The model's optimizer.
    lr_scheduler: torch.optim.lr_scheduler
        The learning rate scheduler to use.
    epochs: int
        Number of epochs to train for.
    patience: int
        Number of epochs to wait for val loss to decrease.
        If patience is exceeded, then training is stopped early.
    weights_folder_path: str
        Path to folder where best weights will be saved.
    writer: torch.utils.tensorboard.SummaryWriter
        Writer for logging values to tensorboard.

    Returns
    -------
    None, but saves model with the overall lowest val loss at the end of every epoch.
    """
    lowest_val_loss = np.inf

    # the best_model_state is the one where the val loss is the lowest overall
    best_model_state = None

    overall_steps_taken = 0  # for logging to tensorboard

    # for gradient accumulation
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE

    for epoch in range(epochs):
        log.info(f"Training epoch {epoch}!")

        train_loss = 0.0
        steps_taken = 0
        for num_batch, batch in tqdm(enumerate(train_dl)):
            # batch is a dict with keys "images" and "targets"
            images, targets = batch.values()

            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)

                # sum up all 4 losses
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()

            if (num_batch + 1) % ACCUMULATION_STEPS == 0:
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

                val_loss, avg_num_detected_classes_per_image, avg_detections_per_class, avg_iou_per_class = get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken)

                writer.add_scalars("_loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)
                writer.add_scalar("avg_num_predicted_classes_per_image", avg_num_detected_classes_per_image, overall_steps_taken)

                # replace white space by underscore for each region name (i.e. "right upper lung" -> "right_upper_lung")
                anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]

                for class_, avg_detections_class in zip(anatomical_regions, avg_detections_per_class):
                    writer.add_scalar(f"num_preds_{class_}", avg_detections_class, overall_steps_taken)

                for class_, avg_iou_class in zip(anatomical_regions, avg_iou_per_class):
                    writer.add_scalar(f"iou_{class_}", avg_iou_class, overall_steps_taken)

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

def get_model(load_pretrain=False):
    if CHECKPOINT == None :
        checkpoint = CHECKPOINT
    else:
        checkpoint = torch.load(CHECKPOINT, map_location=device)

    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    # checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
    # checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")

    model = ObjectDetector(return_feature_vectors=False)
    model.to(device, non_blocking=True)

    if checkpoint and load_pretrain:
        # model.load_state_dict(checkpoint["model"])
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in checkpoint["model"].items():
            if 'object_detector' in k:
                name = k[16:] # remove `object_detector.`
                print(name)
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.train()

    return model

def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path = os.path.join(path_runs,'object_detector', f"run_{RUN}")
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
    train_loader, val_loader = get_data_loaders(load_img=True, path_full_dataset_a=path_full_dataset, SEED=SEED, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, IMAGE_INPUT_SIZE=IMAGE_INPUT_SIZE,
                                            PERCENTAGE_OF_TRAIN_SET_TO_USE=PERCENTAGE_OF_TRAIN_SET_TO_USE, PERCENTAGE_OF_VAL_SET_TO_USE=PERCENTAGE_OF_VAL_SET_TO_USE)
    log.info(f"Train: {len(train_loader.dataset)} images")
    log.info(f"Val: {len(val_loader.dataset)} images")
    config_parameters["TRAIN NUM IMAGES"] = len(train_loader.dataset)
    config_parameters["VAL NUM IMAGES"] = len(val_loader.dataset)
    write_config(config_file_path, config_parameters)

    model = get_model(load_pretrain=True)
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
