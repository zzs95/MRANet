import shutil
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from tqdm import tqdm
import sys
sys.path.append(os.getcwd)
from collections import OrderedDict
from models.object_detector.object_detector import ObjectDetector
from models.object_detector.bbox_completer import BboxCompleter, prepare_bbox
from models.plotter import plot_gt_and_pred_bboxes_to_tensorboard
from path_datasets_and_weights import path_runs
from dataset.create_image_report_dataloader import get_data_loaders
from utils.utils import write_config, seed_everything
from configs.extract_feat_config import *
from utils.file_and_folder_operations import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

seed_everything(SEED)
def sort_coordinates(box):
    box = box[np.argsort(box[:,0])]
    left_points = box[:2]
    left_points = left_points[np.argsort(left_points[:,1])]
    
    right_points = box[2:]
    right_points = right_points[np.argsort(right_points[:,1])]
    
    sorted_points = [left_points[0], left_points[1], right_points[1], right_points[0]]
    
    return sorted_points

def testing(
    model,
    completer,
    val_dl,
    writer
):
    overall_steps_taken = 0  # for logging to tensorboard
    model.eval()
    img_aligned_all = []
    bboxes_all = []
    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(val_dl)):
            print(batch_num)
            images = batch['images']
            batch_size = images.size(0)
            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
            output = images
            targets = [{'boxes': np.array( [[0,0,512,512]] * 29).reshape(29, 4)}] * batch_size

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                losses, detections, top_region_features, class_detected, global_feature = model(images)

                # fix predicted roi bboxes
                pred_bbox = detections['top_region_boxes']
                pred_label = class_detected
                bboxes_masked, bboxes_gt, label_gt, bboxes_max_length = prepare_bbox(pred_bbox, pred_label, if_mask=False)
                bbox_fix = completer.predict(bboxes_masked)
                bbox_fix = bbox_fix * bboxes_max_length
                pred_bbox_fix = torch.clone(pred_bbox)
                pred_bbox_fix[~pred_label] = bbox_fix[~pred_label]
                
                # manual correct
                x_err = (pred_bbox_fix[:,:,3] - pred_bbox_fix[:,:,1]) <= 0
                y_err = (pred_bbox_fix[:,:,2] - pred_bbox_fix[:,:,0]) <= 0
                x_err_idx = np.where(x_err[0].cpu().numpy())[0].tolist()
                y_err_idx = np.where(y_err[0].cpu().numpy())[0].tolist()
                if 25 in y_err_idx:
                    pred_bbox_fix[0,25,2] = pred_bbox_fix[0,23,2] 
                    
                if 13 in x_err_idx:
                    pred_bbox_fix[0,13,3] = pred_bbox_fix[0,5,3] 
                    if pred_bbox_fix[0,13,1] == 0:
                        pred_bbox_fix[0,13,1] = pred_bbox_fix[0,5,1] 
                print(x_err_idx, y_err_idx)
                
            detections['top_region_boxes'] = pred_bbox_fix
            plot_gt_and_pred_bboxes_to_tensorboard(writer, overall_steps_taken, images, detections, targets, torch.ones_like(class_detected).bool(), num_images_to_plot=1)
            overall_steps_taken += 1
            img_aligned_all.append(output[0].detach().cpu())
            bboxes_all.append(pred_bbox_fix.detach().cpu())
            
    img_aligned_all = torch.concat(img_aligned_all)
    bboxes_all = torch.concat(bboxes_all)
    return img_aligned_all, bboxes_all, 

def get_model():
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    checkpoint_completer = torch.load(CHECKPOINT_completer, map_location=device)

    model = ObjectDetector(return_feature_vectors=True)
    model.to(device, non_blocking=True)

    completer = BboxCompleter()
    completer.to(device, non_blocking=True)
    

    # model.load_state_dict(checkpoint["model"])
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in checkpoint["model"].items():
        if 'object_detector' in k:
            name = k[16:] # remove `object_detector.`
            # print(name)
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    completer.load_state_dict(checkpoint_completer)
    return model, completer


def create_run_folder(setname):
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path = os.path.join(path_runs, 'object_detector', f"run_{RUN}", setname)
    extracted_feats_folder_path = os.path.join(run_folder_path, "predicted_bboxes")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")

    if os.path.exists(run_folder_path):
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        log.error("Delete the folder or change the run number.")
        shutil.rmtree(run_folder_path)
        # return None

    maybe_mkdir_p(run_folder_path)
    maybe_mkdir_p(extracted_feats_folder_path)
    maybe_mkdir_p(tensorboard_folder_path)

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_parameters = {
        "RUN": RUN,
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
    }

    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    return extracted_feats_folder_path, tensorboard_folder_path, config_file_path, config_parameters

def main():
    for setname in ['brown', 'penn']:
    # setname = 'brown'
    # setname = 'penn'
        extracted_feats_folder_path, tensorboard_folder_path, config_file_path, config_parameters = create_run_folder(setname)
    
        all_loader, gpt_tokenizer = get_data_loaders(setname=setname, return_all=True, batch_size=1, do_ignore=False)
        log.info(f"All data: {len(all_loader.dataset)} images")
        config_parameters["ALL NUM IMAGES"] = len(all_loader.dataset)
        write_config(config_file_path, config_parameters)

        model, completer = get_model()
        writer = SummaryWriter(log_dir=tensorboard_folder_path)
        log.info("\nStarting training!\n")

        img_aligned_all, bboxes_all = testing(
            model=model,
            completer=completer,
            val_dl=all_loader,
            writer=writer
        )
        bboxes_all = bboxes_all.numpy()
        np.save(join(extracted_feats_folder_path, setname+'_bboxes.npy'), bboxes_all.astype(np.float16))
        bboxes_all = np.load(join(extracted_feats_folder_path, setname+'_bboxes.npy'))
        x = bboxes_all[:,:,2] - bboxes_all[:,:,0]
        y = bboxes_all[:,:,3] - bboxes_all[:,:,1]
        x_err = np.sum(x<=0, axis=1)
        x_err_idx = np.argwhere(x_err)[:,0]
        y_err = np.sum(y<=0, axis=1)
        y_err_idx = np.argwhere(y_err)[:,0]
        for i in y_err_idx:
            err_idx = np.argwhere(y[i]<=0)[0]
            # print(err_idx)
            for j in err_idx:
                if j == 13:
                    bboxes_all[i, j, 3] = bboxes_all[i, 5, 3]
                    bboxes_all[i, j, 1] = bboxes_all[i, 5, 1]
        
        low_bround = bboxes_all[:,:,[0,1]]
        up_bround = bboxes_all[:,:,[2,3]]
        up_bround_err_idx = np.argwhere(up_bround>512)
        bboxes_all[up_bround_err_idx[:,0], up_bround_err_idx[:,1], up_bround_err_idx[:,2]+2] = 512
        
        np.save(join(extracted_feats_folder_path, setname+'_bboxes.npy'), bboxes_all.astype(np.float16))
        
if __name__ == "__main__":
    main()
