import os
import sys
sys.path.append('/media/brownradai/ssd_2t/covid_cxr/region_surv')
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ast import literal_eval
from typing import List, Dict
from torch import Tensor
from torch.utils.data import Dataset
# from path_datasets_and_weights import path_full_dataset
from configs.object_detector_config import *

class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms, load_img=True):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms
        self.load_img = load_img

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            
            # bbox_coordinates (List[List[int]]) is the 2nd column of the dataframes
            bbox_coordinates = self.dataset_df.iloc[index, 1]

            # bbox_labels (List[int]) is the 3rd column of the dataframes
            class_labels = self.dataset_df.iloc[index, 2]
            
            if self.load_img:
                # mimic_image_file_path is the 1st column of the dataframes
                image_path = self.dataset_df.iloc[index, 0]

                # cv2.imread by default loads an image with 3 channels
                # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            else:
                image = np.zeros(np.array(bbox_coordinates).max(0)[[3,2]], dtype=np.float32)

            # apply transformations to image, bboxes and label
            transformed = self.transforms(image=image, bboxes=bbox_coordinates, class_labels=class_labels)

            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            transformed_bbox_labels = transformed["class_labels"]

            sample = {
                "image": transformed_image,
                "boxes": torch.tensor(transformed_bboxes, dtype=torch.float),
                "labels": torch.tensor(transformed_bbox_labels, dtype=torch.int64),
            }
        except Exception:
            return None

        return sample

def collate_fn(batch: List[Dict[str, Tensor]]):
    # each dict in batch (which is a list) is for a single image and has the keys "image", "boxes", "labels"

    # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
    # otherwise, whole training loop will stop (even if only 1 image fails to open)
    batch = list(filter(lambda x: x is not None, batch))

    image_shape = batch[0]["image"].size()
    # allocate an empty images_batch tensor that will store all images of the batch
    images_batch = torch.empty(size=(len(batch), *image_shape))

    for i, sample in enumerate(batch):
        # remove image tensors from batch and store them in dedicated images_batch tensor
        images_batch[i] = sample.pop("image")

    # since batch (which is a list) now only contains dicts with keys "boxes" and "labels", rename it as targets
    targets = batch

    # create a new batch variable to store images_batch and targets
    batch_new = {}
    batch_new["images"] = images_batch
    batch_new["targets"] = targets

    return batch_new

def get_data_loaders(load_img=True, path_full_dataset_a=None):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")
    datasets_as_dfs = get_datasets_as_dfs(path_full_dataset_a)

    train_dataset = CustomImageDataset(datasets_as_dfs["train"], train_transforms, load_img)
    val_dataset = CustomImageDataset(datasets_as_dfs["valid"], val_transforms, load_img=True)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302
    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_datasets_as_dfs(path_full_dataset_a=None):
    if path_full_dataset_a != None:
        path_full_dataset = path_full_dataset_a
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels"]

    # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(path_full_dataset, dataset) + ".csv" for dataset in ["train", "valid"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    return datasets_as_dfs

if __name__=='__main__':
    path_dataset ='/media/brownradai/ssd_2t/covid_cxr/mimic-cxr-jpg-reports_full'
    train_loader, val_loader = get_data_loaders(load_img=False, path_full_dataset_a=path_dataset)
    from matplotlib import pyplot as plt
    x0y0x1y1_list = []
    for data in train_loader:
        # print(data)
        x0y0_l, x1y1_l = data['targets'][0]['boxes'][8][[0,1]], data['targets'][0]['boxes'][8][[2,3]]
        x0y0_r, x1y1_r = data['targets'][0]['boxes'][0][[0,1]], data['targets'][0]['boxes'][0][[2,3]]
        x0y0 = torch.min(torch.concat([x0y0_l[None], x0y0_r[None]], dim=0), dim=0)[0]
        x1y1 = torch.max(torch.concat([x1y1_l[None], x1y1_r[None]], dim=0), dim=0)[0]
        x0y0x1y1 = torch.concat([x0y0[None], x1y1[None]], dim=0)
        x0y0x1y1_list.append(x0y0x1y1)
        # x = x0y0x1y1[:,0]
        # y = x0y0x1y1[:,1]
        # plt.figure()
        # plt.imshow((data['images'][0,0]-data['images'][0,0].min())/(data['images'][0,0].max()-data['images'][0,0].min()))
        # plt.plot(x , y, 'o')
    coords = torch.concat(x0y0x1y1_list).reshape(-1, 4)
    coords_mid = coords.mean(0) # tensor([ 77.9209,  62.9912, 455.0607, 408.8989]) # recalulate 