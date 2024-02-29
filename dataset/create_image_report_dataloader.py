import os
import sys
sys.path.append('/media/brownradai/ssd_2t/covid_cxr/MRANet/')
import random
import torch
import numpy as np
import pandas as pd
from datasets import Dataset as Dataset_table
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from path_datasets_and_weights import path_internal_dataset
from models.tokenizer import get_gpt_tokenizer as get_tokenizer
from dataset.constants import REPORT_KEYS, CLIN_KEYS, REPORT_KEYS_raw, brown_IMAGE_TO_IGNORE
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_name: str, tokenized_dataset, transforms, feats):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenized_dataset = tokenized_dataset
        self.transforms = transforms
        self.image_all = feats['img_all']
        self.text_feats_all = feats['text_feats_all']
        self.coords_all = feats['coord_all']

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            clin_report = self.tokenized_dataset[index]
            # mimic_image_file_path is the 1st column of the dataframes
            image_indx = int(clin_report['index'])
            
            image = self.image_all[image_indx]
            text_feats = self.text_feats_all[image_indx]
            
            # bbox_coordinates (List[List[int]]) is the 2nd column of the dataframes
            bbox_coordinates = self.coords_all[image_indx] # List[List[int]] x1 > x0, y1 > y0,
            # bbox_coordinates = np.array([[0,0,512,512] ] * 29) # List[List[int]] # TODO,
            class_labels = np.arange(0, 29)
            # bbox_coordinates = bbox_coordinates / 512
            # apply transformations to image, bboxes and label
            transformed = self.transforms(image=image.astype(np.float32), bboxes=bbox_coordinates, class_labels=class_labels)
            transformed_image = transformed["image"]
            transformed_bbox_coordinates = transformed["bboxes"]
            transformed_bbox_labels = transformed["class_labels"]
            
            bbox_phrases = []
            reference_report = ''
            if REPORT_KEYS_raw[0] in clin_report.keys():
                bbox_phrases = [clin_report[k] for k in REPORT_KEYS]
                for k in REPORT_KEYS_raw:
                    reference_report += clin_report[k] 
                    reference_report += ' '         
                       
            event_labels = clin_report['death'] 
            time_labels = clin_report['days'] 
 
            clin_feat = np.array([clin_report[k] for k in CLIN_KEYS])
            sample = {
                'idx': image_indx,
                "image": transformed_image,
                "bbox_coordinates": torch.tensor(transformed_bbox_coordinates, dtype=torch.float),
                "bbox_phrases": bbox_phrases,
                "text_feats": torch.from_numpy(text_feats.astype(np.float32)),
                "reference_report": reference_report,
                "event_label": torch.tensor(event_labels, dtype=torch.bool),
                "time_label": torch.tensor(time_labels, dtype=torch.float),
                "clin_feat": torch.tensor(clin_feat, dtype=torch.float)
            }
            
            if "input_ids" in clin_report.keys():
                sample["input_ids"] = clin_report["input_ids"]  # List[List[int]]
                sample["attention_mask"] = clin_report["attention_mask"]  # List[List[int]]
        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_indx}")
            self.log.error(f"Reason: {e}")
            return None

        return sample

class CustomCollator():
    def __init__(self, tokenizer, is_val_or_test):
        self.tokenizer = tokenizer
        self.is_val_or_test = is_val_or_test
        
    def __call__(self, batch: list[dict[str]]):
        """
        batch is a list of dicts where each dict corresponds to a single image and has the keys:
          - image
          - bbox_coordinates
          - bbox_labels
          - input_ids
          - attention_mask
          - bbox_phrase_exists
          - bbox_is_abnormal

        For the val and test datasets, we have the additional key:
          - bbox_phrases
          - reference_report
        """

        # discard samples from batch where __getitem__ from custom_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None
        batch_size = len(batch)

        # allocate an empty tensor images_batch that will store all images of the batch
        image_size = batch[0]["image"].size()
        images_batch = torch.empty(size=(batch_size, *image_size))
        text_feats_size = batch[0]["text_feats"].size()
        text_feats_batch = torch.empty(size=(batch_size, *text_feats_size))
        idx_batch = torch.empty(size=(batch_size, ), dtype=int)
        time_batch = torch.empty(size=(batch_size, ))
        event_batch = torch.empty(size=(batch_size, ))
        clin_feat_batch = torch.empty(size=(batch_size, 16))
        # create an empty list image_targets that will store dicts containing the bbox_coordinates and bbox_labels
        boxes_batch = []
        
        # for a validation and test batch, create a List[List[str]] that hold the reference phrases (i.e. bbox_phrases) to compute e.g. BLEU scores
        # the inner list will hold all reference phrases for a single image
        bbox_phrases_batch = []

        # also create a List[str] to hold the reference reports for the images in the batch
        reference_reports = []
        for i, sample_dict in enumerate(batch):
            # remove image tensors from batch and store them in dedicated images_batch tensor
            idx_batch[i] = sample_dict.pop("idx")
            images_batch[i] = sample_dict.pop("image")
            text_feats_batch[i] = sample_dict.pop("text_feats")
            # remove bbox_coordinates and bbox_labels and store them in list image_targets
            boxes_batch.append(sample_dict.pop("bbox_coordinates"))
            time_batch[i] = sample_dict.pop("time_label")
            event_batch[i] = sample_dict.pop("event_label")
            clin_feat_batch[i] = sample_dict.pop("clin_feat")

            # remove list bbox_phrases from batch and store it in the list bbox_phrases_batch
            bbox_phrases_batch.append(sample_dict.pop("bbox_phrases"))

            # same for reference_report
            reference_reports.append(sample_dict.pop("reference_report"))      

        if "input_ids" in batch[0].keys():
            dict_with_ii_and_am = self.transform_to_dict_with_inputs_ids_and_attention_masks(batch)
            dict_with_ii_and_am = self.tokenizer.pad(dict_with_ii_and_am, padding="longest", return_tensors="pt")
            batch = dict_with_ii_and_am  
        else:
            batch = {}
        
        # add the remaining keys and values to the batch dict
        batch["idxs"] = idx_batch
        batch["images"] = images_batch
        batch["boxes"] = boxes_batch
        batch["clin_feat"] = clin_feat_batch
        batch["text_feats"] = text_feats_batch
        batch["reference_sentences"] = bbox_phrases_batch
        batch["reference_reports"] = reference_reports       
        batch["time_label"] = time_batch
        batch["event_label"] = event_batch
        return batch

    def transform_to_dict_with_inputs_ids_and_attention_masks(self, batch):
        dict_with_ii_and_am = {"input_ids": [], "attention_mask": []}
        for single_dict in batch:
            for key, outer_list in single_dict.items():
                if key in list(dict_with_ii_and_am.keys()):
                    for inner_list in outer_list:
                        dict_with_ii_and_am[key].append(inner_list)

        return dict_with_ii_and_am

def get_tokenized_datasets(tokenizer, raw_dataset_list):
    def tokenize_function(example ):
        keys_list = REPORT_KEYS
        
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + example[k] + eos_token for k in keys_list]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)
    
    raw_dataset_list_new = []
    for raw_dataset in raw_dataset_list:
        raw_dataset_new = raw_dataset.map(tokenize_function)
        raw_dataset_list_new.append(raw_dataset_new)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #
    #   val dataset will have additional column:
    #   - reference_report (str)

    return raw_dataset_list_new


def get_data_loaders(setname='brown', batch_size=2, image_input_size=512, return_all=False, DDP=False, random_state_i=42, worker_seed=None, is_token=False, text_feat='bert', do_ignore=True):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    if worker_seed==None:
        g.manual_seed(random_state_i)
    else:
        g.manual_seed(worker_seed)
    datasets_as_dfs, feats = get_datasets_as_dfs(setname, random_state_i, text_feat, do_ignore)
    raw_train_dataset = Dataset_table.from_pandas(datasets_as_dfs["train"])
    raw_val_dataset = Dataset_table.from_pandas(datasets_as_dfs["valid"])
    raw_test_dataset = Dataset_table.from_pandas(datasets_as_dfs["test"])
    gpt_tokenizer = get_tokenizer()
    
    train_transforms = get_transforms(setname, image_input_size, "train")
    val_transforms = get_transforms(setname, image_input_size, "val")
    if return_all:
        raw_all_dataset = Dataset_table.from_pandas(datasets_as_dfs["all"])
        all_dataset_complete = CustomDataset("test", raw_all_dataset, val_transforms, feats)
        custom_collate_test = CustomCollator(tokenizer=gpt_tokenizer, is_val_or_test=True)
        all_loader = DataLoader(
            all_dataset_complete,
            collate_fn=custom_collate_test,
            # batch_size=len(all_dataset_complete),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # could also be set to NUM_WORKERS, but I had some problems with the val loader stopping sometimes when num_workers != 0
            pin_memory=False,
        )
        return all_loader, gpt_tokenizer
    
    if is_token:
        [tokenized_train_dataset, tokenized_val_dataset] = get_tokenized_datasets(gpt_tokenizer, [raw_train_dataset, raw_val_dataset] )
    else:
        tokenized_train_dataset = raw_train_dataset
        tokenized_val_dataset = raw_val_dataset
    train_dataset_complete = CustomDataset("train", tokenized_train_dataset, train_transforms, feats)
    val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms, feats)
    # train_dataset_complete.__getitem__(0)
    custom_collate_train = CustomCollator(tokenizer=gpt_tokenizer, is_val_or_test=False)
    custom_collate_val = CustomCollator(tokenizer=gpt_tokenizer, is_val_or_test=True)
    
    target = torch.Tensor(np.array(raw_train_dataset.data['death'])).long()
    class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
    mean_weight = 1. / class_sample_count.float()
    reverse_weight = mean_weight / class_sample_count.float()
    mean_samples_weight = torch.tensor([mean_weight[t] for t in target])
    reverse_samples_weight = torch.tensor([reverse_weight[t] for t in target])
    mean_sampler = torch.utils.data.WeightedRandomSampler(mean_samples_weight, len(mean_samples_weight))
    reverse_sampler = torch.utils.data.WeightedRandomSampler(reverse_samples_weight, len(reverse_samples_weight))
    if DDP:
        train_sampler = DistributedSampler(mean_sampler) # DDP
    else:
        train_sampler = mean_sampler # DP
    # train_sampler = None
    train_loader = DataLoader(
        train_dataset_complete,
        collate_fn=custom_collate_train,
        # batch_size=len(train_dataset_complete),
        batch_size=batch_size,
        # shuffle=True,
        num_workers=32,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        sampler=train_sampler
    )
    # target = torch.Tensor(np.array(raw_val_dataset.data['death'])).long()
    # class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
    # mean_weight = 1. / class_sample_count.float()
    # reverse_weight = mean_weight / class_sample_count.float()
    # mean_samples_weight = torch.tensor([mean_weight[t] for t in target])
    # reverse_samples_weight = torch.tensor([reverse_weight[t] for t in target])
    # mean_sampler_val = torch.utils.data.WeightedRandomSampler(mean_samples_weight, len(mean_samples_weight))
    # reverse_sampler = torch.utils.data.WeightedRandomSampler(reverse_samples_weight, len(reverse_samples_weight))
    # # val_sampler = DistributedSampler(mean_sampler_val) # DDP do not val
    # val_sampler =mean_sampler_val
    val_loader = DataLoader(
        val_dataset_complete,
        collate_fn=custom_collate_val,
        # batch_size=len(val_dataset_complete),
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,  # could also be set to NUM_WORKERS, but I had some problems with the val loader stopping sometimes when num_workers != 0
        pin_memory=False,
        drop_last=False,
        sampler=None
    )
    if is_token:
        [tokenized_test_dataset] = get_tokenized_datasets(gpt_tokenizer, [raw_test_dataset] )
    else:
        tokenized_test_dataset = raw_test_dataset
    test_dataset_complete = CustomDataset("test", tokenized_test_dataset, val_transforms, feats)
    custom_collate_test = CustomCollator(tokenizer=gpt_tokenizer, is_val_or_test=True)
    test_loader = DataLoader(
        test_dataset_complete,
        collate_fn=custom_collate_test,
        # batch_size=len(test_dataset_complete),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # could also be set to NUM_WORKERS, but I had some problems with the val loader stopping sometimes when num_workers != 0
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader, gpt_tokenizer, train_sampler

    
def get_transforms(setname, image_input_size, dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    if setname == 'brown':
        mean = 0.48909846 # brown 
        std = 0.20668842
    elif setname == 'penn':
        mean = 0.47929397 # penn
        std = 0.18900576 # penn
    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to image_input_size, and the short edge of the image to be padded to image_input_size on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (image_input_size x image_input_size)
            # LongestMaxSize: resizes the longer edge to image_input_size while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=image_input_size, interpolation=cv2.INTER_AREA),
            # A.ColorJitter(hue=0.0),
            # A.GaussNoise(var_limit=1),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=image_input_size, min_width=image_input_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=image_input_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=image_input_size, min_width=image_input_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_datasets_as_dfs(setname='brown', random_state_i=42, text_feat='bert', do_ignore=True):
    usecols = ["days", "death", ]
    usecols += CLIN_KEYS
    usecols += REPORT_KEYS
    usecols += REPORT_KEYS_raw
    feats = {}
    feats['img_all'] = np.load(os.path.join(path_internal_dataset, setname+'_imgs_255_corr.npy')).astype(np.float32) 
    try:
        feats['coord_all'] = np.load(os.path.join(path_internal_dataset, setname+'_bboxes.npy')).astype(np.float32) 
    except:
        feats['coord_all'] = np.repeat(np.array( [[0,0,512,512]] * 29)[None], repeats=len(feats['img_all']), axis=0).astype(np.float32) 
    if setname == 'brown':
        datasets_df = pd.read_excel(os.path.join(path_internal_dataset, setname+'_table_w_report_split_corr.xlsx'), usecols=usecols)
        if text_feat == 'bert':
            # feats['text_feats_all'] = np.load(os.path.join(path_internal_dataset, setname+'_5text_feats_gatortron_base.npy')).astype(np.float32)
            feats['text_feats_all'] = np.load(os.path.join(path_internal_dataset, setname+'_5text_feats_gatortron_medium.npy')).astype(np.float32)
        elif text_feat == 'img2text':
            feats['text_feats_all'] = np.load(os.path.join(path_internal_dataset, setname+'_img2text_feat.npy')).astype(np.float32)
        if do_ignore:
            brown_remain_idx = list(set(np.arange(len(feats['img_all'])).tolist()) - set(brown_IMAGE_TO_IGNORE))
            for k in feats.keys():
                feats[k] = feats[k][brown_remain_idx]
            datasets_df = datasets_df.iloc[brown_remain_idx]
    elif setname=='penn':
        datasets_df = pd.read_excel(os.path.join(path_internal_dataset, setname+'_data_corr.xlsx'), index_col=0)
        if text_feat == 'bert':
            feats['text_feats_all'] = np.zeros([len(datasets_df), 1] )
        elif text_feat == 'img2text':
            feats['text_feats_all'] = np.load(os.path.join(path_internal_dataset, setname+'_img2text_feat.npy')).astype(np.float32)
    
    datasets_df = datasets_df.reset_index(drop=False) # for brown_IMAGE_TO_IGNORE
    datasets_df.rename(columns={'index':'index_original'}, inplace=True)
    datasets_df['clin_feat_2'] = datasets_df['clin_feat_2'] / 138
    datasets_df['days'] = datasets_df['days'] / 77.025
    datasets_df_shuffled = datasets_df.sample(frac=1, random_state=random_state_i, replace=False).reset_index(drop=False)
    X_df_D = datasets_df_shuffled.loc[datasets_df_shuffled['death'] == 1]
    X_df_noD = datasets_df_shuffled.loc[datasets_df_shuffled['death'] == 0]
    Xtr_D, Xval_D, Xts_D = np.split(X_df_D.sample(frac=1, random_state=random_state_i, replace=False), [int(.7 * len(X_df_D)), int(.8 * len(X_df_D))])
    Xtr_noD, Xval_noD, Xts_noD = np.split(X_df_noD.sample(frac=1, random_state=random_state_i, replace=False), [int(.7 * len(X_df_noD)), int(.8 * len(X_df_noD))])
    Xtr_df = pd.merge(Xtr_D, Xtr_noD, how='outer').sample(frac=1, random_state=random_state_i, replace=False)
    Xts_df = pd.merge(Xts_D, Xts_noD, how='outer').sample(frac=1, random_state=random_state_i, replace=False)
    Xval_df = pd.merge(Xval_D, Xval_noD, how='outer').sample(frac=1, random_state=random_state_i, replace=False)
    datasets_as_dfs = {}
    # datasets_as_dfs["train"] = pd.merge(Xtr_df, Xval_df, how='outer')
    datasets_as_dfs["train"] = Xtr_df
    datasets_as_dfs["valid"] = Xval_df
    datasets_as_dfs["test"] = Xts_df
    datasets_as_dfs["all"] = datasets_df.reset_index(drop=False)
    
    return datasets_as_dfs, feats

if __name__=="__main__":
    # train_loader, val_loader, test_loader, gpt_tokenizer, train_sampler = get_data_loaders(setname='brown', batch_size=10, is_token=False)
    train_loader1, val_loader1, test_loader1, gpt_tokenizer1, train_sampler1 = get_data_loaders(setname='penn', batch_size=4, 
                                                                                                image_input_size=244, is_token=False, random_state_i=0,
                                                                                                worker_seed=23)
    # data = next(iter(train_loader))
    for i, batch in enumerate(train_loader1):
        print(i)
        # print(data['boxes'])
    bs = len(batch['reference_sentences'])
    num_sentence = 8
    i = 9
    i_batch = int(np.floor(i / num_sentence))
    i_curr = int(i%num_sentence)
    name_tokenized = batch['idxs'][i][1:torch.argwhere(batch['idxs'][i]==25)[0,0]]
    name_str = batch['reference_sentences'][i_batch][i_curr].split(':')[0]
    
    
    
    
