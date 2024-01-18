import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BboxCompleter(nn.Module):
    """
    Classifier to determine if a region is abnormal or not.
    This is done as to encode this information more explicitly in the region feature vectors that are passed into the decoder.
    This may help with generating better sentences for abnormal regions (which are the minority class).

    This classifier is only applied during training and evalution, but not during inference.
    """
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=116, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=116),
            nn.ReLU()
        )

        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.MSELoss()

    def forward(self, bboxes_masked, bboxes_gt, label_gt):
        bboxes_masked_f = bboxes_masked.flatten(start_dim=1)
        # bboxes_gt_f = bboxes_gt.flatten(start_dim=1)
        # label_gt_f = label_gt.repeat([1,1,4]).flatten(start_dim=1)
        # logits of shape [batch_size, 29 x 4]
        logits = self.classifier(bboxes_masked_f)
        logits_a = logits.reshape(-1, 29, 4)
        logits_w_gt = logits_a[label_gt.bool()]
        bboxes_gt_nonz = bboxes_gt[label_gt.bool()]
        loss = self.loss_fn(logits_w_gt, bboxes_gt_nonz.type(torch.float32))
        if self.training:
            return loss
        else:
            return loss, logits

    def predict(self, bboxes_with_miss):
        bboxes_with_miss_f = bboxes_with_miss.flatten(start_dim=1)
        logits = self.classifier(bboxes_with_miss_f)
        logits_a = logits.reshape(-1, 29, 4)
        return logits_a

def prepare_bbox(batch_bbox, batch_label, if_mask=True ):
    batch_size = len(batch_bbox)
    device = batch_bbox.device
    bboxes_gt = torch.zeros([batch_size, 29, 4]).to(device, non_blocking=True)
    label_gt = torch.zeros([batch_size, 29]).to(device, non_blocking=True)
    bboxes_masked = torch.zeros([batch_size, 29, 4]).to(device, non_blocking=True)
    bboxes_max_length_list = []
    for i, (bbox_, label_) in enumerate(zip(batch_bbox, batch_label)):
        if len(bbox_) != 0:
            if (label_ > 1).any():
                label_idx = label_ - 1
            else:
                label_idx = torch.where(label_)[0]
            bbox_ = bbox_[label_idx]
            bbox_bbox = torch.concat([bbox_.max(0)[0][[2,3]][None] , bbox_.min(0)[0][[0,1]][None]], dim=0)
            bbox_size = bbox_bbox[0] - bbox_bbox[1]
            max_length = max(torch.abs(bbox_size))
            # bbox_center = bbox_bbox.mean(0)
            bbox_norm = bbox_ / max_length
            bboxes_max_length_list.append(max_length)
            bbox_norm = torch.maximum(bbox_norm, torch.zeros_like(bbox_norm))
            bbox_norm = torch.minimum(bbox_norm, torch.ones_like(bbox_norm))

            label_gt[i][label_idx] = 1
            bboxes_gt[i][label_idx] = bbox_norm
        # print(bbox_norm.max(), bbox_norm.min())
        bboxes_masked[i] = torch.clone(bboxes_gt[i])
        if if_mask:
            rand_mask_indx = np.random.choice(np.arange(29), int(np.random.random(1)*13)+1, replace=False)
            bboxes_masked[i][rand_mask_indx] = 0
    bboxes_max_length = torch.tensor(bboxes_max_length_list)[:,None,None].repeat(1,29,4).to(device, non_blocking=True)
    return bboxes_masked, bboxes_gt, label_gt, bboxes_max_length