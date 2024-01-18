import torch
import torch.nn.functional as F
def compute_box_area(box):
    """
    Calculate the area of a box given the 4 corner values.

    Args:
        box (Tensor[batch_size x 29 x 4])

    Returns:
        area (Tensor[batch_size x 29])
    """
    x0 = box[..., 0]
    y0 = box[..., 1]
    x1 = box[..., 2]
    y1 = box[..., 3]

    return (x1 - x0) * (y1 - y0)


def compute_intersection_and_union_area_per_class(detections, targets, class_detected):
    # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
    # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
    # the 2nd to the 2nd class and so on
    pred_boxes = detections["top_region_boxes"]

    # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
    # gt_boxes is of shape [batch_size x 29 x 4]
    gt_boxes = torch.stack([t["boxes"] for t in targets], dim=0)

    # below tensors are of shape [batch_size x 29]
    x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
    y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
    x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
    y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])

    # intersection_boxes is of shape [batch_size x 29 x 4]
    intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

    # below tensors are of shape [batch_size x 29]
    intersection_area = compute_box_area(intersection_boxes)
    pred_area = compute_box_area(pred_boxes)
    gt_area = compute_box_area(gt_boxes)

    # if x0_max >= x1_min or y0_max >= y1_min, then there is no intersection
    valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

    # also there is no intersection if the class was not detected by object detector
    valid_intersection = torch.logical_and(valid_intersection, class_detected)

    # set all non-valid intersection areas to 0
    intersection_area = torch.where(valid_intersection, intersection_area, torch.tensor(0, dtype=intersection_area.dtype, device=intersection_area.device))

    union_area = (pred_area + gt_area) - intersection_area

    # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
    intersection_area = torch.sum(intersection_area, dim=0)
    union_area = torch.sum(union_area, dim=0)

    return intersection_area, union_area


def simsiam_loss_func(x, y, predictor, flag='image'):
    p_x = predictor(x)
    p_y = predictor(y)
    z_x = x.detach()
    z_y = y.detach()
    return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5
   
def text_local_loss_fn(embed_A, embed_B, logit_scale, norm=True):
    '''
    Similarly to CUT[1], we only utilized internal negative samples in a single report. 
    Although incorporating additional negative sentences from other patients could potentially provide more negative samples, we observed a decline in performance. This outcome is understandable, as different reports may contain highly similar sentences (especially for normal sample).
    [1] Park T, Efros A A, Zhang R, et al. Contrastive learning for unpaired image-to-image translation[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16. Springer International Publishing, 2020: 319-345.
    '''
    # logit_scale = self.local_logit_scale.exp()
    if norm:
        embed_A = F.normalize(embed_A, dim=-1, p=2)
        embed_B = F.normalize(embed_B, dim=-1, p=2)
    lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
    logits_per_image = logit_scale * embed_B @ embed_A.t()
    logits_per_text = logit_scale * embed_A @ embed_B.t()
    image_loss = F.cross_entropy(logits_per_image, lc_labels)
    text_loss = F.cross_entropy(logits_per_text, lc_labels)
    loss = (image_loss + text_loss) / 2   
    return loss

from scipy.optimize import linear_sum_assignment
def matching_label(outputs, targets, norm=False, metric='l1'):
    # use the hungarian algorithm to match the labels
    if norm:
        outputs = F.normalize(outputs, dim=-1)
    if metric == 'l1':
        cost = torch.cdist(outputs, targets, p=1).detach().cpu()
    elif metric == 'l2':
        cost = torch.cdist(outputs, targets, p=2).detach().cpu()
    indices = linear_sum_assignment(cost)  
    return indices




def spg_loss(logits, protos, sentence_bank, lambda_proto):
    '''
    logits: the output of the sentence decoder (reconstructed sentence prototype), [B, N, D]
    features: the sentence features [before SPB], [B, N, D]
    protos: the sentence prototype [after SPB], [B, N, D]
    proto_indexs: the sentence prototype index, [B, N]
    sentence_masks: the sentence mask, [B, N]
    '''
    # TODO: optimize the matching process [use padding prototype to parallel the matching process ?]
    spg_loss = F.l1_loss(logits, protos)
    total_logits = logits.reshape(-1, logits.shape[-1] )

    # calculate the kl_loss
    # kl_loss is aimed to maintain the query consistency between logits and features
    kl_loss = sentence_bank.cal_loss(total_logits)
    return {'spg_loss': spg_loss * lambda_proto, 'kl_loss': kl_loss}, total_logits


# def spg_loss(self, logits, features, protos, proto_index, sentence_bank):
#     '''
#     logits: the output of the sentence decoder (reconstructed sentence prototype), [B, N, D]
#     features: the sentence features [before SPB], [B, N, D]
#     protos: the sentence prototype [after SPB], [B, N, D]
#     proto_indexs: the sentence prototype index, [B, N]
#     sentence_masks: the sentence mask, [B, N]
#     '''
#     spg_loss = 0.
#     total_proto_index = []
#     total_logits = []
#     total_features = []
#     logits_stacks = []
#     total_label = []
#     # TODO: optimize the matching process [use padding prototype to parallel the matching process ?]
#     for logit, feature, proto, label in zip(logits, features, protos, proto_index): 
#         # use hungarian algorithm to match the logit and the prototype
#         inds = matching_label(logit, proto)
#         # rearrange the logit according to the matching label
#         rearrange_logit = logit[inds[1]]
#         # exclude the padding token [mask=0]
#         roi_label = label.int()
#         roi_logit = rearrange_logit
#         roi_feature = feature
#         roi_proto = proto
#         # calculate the spg loss
#         spg_loss += F.l1_loss(rearrange_logit, proto)
#         # append the roi logit and roi feature
#         total_label.append(roi_label)
#         total_logits.append(roi_logit)
#         total_features.append(roi_feature)
#         logits_stacks.append(roi_logit)
#     total_label = torch.cat(total_label, dim=0)
#     total_logits = torch.cat(total_logits, dim=0)
#     total_features = torch.cat(total_features, dim=0)
#     # calculate the spg loss 
#     spg_loss = spg_loss / len(logits)
#     # calculate the kl_loss
#     # kl_loss is aimed to maintain the query consistency between logits and features
#     kl_loss = sentence_bank.cal_loss(total_logits) 
#     return {'spg_loss': spg_loss * self.lambda_proto, 'kl_loss': kl_loss}, logits_stacks