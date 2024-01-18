import torch
import numpy as np
from matplotlib import pyplot as plt
from dataset.constants import ANATOMICAL_REGIONS

def plot_box(box, ax, clr, linestyle, class_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(
            xy=(x0, y0),
            height=h,
            width=w,
            fill=False,
            color=clr,
            linewidth=1,
            linestyle=linestyle
        )
    )

    # add an annotation to the gt box, that the pred box does not exist (i.e. the corresponding class was not detected)
    if not class_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


def get_title(region_set, region_indices, region_colors, class_detected_img):
    # region_set always contains 6 region names (except for region_set_5)

    # get a list of 6 boolean values that specify if that region was detected
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    # add color_code to region name (e.g. "(r)" for red)
    # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
    region_set = [region + f" ({color})" if cls_detect else region + f" ({color}, nd)" for region, color, cls_detect in zip(region_set, region_colors, class_detected)]

    # add a line break to the title, as to not make it too long
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def plot_gt_and_pred_bboxes_to_tensorboard(writer, overall_steps_taken, images, detections, targets, class_detected, num_images_to_plot=2):
    # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
    # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
    # the 2nd to the 2nd class and so on
    pred_boxes_batch = detections["top_region_boxes"]

    # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
    # gt_boxes is of shape [batch_size x 29 x 4]
    # gt_boxes_batch = torch.stack([t["boxes"] for t in targets], dim=0)
    gt_boxes_batch = targets

    # plot 6 regions at a time, as to not overload the image with boxes (except for region_set_5, which has 5 regions)
    # the region_sets were chosen as to minimize overlap between the contained regions (i.e. better visibility)
    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]

    for num_img in range(num_images_to_plot):
        image = images[num_img].cpu().numpy().transpose(1, 2, 0)

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()

        for num_region_set, region_set in enumerate(regions_sets):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap='gray')
            plt.axis('off')

            region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]
            region_colors = ["b", "g", "r", "c", "m", "y"]

            if num_region_set == 4:
                region_colors.pop()

            for region_index, color in zip(region_indices, region_colors):
                box_gt = gt_boxes_img[region_index].tolist()
                box_pred = pred_boxes_img[region_index].tolist()
                box_class_detected = class_detected_img[region_index]

                plot_box(box_gt, ax, clr=color, linestyle="solid", class_detected=box_class_detected)

                # only plot predicted box if class was actually detected
                if box_class_detected:
                    plot_box(box_pred, ax, clr=color, linestyle="dashed")

            title = get_title(region_set, region_indices, region_colors, class_detected_img)
            ax.set_title(title)

            writer.add_figure(f"img_{num_img}_region_set_{num_region_set}", fig, overall_steps_taken)
            
def plot_bboxes_(gt_boxes_img, image=None, surfix=''):
    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]
    region_colors = ["b", "g", "r", "c", "m", "y"]
    for num_region_set, region_set in enumerate(regions_sets):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        if image == None:
            plt.imshow(np.ones([512,512]), cmap='gray')
        else:
            plt.imshow(image, cmap='gray')
        plt.axis('off')

        region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]
        region_colors = ["b", "g", "r", "c", "m", "y"]

        if num_region_set == 4:
            region_colors.pop()

        for region_index, color in zip(region_indices, region_colors):
            box_gt = gt_boxes_img[region_index].tolist()
            box_class_detected = True

            plot_box(box_gt, ax, clr=color, linestyle="solid", class_detected=box_class_detected)
            
        # plt.show()
        plt.savefig(surfix +str(num_region_set)+'.jpg')

