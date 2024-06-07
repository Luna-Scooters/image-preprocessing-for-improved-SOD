import math
import os
from PIL import Image, ImageDraw
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from collections import Counter
import torch
import csv 
import yaml
import argparse
import tqdm
import time 


def draw_gt_boxes(image, gt_boxes, color=(0, 255, 0), thickness=2):
    
    for i,boxes in enumerate(gt_boxes):
        
        _, _, _, x_center, y_center, w, h = gt_boxes[i]
        x1 = int((x_center - w / 2) * image.shape[1])
        y1 = int((y_center - h / 2) * image.shape[0])
        x2 = int((x_center + w / 2) * image.shape[1])
        y2 = int((y_center + h / 2) * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def create_patches(image_path, num_patches, mode, overlap, resolution, output_dir):
    # Load the image
    org_image = Image.open(image_path)
    image = org_image.resize((resolution[0], resolution[1]))
    width, height = image.size
    print(width, height)

    # Validation for grid mode
    if mode == 'grid' and (num_patches < 4 or int(math.sqrt(num_patches)) != math.sqrt(num_patches)):
        raise ValueError("For grid mode, the number of patches must be a perfect square and at least 4.")

    # Initialize variables
    overlap_size = 0
    grid_size = int(math.sqrt(num_patches)) if mode == 'grid' else num_patches
    patch_width = width // grid_size if mode in ['vertical', 'grid'] else width
    patch_height = height // grid_size if mode in ['horizontal', 'grid'] else height

    # Set overlap if applicable
    if overlap.lower() == 'yes':
        overlap_size = int(patch_width * 0.2) if mode in ['vertical', 'grid'] else int(patch_height * 0.2)

    # Create and save patches
    patches = []
    for i in range(grid_size):
        for j in range(grid_size if mode == 'grid' else 1):
            if mode == 'vertical':
                left = max(0, i * patch_width - i * overlap_size)
                right = min(width, left + patch_width + overlap_size)
                patch = image.crop((left, 0, right, height))
                patch.save(f"{output_dir}/patch_vertical_{i+1}.png")
                patches.append((left, 0, right, height))
            elif mode == 'horizontal':
                top = max(0, i * patch_height - i * overlap_size)
                bottom = min(height, top + patch_height + overlap_size)
                patch = image.crop((0, top, width, bottom))
                patch.save(f"{output_dir}/patch_horizontal_{i+1}.png")
                patches.append((0, top, width, bottom))
            elif mode == 'grid':
                left = j * patch_width - (j * overlap_size if overlap.lower() == 'yes' else 0)
                top = i * patch_height - (i * overlap_size if overlap.lower() == 'yes' else 0)
                right = min(width, left + patch_width + overlap_size)
                bottom = min(height, top + patch_height + overlap_size)
                patch = image.crop((left, top, right, bottom))
                patch.save(f"{output_dir}/patch_grid_{i*grid_size+j+1}.png")
                patches.append((left, top, right, bottom))


def detect_objects(patch, idx):
    
    results = model(patch)
    detections = results[0].boxes.data
    norm_boxes = results[0].boxes.xywhn.tolist()
    all_boxes = []
    # Process each detection
    for j, detection in enumerate(detections):

        train_idx = idx

        # Extract class ID and probability
        class_id = detection[-1].int().item()  # Convert to Python int
        class_prob = detection[-2].item()

        # Normalized coordinates in the ROI
        x_center, y_center, bbox_width, bbox_height = norm_boxes[j]

        box_info = [
        train_idx, 
        class_id, 
        class_prob, 
        x_center,  # Already normalized
        y_center,  # Already normalized
        bbox_width,     # Already normalized
        bbox_height     # Already normalized
    ]          

        # results.append(box_info)
        # print(box_info)
        all_boxes.append(box_info)
    return all_boxes

def apply_model(num_patches, mode, patch_width, patch_height, output_dir):

    results = []
    train_idx = 0
    # Loop through saved patches and paste them in the correct positions
    for i in range(num_patches):
        
        patch_path = f"{output_dir}/patch_vertical_{i+1}.png" if mode == 'vertical' else f"{output_dir}/patch_horizontal_{i+1}.png" if mode == 'horizontal' else f"{output_dir}/patch_grid_{i+1}.png"
        patch = cv2.imread(patch_path)
        # print(f"patch is being processed: {patch.shape}")
        train_idx += 1
        boxes = detect_objects(patch, train_idx)
        # print(f"boxes are: {boxes}")
        if len(boxes) != 0:
            draw_gt_boxes(patch, boxes)
            for b in boxes:
                results.append(b)
        else:
            continue
        
        
        cv2.imwrite(f"{output_dir}/patch_with_boxes_{i+1}.png", patch)

    return results, patch_width, patch_height



# def convert_to_pixel_coords(dets, patch_width, patch_height, num_patches_per_side):
#     pixel_coords = []
#     for bbox in dets:
#         patch_index, class_id, class_prob, x, y, w, h = bbox
        
#         # Adjust patch_index to be 0-based for calculation
#         patch_index -= 1
        
#         # Calculate the offset based on the patch index
#         offset_x = (patch_index % num_patches_per_side) * patch_width
#         offset_y = (patch_index // num_patches_per_side) * patch_height
        
#         # Calculate pixel coordinates within the patch and add the offset
#         x_min = (x - w / 2) * patch_width + offset_x
#         y_min = (y - h / 2) * patch_height + offset_y
#         x_max = (x + w / 2) * patch_width + offset_x
#         y_max = (y + h / 2) * patch_height + offset_y
        
#         pixel_coords.append([patch_index + 1, class_id, class_prob, x_min, y_min, x_max, y_max])
    
#     return pixel_coords
def convert_to_pixel_coords(dets, patch_width, patch_height, num_patches, mode):
    pixel_coords = []
    for bbox in dets:
        patch_index, class_id, class_prob, x, y, w, h = bbox
        
        # Adjust patch_index to be 0-based for calculation
        patch_index -= 1
        
        if mode == "vertical":
            # Calculate the offset based on the patch index 
        
            offset_x = (patch_index % num_patches) * patch_width
            offset_y = (patch_index // num_patches) * patch_height
            
            # Calculate pixel coordinates within the patch and add the offset
            x_min = (x - w / 2) * patch_width + offset_x
            y_min = (y - h / 2) * patch_height + offset_y
            x_max = (x + w / 2) * patch_width + offset_x
            y_max = (y + h / 2) * patch_height + offset_y
            
            pixel_coords.append([patch_index + 1, class_id, class_prob, x_min, y_min, x_max, y_max])

        if mode == "grid":

            num_patches_per_row = int(math.sqrt(num_patches))
            
            offset_x = (patch_index % num_patches_per_row) * patch_width
            offset_y = (patch_index // num_patches_per_row) * patch_height
            
            # Calculate pixel coordinates within the patch and add the offset
            x_min = (x - w / 2) * patch_width + offset_x
            y_min = (y - h / 2) * patch_height + offset_y
            x_max = (x + w / 2) * patch_width + offset_x
            y_max = (y + h / 2) * patch_height + offset_y
            
            pixel_coords.append([patch_index + 1, class_id, class_prob, x_min, y_min, x_max, y_max])
    
    return pixel_coords

# Merge bounding boxes if they are split across adjacent patches
def merge_bboxes(pixel_bboxes, patch_width, train_idx, num_patches, mode):
    print("train_idx: ", train_idx)
    num_patches_per_row = int(math.sqrt(num_patches))
    merged_bboxes = []
    pixel_bboxes.sort(key=lambda x: (x[0], x[3]))  # Sort by patch_index and x_min
    merged = False
    for i in range(len(pixel_bboxes)):
        if merged:
            merged = False
            continue

        for j in range(i + 1, len(pixel_bboxes)):
            if mode == "vertical":
                if (pixel_bboxes[i][1] == pixel_bboxes[j][1]  # Same class
                    and pixel_bboxes[j][0] == pixel_bboxes[i][0] + 1  # Adjacent patches
                    and abs(pixel_bboxes[i][6] - pixel_bboxes[j][4]) < 5):  # Check 5-pixel margin between patches
                    # Merge the boxes
                    new_x_min = min(pixel_bboxes[i][3], pixel_bboxes[j][3])
                    new_y_min = min(pixel_bboxes[i][4], pixel_bboxes[j][4])
                    new_x_max = max(pixel_bboxes[i][5], pixel_bboxes[j][5])
                    new_y_max = max(pixel_bboxes[i][6], pixel_bboxes[j][6])
                    merged_bboxes.append([train_idx, pixel_bboxes[i][1], (pixel_bboxes[i][2] + pixel_bboxes[j][2]) / 2, new_x_min, new_y_min, new_x_max, new_y_max, "m"])
                    merged = True
                    break
            elif mode == "grid":
                if (pixel_bboxes[i][1] == pixel_bboxes[j][1]  # Same class
                    and pixel_bboxes[j][0] == pixel_bboxes[i][0] + 1  # Adjacent row
                    or pixel_bboxes[j][0] == pixel_bboxes[i][0] + num_patches_per_row # Adjacent column
                    and abs(pixel_bboxes[i][5] - pixel_bboxes[j][3]) < 5
                    or abs(pixel_bboxes[i][6] - pixel_bboxes[j][4]) < 5):  # Check 5-pixel margin between patches
                    # Merge the boxes
                    new_x_min = min(pixel_bboxes[i][3], pixel_bboxes[j][3])
                    new_y_min = min(pixel_bboxes[i][4], pixel_bboxes[j][4])
                    new_x_max = max(pixel_bboxes[i][5], pixel_bboxes[j][5])
                    new_y_max = max(pixel_bboxes[i][6], pixel_bboxes[j][6])
                    merged_bboxes.append([train_idx, pixel_bboxes[i][1], (pixel_bboxes[i][2] + pixel_bboxes[j][2]) / 2, new_x_min, new_y_min, new_x_max, new_y_max, "m"])
                    merged = True
                    break
        if not merged:
            pixel_bboxes[i][0]=train_idx
            pixel_bboxes[i].append("nm")
            merged_bboxes.append(pixel_bboxes[i])
        
    return merged_bboxes

# Convert pixel coordinates back to normalized coordinates for the original image
def convert_to_normalized_coords(dets, img_width, img_height):
    normalized_coords = []
    for bbox in dets:
        img_index, class_id, class_prob, x_min, y_min, x_max, y_max, _ = bbox
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height
        normalized_coords.append([img_index, class_id, class_prob, x_center, y_center, box_width, box_height, _])
    return normalized_coords

# Draw bounding boxes on the original image
def draw_bboxes(image, bboxes):
    for bbox in bboxes:
        # class_id, class_prob, x_min, y_min, x_max, y_max = bbox
        img_index, class_id, class_prob, x_center, y_center, w, h, m = bbox
        # print(f"class_id: {class_id}, class_prob: {class_prob}, x_center: {x_center}, y_center: {y_center}, w: {w}, h: {h}")

        
        x1 = int((x_center - w / 2) * image.shape[1])
        y1 = int((y_center - h / 2) * image.shape[0])
        x2 = int((x_center + w / 2) * image.shape[1])
        y2 = int((y_center + h / 2) * image.shape[0])
        if m == "m":
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # cv2.putText(image, f'{class_prob:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image



def intersection_over_union(boxes_preds, boxes_labels, box_format):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


#MAP CALCULATION CODE: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold, box_format, mAP_num_classes
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        mAP_num_classes (int): number of classes to calculate mAP

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(mAP_num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

# Function to calculate area of a bounding box
def calculate_area(bbox):
    _,_,_,_,_, w, h = bbox
    return w * h




if __name__ == "__main__":

    with open('/Users/chinya07/Desktop/PROJECTS/PHD/image-preprocessing-for-improved-SOD/image-preprocessing-for-improved-SOD/image-patching/img_patching_config.yaml', 'r') as f:
        config = yaml.safe_load(f)


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Image patching arg parser')

    # Add arguments
    parser.add_argument('--model', type=str, help='model path.')
    parser.add_argument('--image_dir', type=str, help='Path to the input images dir.')
    parser.add_argument('--output_patches_dir', type=str, help='Path to the output patches dir.')
    parser.add_argument('--GT_label_dir', type=str, help='Path to the GT labels dir.')
    parser.add_argument('--small_bbox_area_threshold', type=float, help='Small bounding box area thresholds as per COCO dataset.')
    parser.add_argument('--medium_bbox_area_threshold', type=float, help='Medium bounding box area thresholds as per COCO dataset.')
    parser.add_argument('--number_of_patches', type=int, help='number of patches.')
    parser.add_argument('--patch_mode', type=str, choices=['horizontal', 'vertical', 'grid'], help='Patching mode. Choose from horizontal, vertical, or grid.')
    parser.add_argument('--overalpping_patches', type=str, help='want overlapping patches or not.')
    parser.add_argument('--image_resolution', type=int, help='image resolution in [w,h] format.')
    parser.add_argument('--output_image_dir', type=str, help='Directory to save the output.')
    parser.add_argument('--mAP_num_classes', type=int, help='how many number of classes you want to calculate mAP with.')
    parser.add_argument('--iou_threshold', type=float, help='iou_threshold for mAP.')
    parser.add_argument('--box_format', type=str, help='box_format for mAP.')
    # Add more arguments as needed

    args = parser.parse_args()

    # Set defaults from config file
    args.model = args.model or config['model']
    args.image_dir = args.image_dir or config['image_dir']
    args.output_patches_dir = args.output_patches_dir or config['output_patches_dir']
    args.GT_label_dir = args.GT_label_dir or config['GT_label_dir']
    args.small_bbox_area_threshold = args.small_bbox_area_threshold or config['small_bbox_area_threshold']
    args.medium_bbox_area_threshold = args.medium_bbox_area_threshold or config['medium_bbox_area_threshold']
    args.number_of_patches = args.number_of_patches or config['number_of_patches']
    args.patch_mode = args.patch_mode or config['patch_mode']
    args.overalpping_patches = args.overalpping_patches or config['overalpping_patches']
    args.image_resolution = args.image_resolution or config['image_resolution']
    args.output_image_dir = args.output_image_dir or config['output_image_dir']
    args.mAP_num_classes = args.mAP_num_classes or config['mAP_num_classes']
    args.iou_threshold = args.iou_threshold or config['iou_threshold']
    args.box_format = args.box_format or config['box_format']


    model = YOLO(args.model)
    class_name_mapping = config['class_name_mapping']
    total_class_counts = defaultdict(int)
    all_boxes = []
    train_idx = 0

    # Lists to hold categorized bounding boxes
    GT_small_boxes = []
    GT_medium_boxes = []
    GT_large_boxes = []

    # Create directory for patches
    if not os.path.exists(args.output_patches_dir):
        os.makedirs(args.output_patches_dir)

    num_patches = args.number_of_patches
    mode = args.patch_mode
    overlap = args.overalpping_patches
    resolution = args.image_resolution
    mAP_num_classes = args.mAP_num_classes
    iou_threshold = args.iou_threshold
    box_format = args.box_format

    # Initialize tqdm for overall progress
    total_images = len(os.listdir(args.image_dir))
    pbar = tqdm.tqdm(total=total_images, desc="Processing images")

    for image in sorted(os.listdir(args.image_dir)):        
        image_path = args.image_dir + image

        # Define the dimensions of the original image and patches
        original_img_width = resolution[0]  # Example width
        original_img_height = resolution[1]  # Example height
        if mode == "vertical":
            patch_width = original_img_width // num_patches
            patch_height = original_img_height
        if mode == "grid":
            patch_width = original_img_width // int(math.sqrt(num_patches))
            patch_height = original_img_height // int(math.sqrt(num_patches))     

        create_patches(image_path, num_patches, mode, overlap, resolution, args.output_patches_dir)
        start_time = time.time()
        dets, patch_width, patch_height = apply_model(num_patches, mode, patch_width, patch_height, args.output_patches_dir)
        end_time = time.time()
        # print(f"dets look like this: {dets}")

        # Convert bboxes to pixel coordinates
        pixel_bboxes = convert_to_pixel_coords(dets, patch_width, patch_height, num_patches, mode)
        # print(f"bboxes in pixel coords are : -----____-----_____------: {pixel_bboxes}")

        # draw_bboxes_on_image(image_path,pixel_bboxes)
        # Merge bboxes that are split across adjacent patches
        merged_bboxes = merge_bboxes(pixel_bboxes, patch_width, train_idx, num_patches, mode)
        # print(f"merged bboxes in pixel coords are : -----____-----_____------: {merged_bboxes}")
        # Convert merged bboxes back to normalized coordinates
        normalized_bboxes = convert_to_normalized_coords(merged_bboxes, original_img_width, original_img_height)
        # print(f"final normalised bbox coords are : -----____-----_____------: {normalized_bboxes}")

        # Load original image
        train_idx+=1
        original_image = cv2.imread(image_path)

        # Draw bounding boxes on the original image
        image_with_bboxes = draw_bboxes(original_image, normalized_bboxes)

        # Save or display the image with bounding boxes
        cv2.imwrite(args.output_image_dir + image, image_with_bboxes)


        if len(normalized_bboxes) != 0:
            for box in normalized_bboxes:
                del box[-1]
            all_boxes.append(normalized_bboxes[0])
        else: 
            continue

        elapsed_time = (end_time - start_time)*1000

        # Print the elapsed time
        print("Time taken for one image:", elapsed_time, "ms")
        pbar.update(1)

    pbar.close()

    # Lists to hold segregated bounding boxes
    small_boxes = []
    medium_boxes = []
    large_boxes = []

    # Iterate over each frame's list of bounding boxes

    for bbox in all_boxes:
        area = calculate_area(bbox)
        if area < args.small_bbox_area_threshold:
            small_boxes.append(bbox)
        elif area < args.medium_bbox_area_threshold:
            medium_boxes.append(bbox)
        else:
            large_boxes.append(bbox)





    for frame_id, filename in enumerate(sorted(os.listdir(args.GT_label_dir))):
        if filename.endswith('.txt'):
            file_path = os.path.join(args.GT_label_dir, filename)

            # Check if the corresponding image's ROI coordinates are available
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        class_prob = 1.0
                        bbox = [frame_id, class_id, class_prob, x, y, w, h]
                        
                        area = calculate_area(bbox)
                        if area < args.small_bbox_area_threshold:
                            GT_small_boxes.append(bbox)
                        elif area < args.medium_bbox_area_threshold:
                            GT_medium_boxes.append(bbox)
                        else:
                            GT_large_boxes.append(bbox)



    GT_medium_boxes = [box for box in GT_medium_boxes if box[1] == 0 or box[1] == 3 or box[1] == 6]
    GT_small_boxes = [box for box in GT_small_boxes if box[1] == 0 or box[1] == 3 or box[1] == 6]
    GT_large_boxes = [box for box in GT_large_boxes if box[1] == 0 or box[1] == 3 or box[1] == 6]

    medium_boxes = [box for box in medium_boxes if box[1] == 0 or box[1] == 3 or box[1] == 6]
    small_boxes = [box for box in small_boxes if box[1] == 0 or box[1] == 3 or box[1] == 6]
    large_boxes = [box for box in large_boxes if box[1] == 0 or box[1] == 3 or box[1] == 6]


    # print(f"GT_small_boxes look like this: {GT_small_boxes}")
    # print(f"Predicted small_boxes look like this: {small_boxes}")


    mAP = mean_average_precision(
    small_boxes,
    GT_small_boxes,
    iou_threshold,
    box_format,
    mAP_num_classes,
    )
    print(f"mAP for Small boxes ========:> {mAP.item()}")


    mAP = mean_average_precision(
    medium_boxes,
    GT_medium_boxes,
    iou_threshold,
    box_format,
    mAP_num_classes,
    )
    print(f"mAP for Medium boxes ========:> {mAP.item()}")

    mAP = mean_average_precision(
    large_boxes,
    GT_large_boxes,
    iou_threshold,
    box_format,
    mAP_num_classes,
    )
    print(f"mAP for Large boxes ========:> {mAP.item()}")        


    mAP = mean_average_precision(
    small_boxes+medium_boxes,
    GT_small_boxes+GT_medium_boxes,
    iou_threshold,
    box_format,
    mAP_num_classes,
    )
    print(f"mAP for small + medium boxes ========:> {mAP.item()}")        


