import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os 
from collections import Counter
import torch
import csv 
import yaml
import argparse
import tqdm
import time 



def adjust_and_normalize_bbox(bbox, x1, y1, x2, y2, width, height):
    frame_id, class_id, class_probability, x_center, y_center, bbox_width, bbox_height = bbox
    
    # Convert normalized coordinates to absolute pixel coordinates
    bbox_x1 = max((x_center - bbox_width / 2) * width, x1)
    bbox_y1 = max((y_center - bbox_height / 2) * height, y1)
    bbox_x2 = min((x_center + bbox_width / 2) * width, x2)
    bbox_y2 = min((y_center + bbox_height / 2) * height, y2)

    # Check if the GT bounding box is completely outside the ROI
    if bbox_x1 >= x2 or bbox_x2 <= x1 or bbox_y1 >= y2 or bbox_y2 <= y1:
        # return [frame_id, class_id, class_probability, x_center, y_center, bbox_width, bbox_height]
        return []
        

    # Normalize coordinates back to the original image size
    norm_x_center = ((bbox_x1 + bbox_x2) / 2) / width
    norm_y_center = ((bbox_y1 + bbox_y2) / 2) / height
    norm_bbox_width = (bbox_x2 - bbox_x1) / width
    norm_bbox_height = (bbox_y2 - bbox_y1) / height

    # print("adjusted and normalised GT boxes are:--->",norm_x_center, norm_y_center, norm_bbox_width, norm_bbox_height)

    return [frame_id, class_id, class_probability, norm_x_center, norm_y_center, norm_bbox_width, norm_bbox_height]


def draw_gt_boxes(image, gt_boxes, color=(0, 255, 0), thickness=2):
    
    _, _, _, x_center, y_center, w, h = gt_boxes
    x1 = int((x_center - w / 2) * image.shape[1])
    y1 = int((y_center - h / 2) * image.shape[0])
    x2 = int((x_center + w / 2) * image.shape[1])
    y2 = int((y_center + h / 2) * image.shape[0])
    # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)



def handling_GT(labels_directory, image_folder_path, output_folder_path, height, width):

    GT_small_boxes = []
    GT_medium_boxes = []
    GT_large_boxes = []
    total_small_medium_gt_box_temp_count = 0
    # Process each label file
    for frame_id, filename in enumerate(sorted(os.listdir(labels_directory))):
        if filename.endswith('.txt'):
            image_name = filename.replace('.txt', '.png')  # Match the label file to the corresponding image file
            file_path = os.path.join(labels_directory, filename)

            # Check if the corresponding image's ROI coordinates are available
            if image_name in img_roi_dict and image_name in updates_frame_and_name:
                image_full_name = image_folder_path + image_name
                img = cv2.imread(image_full_name)
                x1, y1, x2, y2 = img_roi_dict[image_name]
                # cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 2)
                small_medium_gt_box_temp_count = 0  #FOR % CALCULATION 
                with open(file_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            class_prob = 1.0
                            bbox = [frame_id, class_id, class_prob, x, y, w, h]
                            # adjusted_bbox = bbox
                            adjusted_bbox = adjust_and_normalize_bbox(bbox, x1, y1, x2, y2, width, height)
                            # print(f"print adjusted box^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ {adjusted_bbox}")
                            if adjusted_bbox:
                                area = adjusted_bbox[5] * adjusted_bbox[6]  # w * h
                                if area < args.small_bbox_area_threshold:
                                    draw_gt_boxes(img, adjusted_bbox)
                                    small_medium_gt_box_temp_count += 1
                                    GT_small_boxes.append(adjusted_bbox)
                                elif area < args.medium_bbox_area_threshold:
                                    small_medium_gt_box_temp_count += 1
                                    GT_medium_boxes.append(adjusted_bbox)
                                else:
                                    GT_large_boxes.append(adjusted_bbox)
                # print("small_medium_gt_box_temp_count------------------->",small_medium_gt_box_temp_count)                    
                total_small_medium_gt_box_temp_count += small_medium_gt_box_temp_count
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (0, 255, 0)
                line_type = 2
                frame, total_predcited_objects = updates_frame_and_name[image_name]
                total_percent = (total_predcited_objects/total_small_medium_gt_box_temp_count)*100
                total_text = f"Total % of Objects Detected as compare to GT: {total_percent: .2f} %"
                cv2.putText(frame, total_text, (15, 120), font, font_scale, font_color, line_type)  
                output_image_path = os.path.join(output_folder_path, image_name)
                cv2.imwrite(output_image_path, frame)      


    GT_medium_boxes = [box for box in GT_medium_boxes if box[1] == 0 or box[1] == 3 or box[1] == 2]
    GT_small_boxes = [box for box in GT_small_boxes if box[1] == 0 or box[1] == 3 or box[1] == 2]
    GT_large_boxes = [box for box in GT_large_boxes if box[1] == 0 or box[1] == 3 or box[1] == 2]

    return  GT_small_boxes, GT_medium_boxes, GT_large_boxes

# Function to calculate area of a bounding box
def calculate_area(bbox):
    _,_,_,_,_, w, h = bbox
    return w * h


def zooming_centre_ROI(image_folder_path, resolution, zoom_factors_dict, model, display_only_small_boxes, train_idx):

    total_detected_small_objects = 0

    # Iterate over the images in the folder
    for image_name in sorted(os.listdir(image_folder_path)):
        small_objects_detected = 0
        image_path = os.path.join(image_folder_path, image_name)
        
        # Read the image
        original_frame = cv2.imread(image_path)
        if original_frame is None:
            continue

        frame = cv2.resize(original_frame, (resolution[0], resolution[1]))
        height, width, _ = frame.shape

        if image_name in zoom_factors_dict:
            factors = zoom_factors_dict[image_name]
            left, right, top, bottom = factors['left'], factors['right'], factors['top'], factors['bottom']
            left, right, top, bottom = left/100.0, right/100.0, top/100.0, bottom/100.0
            print(f"left: {left}, right: {right}, top: {top}, bottom: {bottom}")
            print(f"Train Index: {train_idx}")
            x1 = int(left * width)
            y1 = int(top * height)
            x2 = width - int(right * width)
            y2 = height - int(bottom * height)
            roi = np.copy(frame[y1:y2, x1:x2])
            # cv2.imshow('FRAMW',frame)
            # cv2.imshow('ROI',roi)
            # key = cv2.waitKey(-1)
            # if key == ord('q'):
            #     exit()
            
            img_roi_dict[image_name] = [x1, y1, x2, y2]

            print("roi shape each time ------>", roi.shape)      
            # zoomed_frame = cv2.resize(roi, (width, height))
        else:
            roi = frame.copy()
            x1, y1, x2, y2 = 0, 0, width, height
        
        results = model(roi)
        detections = results[0].boxes.data


        # cv2.rectangle(frame, (x1,y1), (x1+(x2-x1), y1+(y2-y1)), (0,0,0), 1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)

        norm_boxes = results[0].boxes.xywhn.tolist()


        

        # Process each detection
        for i, detection in enumerate(detections):

            # Extract class ID and probability
            class_id = detection[-1].int().item()  # Convert to Python int
            class_prob = detection[-2].item()

            # Normalized coordinates in the ROI
            x_center, y_center, bbox_width, bbox_height = norm_boxes[i]

            # Convert to pixel coordinates in the ROI
            roi_x_center = x_center * (x2 - x1)
            roi_y_center = y_center * (y2 - y1)
            roi_width_pixel = bbox_width * (x2 - x1)
            roi_height_pixel = bbox_height * (y2 - y1)

            # Rescale to original image coordinates
            original_x_center = roi_x_center + x1
            original_y_center = roi_y_center + y1
            original_width_pixel = roi_width_pixel
            original_height_pixel = roi_height_pixel

            # Normalize rescaled coordinates to original image size
            norm_x_center = original_x_center / width
            norm_y_center = original_y_center / height
            norm_width = original_width_pixel / width
            norm_height = original_height_pixel / height
            
            # Convert tensor values to floats and append to all_boxes
            
            box_info = [
                train_idx, 
                class_id, 
                class_prob, 
                norm_x_center,  # Already normalized
                norm_y_center,  # Already normalized
                norm_width,     # Already normalized
                norm_height     # Already normalized
            ]

            all_boxes.append(box_info)      


            # Convert to top-left coordinates for drawing
            x = int(original_x_center - original_width_pixel / 2)
            y = int(original_y_center - original_height_pixel / 2)
            w = int(original_width_pixel)
            h = int(original_height_pixel)

            # Draw the bounding box on the original frame
            area=calculate_area(box_info)
            if area <= args.medium_bbox_area_threshold:
                small_objects_detected += 1
                if display_only_small_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if not display_only_small_boxes:        
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            total_detected_small_objects += small_objects_detected 
            # Add the count of small objects to the top of the image

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Small Objects Detected: {small_objects_detected}"
        position = (15, 40)
        font_scale = 0.7
        font_color = (0, 255, 0)
        line_type = 2

        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)


        # Add the total count of objects detected so far
        total_text = f"Total Objects Detected So Far: {total_detected_small_objects}"
        cv2.putText(frame, total_text, (15, 80), font, font_scale, font_color, line_type)   


        class_id_temp = []
        class_prob_temp = []
        # Extract class IDs and convert to integers
        class_ids = results[0].boxes.data[:, -1].int()
        class_probs = results[0].boxes.data[:, -2]
        class_id_temp = class_ids.tolist()
        class_prob_temp = class_probs.tolist()
        

        train_idx += 1       

        # Count occurrences of each class ID
        class_counts = defaultdict(int)
        for class_id in class_ids:
            class_counts[class_id.item()] += 1

        # Map class IDs to class names and update total counts
        for class_id, count in class_counts.items():
            class_name = class_name_mapping.get(class_id, "Unknown")
            total_class_counts[class_name] += count



        
        # output_image_path = os.path.join(output_folder_path, image_name)
        updates_frame_and_name[image_name] = (frame, total_detected_small_objects)
        # cv2.imwrite(output_image_path, frame)


    # Print the total counts for each class
    for class_name, count in total_class_counts.items():
        print(f"------------{class_name}: {count} objects detected")
    # print(all_boxes)

    # print("ALL BOXES:", all_boxes[:5])

    #ADJUSTING GT BOXES AS PER THE ZOOMING ROI FOR FAIR MAP CALCULATION

    #SEGREGATING MODEL PREDICTIONS ON ZOOMED IMAGES INTO SMALL, MEDIUM AND LARGE BOXES

    #CALCULATED THE SMALL AND MEDIUM AND LARGE BOX AREA THREHOLD BY CALCULATING PERCENTILE

    # Thresholds for box sizes
    return height, width


def separate_boxes_as_per_size(all_boxes):

    # Iterate over each frame's list of bounding boxes
    small_boxes= []
    medium_boxes= []
    large_boxes= []
    for bbox in all_boxes:
        area = calculate_area(bbox)
        if area < args.small_bbox_area_threshold:
            small_boxes.append(bbox)
        elif area < args.medium_bbox_area_threshold:
            medium_boxes.append(bbox)
        else:
            large_boxes.append(bbox)


    # print(f"small boxes: ------> {small_boxes[:2]}, medium boxes: ------> {medium_boxes[:2]}, large boxes: ------> {large_boxes[:2]}")

    medium_boxes = [box for box in medium_boxes if box[1] == 0 or box[1] == 3 or box[1] == 2]
    small_boxes = [box for box in small_boxes if box[1] == 0 or box[1] == 3 or box[1] == 2]
    large_boxes = [box for box in large_boxes if box[1] == 0 or box[1] == 3 or box[1] == 2]

    return small_boxes, medium_boxes, large_boxes



#IOU CODE: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/


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


def decorative_message(function_name):
    message = f"Function '{function_name}' has been executed!"
    decoration = "*" * (len(message) + 4)
    print(decoration)
    print(f"* {message} *")
    print(decoration)



if __name__ == "__main__":

    with open('/Users/chinya07/Desktop/PROJECTS/PHD/image-preprocessing-for-improved-SOD/image-preprocessing-for-improved-SOD/image-patching/ROI_zooming_config.yaml', 'r') as f:
        config = yaml.safe_load(f)


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Image patching arg parser')

    # Add arguments
    parser.add_argument('--dummy_speed_info_csv', type=str, help='dummy speed info csv path.')
    parser.add_argument('--no_speed_csv', type=str, help='no speed csv path.')
    parser.add_argument('--model', type=str, help='model path.')
    parser.add_argument('--image_dir', type=str, help='Path to the input images dir.')
    parser.add_argument('--output_folder_path', type=str, help='Path to the output folder.')
    parser.add_argument('--GT_label_dir', type=str, help='Path to the GT labels dir.')
    parser.add_argument('--small_bbox_area_threshold', type=float, help='Small bounding box area thresholds as per COCO dataset.')
    parser.add_argument('--medium_bbox_area_threshold', type=float, help='Medium bounding box area thresholds as per COCO dataset.')
    parser.add_argument('--image_resolution', type=int, help='image resolution in [w,h] format.')
    parser.add_argument('--mAP_num_classes', type=int, help='how many number of classes you want to calculate mAP with.')
    parser.add_argument('--iou_threshold', type=float, help='iou_threshold for mAP.')
    parser.add_argument('--box_format', type=str, help='box_format for mAP.')
    parser.add_argument('--display_only_small_boxes', type=str, help='display_only_small_boxes flag to display only small or all the boxes.')
    parser.add_argument('--zoom', type=bool, help='zoom or normal.')
    # Add more arguments as needed

    args = parser.parse_args()

    # Set defaults from config file
    args.dummy_speed_info_csv = args.dummy_speed_info_csv or config['dummy_speed_info_csv']
    args.no_speed_csv = args.no_speed_csv or config['no_speed_csv']
    args.model = args.model or config['model']
    args.image_dir = args.image_dir or config['image_dir']
    args.output_folder_path = args.output_folder_path or config['output_folder_path']
    args.GT_label_dir = args.GT_label_dir or config['GT_label_dir']
    args.small_bbox_area_threshold = args.small_bbox_area_threshold or config['small_bbox_area_threshold']
    args.medium_bbox_area_threshold = args.medium_bbox_area_threshold or config['medium_bbox_area_threshold']
    args.image_resolution = args.image_resolution or config['image_resolution']
    args.mAP_num_classes = args.mAP_num_classes or config['mAP_num_classes']
    args.iou_threshold = args.iou_threshold or config['iou_threshold']
    args.box_format = args.box_format or config['box_format']
    args.display_only_small_boxes = args.display_only_small_boxes or config['display_only_small_boxes']
    args.zoom = args.zoom or config['zoom']

    # =========================================================================================================================================================

    # Load the CSV file and create a dictionary mapping image names to zoom factors
    zoom_factors_dict = {}

    csv_file_path = args.dummy_speed_info_csv if args.zoom else args.no_speed_csv

    with open(csv_file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name = row['image_name']
            zoom_factors_dict[image_name] = {
                'left': float(row['left']),
                'right': float(row['right']),
                'top': float(row['top']),
                'bottom': float(row['bottom'])
            }
    # =========================================================================================================================================================


    # Load the YOLOv8 model
    model = YOLO(args.model)

    class_name_mapping = config['class_name_mapping']
    total_class_counts = defaultdict(int)

    # Paths to the folder with images and the output folder
    image_folder_path = args.image_dir 
    output_folder_path = args.output_folder_path
    labels_directory = args.GT_label_dir
    # output_folder_path_without_zoom = '/Users/chinya07/Desktop/PROJECTS/LUNA/IDD_Annotation/video1/output_images_without_zoom'

    resolution = args.image_resolution
    mAP_num_classes = args.mAP_num_classes
    iou_threshold = args.iou_threshold
    box_format = args.box_format    
    display_only_small_boxes = args.display_only_small_boxes
    frame_number = 0
    all_boxes = []
    train_idx = 0
    updates_frame_and_name = {}
    img_roi_dict = {}


    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    height, width = zooming_centre_ROI(image_folder_path, resolution, zoom_factors_dict, model, display_only_small_boxes, train_idx)
    decorative_message("zooming_centre_ROI")

    small_boxes, medium_boxes, large_boxes = separate_boxes_as_per_size(all_boxes)
    decorative_message("separate_boxes_as_per_size")

    GT_small_boxes, GT_medium_boxes, GT_large_boxes = handling_GT(labels_directory, image_folder_path, output_folder_path, height, width)
    decorative_message("handling_GT")


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