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

#/Users/chinya07/Downloads/openmv.png
model = YOLO('/Users/chinya07/Downloads/IDD_Train_V0_160x160.pt')


# class_name_mapping = {0: 'BIKE_LANE_MARKER', 1: 'TRAFFIC_SIGN', 2: 'SCOOTER', 3: 'CARS', 4: 'PERSONS', 5: 'BIKE', 6: 'ANIMAL'}
class_name_mapping = {0: 'Car',
  1: 'Bus',
  2: 'Truck',
  3: 'Motor_Bike',
  4: 'Auto_rickshaw',
  5: 'Bike',
  6: 'People',
  7: 'Animals',
  8: 'Traffic_signs',
  9: 'Scooter',
  10: 'Traffic_light'}
total_class_counts = defaultdict(int)

def draw_gt_boxes(image, gt_boxes, color=(0, 255, 0), thickness=2):
    
    for i,boxes in enumerate(gt_boxes):
        
        _, _, _, x_center, y_center, w, h = gt_boxes[i]
        x1 = int((x_center - w / 2) * image.shape[1])
        y1 = int((y_center - h / 2) * image.shape[0])
        x2 = int((x_center + w / 2) * image.shape[1])
        y2 = int((y_center + h / 2) * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def create_patches(image_path, num_patches, mode, overlap):
    # Load the image
    image = Image.open(image_path)
    # image = org_image.resize((640, 640))
    width, height = image.size
    print(width, height)

    # Validation for grid mode
    if mode == 'grid' and (num_patches < 4 or int(math.sqrt(num_patches)) != math.sqrt(num_patches)):
        raise ValueError("For grid mode, the number of patches must be a perfect square and at least 4.")

    # Create directory for patches
    output_dir = op_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def apply_model(num_patches, mode, output_dir):
    # Calculate grid size
    grid_size = int(math.sqrt(num_patches)) if mode == 'grid' else num_patches

    # Load the first patch to get individual dimensions
    first_patch_path = f"{output_dir}/patch_vertical_1.png" if mode == 'vertical' else f"{output_dir}/patch_horizontal_1.png" if mode == 'horizontal' else f"{output_dir}/patch_grid_1.png"
    print(f"(((((((((((((({first_patch_path}))))))))))))))")
    first_patch = cv2.imread(first_patch_path)
    patch_height, patch_width, _ = first_patch.shape

    # Determine dimensions of the full image
    full_width = patch_width * grid_size if mode in ['vertical', 'grid'] else patch_width
    full_height = patch_height * grid_size if mode in ['horizontal', 'grid'] else patch_height

    # Create a new image to paste the patches into
    # full_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)

    results = []
    train_idx = 0
    # Loop through saved patches and paste them in the correct positions
    for i in range(grid_size):
        
        patch_path = f"{output_dir}/patch_vertical_{i+1}.png"
        patch = cv2.imread(patch_path)
        # print(f"patch is being processed: {patch.shape}")
        train_idx += 1
        boxes = detect_objects(patch, train_idx)
        print(f"boxes are: {boxes}")
        if len(boxes) != 0:
            # draw_gt_boxes(patch, boxes)
            for b in boxes:
                results.append(b)
        else:
            continue
        
        
        cv2.imwrite(f"{output_dir}/repatched_image_{i+1}.png", patch)

    return results, patch_width, patch_height



def convert_to_pixel_coords(dets, patch_width, patch_height, num_patches_per_side):
    pixel_coords = []
    for bbox in dets:
        patch_index, class_id, class_prob, x, y, w, h = bbox
        
        # Adjust patch_index to be 0-based for calculation
        patch_index -= 1
        
        # Calculate the offset based on the patch index
        offset_x = (patch_index % num_patches_per_side) * patch_width
        offset_y = (patch_index // num_patches_per_side) * patch_height
        
        # Calculate pixel coordinates within the patch and add the offset
        x_min = (x - w / 2) * patch_width + offset_x
        y_min = (y - h / 2) * patch_height + offset_y
        x_max = (x + w / 2) * patch_width + offset_x
        y_max = (y + h / 2) * patch_height + offset_y
        
        pixel_coords.append([patch_index + 1, class_id, class_prob, x_min, y_min, x_max, y_max])
    
    return pixel_coords


# Merge bounding boxes if they are split across adjacent patches
def merge_bboxes(pixel_bboxes, patch_width):
    merged_bboxes = []
    pixel_bboxes.sort(key=lambda x: (x[0], x[3]))  # Sort by patch_index and x_min
    merged = False
    for i in range(len(pixel_bboxes)):
        if merged:
            merged = False
            continue

        for j in range(i + 1, len(pixel_bboxes)):
            if (pixel_bboxes[i][1] == pixel_bboxes[j][1]  # Same class
                    and pixel_bboxes[j][0] == pixel_bboxes[i][0] + 1  # Adjacent patches
                    # and pixel_bboxes[i][4] >= patch_width * (pixel_bboxes[i][0] + 1) - 10  # Right edge of left patch
                    # and pixel_bboxes[j][3] <= patch_width * (pixel_bboxes[j][0]) + 10):  # Left edge of right patch
                    and abs(pixel_bboxes[i][5] - patch_width * (pixel_bboxes[i][0]))<5  # Right edge of left patch
                    and abs(pixel_bboxes[j][3] - patch_width * (pixel_bboxes[i][0]))<5):  # Left edge of right patch                            
                # Merge the boxes

                new_x_min = min(pixel_bboxes[i][3], pixel_bboxes[j][3])
                new_y_min = min(pixel_bboxes[i][4], pixel_bboxes[j][4])
                new_x_max = max(pixel_bboxes[i][5], pixel_bboxes[j][5])
                new_y_max = max(pixel_bboxes[i][6], pixel_bboxes[j][6])
                merged_bboxes.append([pixel_bboxes[i][0], pixel_bboxes[i][1], pixel_bboxes[i][2], new_x_min, new_y_min, new_x_max, new_y_max, "m"])
                # pixel_bboxes.pop(i)
                # pixel_bboxes.pop(i+1)
                merged = True
                break
        if not merged:
            pixel_bboxes[i].append("nm")
            merged_bboxes.append(pixel_bboxes[i])

    return merged_bboxes

# Convert pixel coordinates back to normalized coordinates for the original image
def convert_to_normalized_coords(dets, img_width, img_height):
    normalized_coords = []
    for bbox in dets:
        patch_index, class_id, class_prob, x_min, y_min, x_max, y_max, _ = bbox
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height
        normalized_coords.append([class_id, class_prob, x_center, y_center, box_width, box_height, _])
    return normalized_coords

# Draw bounding boxes on the original image
def draw_bboxes(image, bboxes):
    for bbox in bboxes:
        # class_id, class_prob, x_min, y_min, x_max, y_max = bbox
        class_id, class_prob, x_center, y_center, w, h, m = bbox
        print(f"class_id: {class_id}, class_prob: {class_prob}, x_center: {x_center}, y_center: {y_center}, w: {w}, h: {h}")

        
        x1 = int((x_center - w / 2) * image.shape[1])
        y1 = int((y_center - h / 2) * image.shape[0])
        x2 = int((x_center + w / 2) * image.shape[1])
        y2 = int((y_center + h / 2) * image.shape[0])
        if m == "m":
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_prob:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


if __name__ == "__main__":

    path = "/Users/chinya07/Desktop/PROJECTS/LUNA/IDD_Annotation/video1/RESIZED_images/"
    op_path = "/Users/chinya07/Desktop/patches"
    # for img in sorted(os.listdir(path)):
    for image in sorted(os.listdir(path)):        
        image_path = path + image
        num_patches = 4
        # mode = input("Enter patching mode (vertical/horizontal/grid): ")
        mode = "vertical"
        # overlap = input("Do you want overlap? (yes/no): ")
        overlap = "no"
        output_dir = op_path
        create_patches(image_path, num_patches, mode, overlap)
        dets, patch_width, patch_height = apply_model(num_patches, mode, output_dir)
        print(f"dets look like this: {dets}")


        # Define the dimensions of the original image and patches
        original_img_width = 640  # Example width
        original_img_height = 160  # Example height
        num_patches = 4  # Number of patches along the width
        patch_width = original_img_width // num_patches
        patch_height = original_img_height

    

        # Convert bboxes to pixel coordinates
        pixel_bboxes = convert_to_pixel_coords(dets, patch_width, patch_height,num_patches)
        print(f"bboxes in pixel coords are : -----____-----_____------: {pixel_bboxes}")

        # draw_bboxes_on_image(image_path,pixel_bboxes)
        # Merge bboxes that are split across adjacent patches
        merged_bboxes = merge_bboxes(pixel_bboxes, patch_width)
        print(f"merged bboxes in pixel coords are : -----____-----_____------: {merged_bboxes}")
        # Convert merged bboxes back to normalized coordinates
        normalized_bboxes = convert_to_normalized_coords(merged_bboxes, original_img_width, original_img_height)
        print(f"final normalised bbox coords are : -----____-----_____------: {normalized_bboxes}")
        # Load original image
        original_image = cv2.imread(image_path)

        # Draw bounding boxes on the original image
        image_with_bboxes = draw_bboxes(original_image, normalized_bboxes)

        # Save or display the image with bounding boxes
        cv2.imwrite('/Users/chinya07/Desktop/IMAGE-PATCHING-RESULTS/'+ image, image_with_bboxes)

        print("Normalized Bounding Boxes:")
        for bbox in normalized_bboxes:
            print(bbox)

