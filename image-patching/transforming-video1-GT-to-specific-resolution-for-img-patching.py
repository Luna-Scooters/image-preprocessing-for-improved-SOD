import os
import cv2 
def convert_yolo_format(file_path, output_path, old_width, old_height, new_width, new_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # Since the coordinates are normalized, they remain the same
        new_x = x
        new_y = y
        new_w = w
        new_h = h
        
        new_line = f"{class_id} {new_x} {new_y} {new_w} {new_h}\n"
        new_lines.append(new_line)
    
    with open(output_path, 'w') as file:
        file.writelines(new_lines)



def draw_gt_boxes(color=(0, 255, 0), thickness=1):
    
    for img in sorted(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (640, 640))
        for label in sorted(os.listdir(output_directory)):
            print(label)
            if img[:-4] == label[:-4]:
                
                with open(output_directory + label,'r') as file:
                    lines = file.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

        
                    
                    x1 = int((x_center - w / 2) * image.shape[1])
                    y1 = int((y_center - h / 2) * image.shape[0])
                    x2 = int((x_center + w / 2) * image.shape[1])
                    y2 = int((y_center + h / 2) * image.shape[0])
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    # cv2.putText(image, class_id, (x1, y1), cv2.FONT_HERSHEYPLAIN, 1, color, thickness)
                cv2.imwrite(out_img_folder + img, image)


# Example usage
input_directory = '/Users/chinya07/Desktop/PROJECTS/LUNA/IDD_Annotation/video1/test_labels/'
output_directory = '/Users/chinya07/Desktop/PROJECTS/LUNA/IDD_Annotation/video1/640-640-transformed-test-labels/'
img_folder = '/Users/chinya07/Desktop/PROJECTS/LUNA/IDD_Annotation/video1/test_images'
out_img_folder = '/Users/chinya07/Desktop/PROJECTS/LUNA/IDD_Annotation/video1/640-640-images/'
old_width, old_height = 1280, 720
new_width, new_height = 640,640
# new_width, new_height = 320, 320

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(out_img_folder):
    os.makedirs(out_img_folder)

for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        convert_yolo_format(input_file_path, output_file_path, old_width, old_height, new_width, new_height)
draw_gt_boxes()
print("Conversion complete.")
