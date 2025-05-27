import os
import torch
import cv2 
import random

from PIL import Image 
from torchvision import transforms 

transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.6087, 0.6015, 0.5598), (0.1883, 0.1921, 0.2182))
]) 

class data_processing: 
    def __init__(self, image_path, logging_path, merge_log_file): 
        self.image_path = image_path
        self.logging_path = logging_path
        self.merge_log_file = merge_log_file 
        self.count = 0
        self.image_used = [] 

        self.merge_log_files() 
        self.count_files()

    def augment_image(self, front_img: Image, steering: float) -> (Image, float):
        is_flip = random.choice([True, False])

        if is_flip == True: 
            front_img = transforms.v2.functional.horizontal_flip(front_img)
            steering = steering * -1
        
        return (front_img, steering)
    
    def check_if_folders_exists(self) -> None: 
        if not os.path.exists(self.logging_path):
            print("Logging path doesn't exist")
            raise IOError("%s: %s" % (self.logging_path, "Logging path doesn't exist, I assume to check path in training.py")) 

        if not os.path.exists(self.merge_log_file):
            print("File for merged log file doesn't exist")
            raise IOError("%s: %s" % (self.merge_log_file, "Merge file doesn't exist, I assume that you need to check path in training.py"))

        if os.path.exists(self.merge_log_file + "merged_log_file.txt") == True: 
            return

    def merge_log_files(self) -> None: 
        file_count = 0 
        self.check_if_folders_exists()

        file_names = [] 

        # Check whether path exists and file isn't a hidden file
        for files in os.listdir(self.logging_path): 
            file = os.path.join(self.logging_path, files) 
            if os.path.exists(file) and not files.startswith("."): 
                file_names.append(file)
                file_count += 1

        # Create merged log file
        merged_log_dir = os.path.join(self.merge_log_file, "merged_log_file.txt") 
        merged_log_file = open(merged_log_dir, 'w') 

        i = 0
        # Iterate through the log files 
        while i < file_count: 
            log_file_name = os.path.join(self.logging_path, file_names[i]) 
            log_file = open(log_file_name) 
        
            # Write data from the log file to merged file
            for idx, line in enumerate(log_file): 
                merged_log_file.write(line) 
                 
            log_file.close() 
            i += 1

        merged_log_file.close() 

            
    def count_files(self) -> int:
        if not os.path.exists(self.merge_log_file + "merged_log_file.txt"): 
            print("Merge log path doesn't exist") 
            raise IOError("%s: %s" % (self.merge_log_file, "Merge log file text doesn't exist")) 

        self.count = 0
        log_file = open(self.merge_log_file + "merged_log_file.txt") 
        for (idx, line) in enumerate(log_file): 
            self.count += 1

        return self.count
            
    def __len__(self) -> int: 
        return self.count
             

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Check if Image path and Logging path 
        self.check_if_folders_exists()

        line_req = None
        log_file = open(self.merge_log_file + "merged_log_file.txt"); 
        ran_idx = None

        for (line_idx, line) in enumerate(log_file): 

            if line_idx == idx: 
                line_req = line            
                break 

        if line_req is None: 
            raise IndexError("No Index Found")

        # Split the line whenver there's a space
        line_arr = line_req.split(" ") 
        
        front_img_name = line_arr[1].split("/")[-1] 
        left_img_name = line_arr[2].split("/")[-1] 
        right_img_name = line_arr[3].split("/")[-1] 
        
        front_img_name = self.image_path + front_img_name + ".jpg"  
        left_img_name = self.image_path + left_img_name + ".jpg" 
        right_img_name = self.image_path + right_img_name + ".jpg"

        # Get all Images and steering value 
        front_img = Image.open(front_img_name) 
        right_img = Image.open(right_img_name) 
        left_img = Image.open(left_img_name) 

        # Crop the image to remove any uneccesarry part of the vehicle
        # front_img = right_img.crop((110, 0, 540, 450))
        front_img = right_img.crop((110, 0, 540, 450))

        # Convert to yuv image 
        front_image_yuv = front_img.convert("YCbCr")
        right_image_yuv = right_img.convert("YCbCr")
        left_image_yuv = left_img.convert("YCbCr")

        steering_val = int(line_arr[4]) 
        steering_val = 2 * ((steering_val - 451) / (573 - 451)) - 1 

        # front_image_yuv, steering_val = self.augment_image(front_image_yuv, steering_val)

        front_image_yuv = transform(front_image_yuv)
        right_image_yuv = transform(right_image_yuv)
        left_image_yuv = transform(left_image_yuv)

        # Max 573 
        # Min 451
        steering = torch.tensor(steering_val, dtype=torch.float32) 

        return front_image_yuv, steering 
        

if __name__ == '__main__':
    preprocess = data_processing()

    right_img, steering = preprocess[500]
    frnt_img = transforms.functional.to_pil_image(right_img)    
    # frnt_img = frnt_img.show()

    print(frnt_img.size())
    print("finish")
