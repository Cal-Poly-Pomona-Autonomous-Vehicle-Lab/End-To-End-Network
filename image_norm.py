from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np 
import numpy.typing as npt

image_folder = ""
logging_folder = ""
merge_folder = ""

BATCH_SIZE = 1

def calculate_std(img: Image, width: int, height: int, mean: npt.NDArray) -> npt.NDArray: 
    yuv_std = np.zeros(3, dtype=np.float64)
    yuv_sum = np.zeros(3, dtype=np.float64) 

    for x in range(width): 
        for y in range(height): 
            pixel_val = np.array(img.getpixel((x, y)), dtype=np.float64)
            yuv_sum += (pixel_val) ** 2

    
    return yuv_sum

def calculate_mean(img: Image, width: int, height: int) -> npt.NDArray: 
    yuv_sum = np.zeros(3, dtype=np.float64)

    for x in range(width): 
        for y in range(height): 
            pixel_val = np.array(img.getpixel((x, y)), dtype=np.float64)
            yuv_sum += pixel_val

    return yuv_sum

def iterate_files() -> None: 
    total_pixels = 0

    yuv_mean = np.zeros(3, dtype=np.float64)
    yuv_std = np.zeros(3, dtype=np.float64)

    image_file_names = os.listdir(image_folder)

    print("Number of files: " + str(len(image_file_names)))
    processed_files = 1

    for image in image_file_names: 

        # Ensure hidden files are not included
        if image.startswith("."): 
            continue 

        if processed_files % 100 == 0: 
            print(f"Files processed {processed_files / len(image_file_names)} %")


        img = Image.open(image_folder + image)
        # Convert to YUV scale
        img = img.convert("YCbCr")
        
        (width, height) = img.size 
        total_pixels += width * height * BATCH_SIZE

        yuv_sum = calculate_mean(img, width, height)
        yuv_mean += yuv_sum 

        std = calculate_std(img, width, height, yuv_mean) 
        yuv_std += std 

        processed_files += 1
    
    yuv_mean /= total_pixels 


    # E[x^2] - E[x]^2
    # sqrt ( 1/N * sum(x^2) - (1/N * sum(x))^2) 
    yuv_std = np.sqrt((yuv_std / total_pixels) - yuv_mean ** 2)

    # convert to [0, 1] range
    print("Mean: " + str(yuv_mean / 255.0))
    print("Std: " + str(yuv_std / 255.0))


if __name__ == '__main__': 
    iterate_files()