import torch

import torch.nn as nn
import sys; sys.path.append('..')
import preprocessing

from torch.utils.data import DataLoader
from gan import Generator


def training(): 
    EPOCH = 1

    image_folder = "/Volumes/joeham/logging_camera_down/image_data/" 
    logging_folder = "/Volumes/joeham/logging_camera_down/logging_data/"
    merge_folder = "/Volumes/joeham/logging_camera_down/"

    if sys.platform == "linux": 
        image_folder = "/media/jojo-main/joeham/logging_camera_down/image_data"
        logging_folder = "/media/jojo-main/joeham/logging_camera_down/logging_data"
        merge_folder = "/media/jojo-main/joeham/logging_camera_down/"

    data = preprocessing.data_processing(image_folder, logging_folder, merge_folder)
    train_dataloader = DataLoader(data, batch_size=10, shuffle=True)

    model = Generator() 
    for epoch in range(EPOCH): 
        for (front_img, steering) in train_dataloader: 
            y = model(front_img)
            break
        
    


if __name__ == '__main__': 
    training()
