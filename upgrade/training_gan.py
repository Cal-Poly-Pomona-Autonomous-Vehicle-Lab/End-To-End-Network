import torch

import torch.nn as nn
import sys; sys.path.append('..')
import preprocessing

import matplotlib.pyplot as plt
import matplotlib.image as mpng

from torch.utils.data import DataLoader
from dccgan import Generator, Discriminator


def training(): 
    EPOCH = 1

    device = None
    if torch.cuda.is_available(): 
        device = 'cuda'
    elif torch.backends.mps.is_available(): 
        device = 'mps'
    else:
        device = 'cuda'
    

    image_folder = "/Volumes/joeham/logging_camera_down/image_data/" 
    logging_folder = "/Volumes/joeham/logging_camera_down/logging_data/"
    merge_folder = "/Volumes/joeham/logging_camera_down/"

    if sys.platform == "linux": 
        image_folder = "/media/jojo-main/joeham/logging_camera_down/image_data"
        logging_folder = "/media/jojo-main/joeham/logging_camera_down/logging_data"
        merge_folder = "/media/jojo-main/joeham/logging_camera_down/"

    data = preprocessing.data_processing(image_folder, logging_folder, merge_folder)
    train_dataloader = DataLoader(data, batch_size=1, shuffle=True)
    
    Dis = Discriminator() 
    Gen = Generator(noise_size=100, conv_dim=3)

    optimizer_D = torch.optim.Adam(Dis.parameters(), lr=0.0001) 
    optimizer_G = torch.optim.Adam(Gen.parameters(), lr=0.0001) 

    loss_fn = nn.MSELoss()
    noise = torch.randn(1, 100, 1, 1)

    img_count = 1 
    total_img_count = len(data)
    for epoch in range(EPOCH): 
        for (front_img, steering) in train_dataloader: 

            if img_count > 1000: 
                break
            
            pred_ds = Dis.forward(front_img)
            fake_img = Gen.forward(noise)

            target = torch.full_like(pred_ds, 1.0)
            loss_real = loss_fn(pred_ds, target)

            output_fake = Dis.forward(fake_img)
            label_fake = torch.full_like(output_fake, 1)
            loss_fake = loss_fn(output_fake, label_fake)

            loss_D = 0.5 * (loss_real + loss_fake)

            Dis.zero_grad() 
            loss_D.backward() 
            optimizer_D.step()

            fake_output = Gen.forward(noise)

            f_output = Dis.forward(fake_output)
            label_gen = torch.full_like(f_output, 1)
            loss_G = 0.5 * loss_fn(f_output, label_gen)

            Gen.zero_grad() 
            loss_G.backward() 
            optimizer_G.step()

            print(f"Loss Dis: {loss_D}")
            print(f"Loss Gen: {loss_G}")
            print(f"Progress: {img_count / total_img_count * 100:.2f}%")
            img_count = img_count + 1


    f_img = Gen.forward(noise)

    f_img = f_img.detach().numpy() 
    f_img = f_img.squeeze()
    f_img = f_img.transpose(1, 2, 0)

    plt.imshow(f_img)
    plt.show()
            
def validation() -> None: 
    image_folder = "/Volumes/joeham/logging_camera_down/image_data/" 
    logging_folder = "/Volumes/joeham/logging_camera_down/logging_data/"
    merge_folder = "/Volumes/joeham/logging_camera_down/"

    if sys.platform == "linux": 
        image_folder = "/media/jojo-main/joeham/logging_camera_down/image_data"
        logging_folder = "/media/jojo-main/joeham/logging_camera_down/logging_data"
        merge_folder = "/media/jojo-main/joeham/logging_camera_down/"
            
    data = preprocessing.data_processing(image_folder, logging_folder, merge_folder)
    train_dataloader = DataLoader(data, batch_size=1, shuffle=True)
        
    


if __name__ == '__main__': 
    training()
