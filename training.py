import torch 
import torchvision 
import cv2
import matplotlib.pyplot as plt
import sys

import preprocessing 

from torch.utils.data import DataLoader
from torchvision import transforms 
from torchvision.transforms import v2
from nn import End_to_End_NN
from torch import nn
from PIL import Image


EPOCH = 5
transform = transforms.Compose([
    transforms.ToTensor(), 
]) 

def eval_model() -> None: 
    model = End_to_End_NN()
    model.load_state_dict(torch.load("/Volumes/joeham/model_extra_epoch_pil_less.pth", weights_only=False)) 
    model.eval()  

    # Validation data
    image_folder = "/Volumes/joeham/valid_test_1/image_data/"
    logging_folder = "/Volumes/joeham/valid_test_1/logging_data/"
    merge_folder = "/Volumes/joeham/valid_test_1/"

    data = preprocessing.data_processing(image_folder, logging_folder, merge_folder)
    valid_dataloader = DataLoader(data, batch_size=1, shuffle=False)

    epoch = 1
    correct = 0
    is_exit = None

    with torch.no_grad():
        for (front_img, steering) in valid_dataloader: 
            output = model(front_img)
            accuracy = is_correct(steering, output)

            # Every 100 Images stop and anaylze the image and assoicated steering 
            if epoch % 100 == 0 or is_exit == "x": 
                is_exit = input("Press n for continue otherwise x to exit: ")

                if is_exit == "x": 
                    epoch += 1
                    break
                else: 
                    print("Current steering checkpoint:", str(output))
                    print("True steering checkpoint: ", str(steering))

                    front_img = torch.squeeze(front_img)
                    front_img = transforms.functional.to_pil_image(front_img)
                    front_img.show()

                    epoch += 1
                    is_exit = input("Stopping, press any chracter to continue otherwise press \"x\" to exit : ")
                    if is_exit == "x":
                        return 

                    continue 


            if accuracy == 1.0:
                correct += 1 


            print("Current steering:", str(output))
            print("True steering: ", str(steering))
            print("Accuracy of the current Model: ", (correct / epoch))
            print("\n")
            epoch += 1 
            
    print("Total Accuracy of the current model: ", (correct / epoch))


def is_correct(steering, output, threshold=0.3) -> float: 
    correct = torch.abs(steering - output) < threshold
    accuracy = correct.float().mean().item() 
    return accuracy 

def train() -> None:
    save_model = input("Save model? (y or n) ") 

    # Training data 
    image_folder = "/Volumes/joeham/logging_camera_down/image_data/" 
    logging_folder = "/Volumes/joeham/logging_camera_down/logging_data/"
    merge_folder = "/Volumes/joeham/logging_camera_down/"

    if sys.platform == "linux": 
        image_folder = "/media/jojo-main/joeham/logging_camera_down/image_data"
        logging_folder = "/media/jojo-main/joeham/logging_camera_down/logging_data"
        merge_folder = "/media/jojo-main/joeham/logging_camera_down/"

    data = preprocessing.data_processing(image_folder, logging_folder, merge_folder)
    train_dataloader = DataLoader(data, batch_size=10, shuffle=True)

	# Load the Model
    model = End_to_End_NN()
    loss_fn = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 

    total_loss = 0  
    loss_every_epoch = [] 
    correct = 0

    for epoch in range(EPOCH):
        for (front_img, steering) in train_dataloader: 
            optimizer.zero_grad()

            output = model(front_img) 
            steering = steering.unsqueeze(1)

            loss = loss_fn(output, steering) 
            accuracy = is_correct(steering, output)

            if accuracy == 1.0: 
                correct += 1

            loss.backward() 
            optimizer.step()

            loss_every_epoch.append(loss.item()) 
            print("The loss is: " + str(loss.item())) 
            total_loss = 0
    
        print(f"Checkpoint {epoch}")
        print("Saving Model!....")
        torch.save(model.state_dict(), "/Volumes/joeham/model_extra_epoch_pil_less.pth") 
        print("The accuracy is: " + str(correct / (len(train_dataloader) * (epoch + 1))))

    plt.plot(loss_every_epoch) 
    plt.xlabel("Iteration") 
    plt.ylabel("Loss") 
    plt.title("Training Loss") 
    plt.show() 

    if save_model == "y" and sys.platform == "darwin": 
        torch.save(model.state_dict(), "/Volumes/joeham/model_extra_epoch_pil_less.pth") 
    elif save_model == "y" and sys.platform == "linux": 
        torch.save(model.state_dict(), "/media/jojo-main/joeham/model_batch_size_1.pth") 


def main(): 
    is_train_or_valid = int(input("Would you like to train a model (1) or validate? (2) ")) 

    if is_train_or_valid == 1: 
        train()
        return 
    
    eval_model()

if __name__ == '__main__': 
    main()
