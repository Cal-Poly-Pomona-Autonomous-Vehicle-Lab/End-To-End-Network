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


EPOCH = 3

HARD_DRIVE_MACOS = "/Volumes"
HARD_DRIVE_LINUX = "/media"

# For Linux
COMPUTER_NAME = ""
HARD_DRIVE_NAME = "/joeham"

MODEL_NAME = "/model_extra_epoch_pil_less.pth"
MODEL_PATH = f"{HARD_DRIVE_NAME}{MODEL_NAME}"

TRAINING_DATA_NAME = "/valid_test_1"

MODEL_EVAL = "/model_extra_epoch_pil_less.pth"
model_eval_path = f"{HARD_DRIVE_NAME}{TRAINING_DATA_NAME}{MODEL_EVAL}"

image_folder = f"{HARD_DRIVE_NAME}{TRAINING_DATA_NAME}/image_data/"
logging_folder = f"{HARD_DRIVE_NAME}{TRAINING_DATA_NAME}/logging_data/"
merge_folder = f"{HARD_DRIVE_NAME}{TRAINING_DATA_NAME}/"

BATCH_SIZE = 10

transform = transforms.Compose([
    transforms.ToTensor(), 
]) 

def eval_model() -> None: 
    model = End_to_End_NN()
    model.load_state_dict(
        torch.load("{MODEL_PATH}", weights_only=False)) 
    model.eval()  


    match sys.platform: 
        case "linux": 
            image_folder = f"{HARD_DRIVE_LINUX}{COMPUTER_NAME}{image_folder}"
            logging_folder = f"{HARD_DRIVE_LINUX}{COMPUTER_NAME}{logging_folder}"
            merge_folder = f"{HARD_DRIVE_LINUX}{COMPUTER_NAME}{merge_folder}"
        case "darwin": 
            image_folder = f"{HARD_DRIVE_MACOS}{image_folder}"
            logging_folder = f"{HARD_DRIVE_MACOS}{logging_folder}"
            merge_folder = f"{HARD_DRIVE_LINUX}{merge_folder}"

    data = preprocessing.NNDataProcessing(image_folder, logging_folder, merge_folder)
    valid_dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    epoch = 1
    correct = 0
    is_exit = None

    with torch.no_grad():
        for (front_img, steering) in valid_dataloader: 
            output = model(front_img, steering)
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
    global model_eval_path 
    global logging_folder
    global merge_folder
    global image_folder

    save_model = input("Save model? (y or n) ") 

    match sys.platform: 
        case "linux": 
            model_eval_path = f"{HARD_DRIVE_LINUX}{model_eval_path}"

            image_folder = f"{HARD_DRIVE_LINUX}{COMPUTER_NAME}{image_folder}"
            logging_folder = f"{HARD_DRIVE_LINUX}{COMPUTER_NAME}{logging_folder}"
            merge_folder = f"{HARD_DRIVE_LINUX}{COMPUTER_NAME}{merge_folder}"
        case "darwin": 
            model_eval_path = f"{HARD_DRIVE_MACOS}{model_eval_path}"

            image_folder = f"{HARD_DRIVE_MACOS}{image_folder}"
            logging_folder = f"{HARD_DRIVE_MACOS}{logging_folder}"
            merge_folder = f"{HARD_DRIVE_MACOS}{merge_folder}"

    print(image_folder)

    data = preprocessing.NNDataProcessing(image_folder, logging_folder, merge_folder)
    train_dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

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

            steering = steering.unsqueeze(1)
            output = model(front_img, steering) 

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
        torch.save(model.state_dict(), f"/Volumes/joeham/{MODEL_NAME}") 
        print("The accuracy is: " + str(correct / (len(train_dataloader) * (epoch + 1))))

    plt.plot(loss_every_epoch) 
    plt.xlabel("Iteration") 
    plt.ylabel("Loss") 
    plt.title("Training Loss") 
    plt.show() 

    if save_model == "n": 
        return
    
    torch.save(model.state_dict(), model_eval_path)


def main(): 
    is_train_or_valid = int(input("Would you like to train a model (1) or validate? (2) ")) 

    if is_train_or_valid == 1: 
        train()
        return 
    
    eval_model()

if __name__ == '__main__': 
    main()
