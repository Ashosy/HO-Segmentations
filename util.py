import torch 
import torchvision
from dataset import CustomDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np


def save_checkpoint(state, args):
    filename= f'{args.ckpt_folder}/{args.model}.pth'
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])



def get_train_loader(
    train_dir,
    train_maskdir,
    batch_size,
    train_transform,
    num_workers,
    pin_memory=True,
):
    train_ds = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader


def get_val_loader(
    val_dir,
    batch_size,
    val_transform,
    num_workers,
    pin_memory=True
):

    val_ds = CustomDataset( 
    image_dir=val_dir,
    transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory, 
        shuffle=False,
    )
    return val_loader

    

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device) 
            y = y.to(device).unsqueeze(1) 
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() 
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    print(dice_score) 
    print(len(loader))
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader): 
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png" 
        )
        # torchvision.utils.save_image(
        #     x, f"{folder}/actual_{idx}.png" 
        # )
    model.train() 