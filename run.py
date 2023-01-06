import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import argparse
from models.unet.unet import Unet
from util import (
    load_checkpoint,
    save_checkpoint,
    get_train_loader,
    get_val_loader,
    save_predictions_as_imgs
) 
from models.hrnet import config
from models.SETR.setr import ViT
from models.swin.swin import SwinTransformer
from models.hrnet.hrnet import get_seg_model
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(args):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "hrnet":
        config.defrost()
        config.merge_from_list(
            [
                'DATASET.NUM_CLASSES', 1, 
                'MODEL.PRETRAINED', './pretrained/hrnetv2_w48_imagenet_pretrained.pth', 
            ]
        ) 
        config.freeze()
        model = get_seg_model(config).to(DEVICE)
    elif args.model == 'unet':
        model = Unet().to(DEVICE)
    elif args.model == 'swin':
        model = SwinTransformer().to(DEVICE)
    elif args.model == 'setr':
        model = ViT().to(DEVICE)

    return model


# training the model
def train(args):
    BATCH_SIZE = args.batch_size
    TRAIN_IMG_DIR = f'{args.train_dst_dir}/image'
    TRAIN_MASK_DIR = f'{args.train_dst_dir}/mask'

    # preprocess the input
    train_transform = A.Compose(
        [
            A.Resize(height=480, width=640),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ), 
            ToTensorV2(),
        ], 
    ) 

    model = get_model(args)

    loss_fn = nn.BCEWithLogitsLoss()

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = get_train_loader(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        NUM_WORKERS,
        PIN_MEMORY=True,
    )

    if args.load_ckpt:
        load_checkpoint(torch.load(f'{args.ckpt_folder}/{args.type}/{args.ckpt}'), model) 
 
    # training starts
    scaler = torch.cuda.amp.GradScaler()
    epoch = 0
    for epoch in range(args.epochs):
        count = 0
        running_loss = 0.0
        loss_values = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)
            # forward
            with torch.cuda.amp.autocast(): 
                predictions = model(data)
                loss = loss_fn(predictions, targets) 
                writer.add_scalar("Loss(train) - Epoch", loss, epoch) 

            # backward 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer) 
            scaler.update() 
            running_loss += loss.item()
            loss_values.append(running_loss)

            if batch_idx % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 10))
                running_loss = 0.0

            print(f'NUMBER OF ELEMENTS : {count}') 
            count+=1

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        loss = loss_fn(predictions, targets) 
        writer.add_scalar("Loss(train) - Epoch", loss, epoch) 
     

        save_checkpoint(args, checkpoint) 
        epoch+=1

writer.flush()


# put the model into test
def evaluate(args):
    
    BATCH_SIZE = args.batch_size
    VAL_IMG_DIR = f'{args.val_dst_dir}'

    val_transforms = A.Compose(
        [
            A.Resize(height=480, width=640),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_loader = get_val_loader(
        VAL_IMG_DIR,
        BATCH_SIZE,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY=True,
    )
    
    model = get_model(args)
    if args.load_ckpt:
        load_checkpoint(torch.load(f'{args.ckpt_folder}/{args.type}/{args.ckpt}'), model)

    save_predictions_as_imgs(
            val_loader, model, folder=f"{args.output_dir}", device=DEVICE
            ) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand/Held Object Segmentation Models')
    
    # mode
    parser.add_argument('--train', help='model training')
    parser.add_argument('--eval', help='model evaluation')

    # training/evaluation params
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--optim', help='optimizer choice', choices=["adam", "sgd", "adamW"])
    parser.add_argument('--lr', help='learning rate', default=1e-5)
    parser.add_argument('--epochs', default=5, help="number of training epochs")
    parser.add_argument('--model', default=5, help="architecture of choice", choices=['unet', 'hrnet', 'vit', 'swin', 'segformer'])
    parser.add_argument('--ckpt_folder', default='./checkpoints', help="Save/load your checkpoints to/from here")
    parser.add_argument('--load_ckpt', default=False, help='Load checkpoint')
    parser.add_argument('--ckpt', default=None, help='Checkpoint to be used') 
    parser.add_argument('--val_dst_dir', help='Dataset of images to evaluate model on') 
    parser.add_argument('--output_dir', help='Folder to save predictions in') 
    parser.add_argument('--train_dst_dir', help='Path to dataset of images to train model on. The folder should have image and mask folder inside') 
    parser.add_argument('--type', choices=['hands', 'held_objects'])


    args = parser.parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        raise Exception('Please set a mode for training or evaluation') 