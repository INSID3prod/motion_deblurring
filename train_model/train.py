import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import v2

from arcs.UNet_UNet_Wienner import Net
from utils.Losses import Charbonnier_Scharr

from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

#from fft_deblur.freq_filtering import fft_img, ifft_img, create_filter, create_gaussian_filter, normalize
from utils.Losses import Scharr_loss

if Path(r'/content/drive/MyDrive').exists():
    path = Path(r'/content/drive/MyDrive/gopro_deblur')
    format_path = path / 'format.json'
    blur_path = path / 'blur/images'
    sharp_path = path / 'sharp/images'
    model_save_dir = path / 'checkpoints'
    if not model_save_dir.exists():
        model_save_dir.mkdir()
else:
    path = Path(r'Z:\datasets\gopro_deblur')
    format_path = path / 'format.json'
    blur_path = path / 'blur\images renamed'
    sharp_path = path / 'sharp\images renamed'
    model_save_dir = Path(r'Z:\PYTHON\projects\Motion deblurring\models')

result_save_dir = Path(r'Z:\PYTHON\projects\Motion deblurring\train_model\results')

if not model_save_dir.exists():
    model_save_dir.mkdir()

MODEL_NAME = '384x384_20_images'

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop((384, 384)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomChoice([
        v2.Lambda(lambda x: x),
        v2.RandomRotation((90, 90)),
        v2.RandomRotation((180, 180)),
        v2.RandomRotation((270, 270)),
    ]),
    v2.ToDtype(torch.float32, scale=True),
])
val_transforms = v2.Compose([
    v2.ToImage(),
    # v2.RandomCrop((384, 384)),
    v2.ToDtype(torch.float32, scale=True),
])


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    return generator

generator = set_seed(42)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cuda')):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']

    return start_epoch, best_val_loss

class Dataset(data.Dataset):
    def __init__(self, blur_path, sharp_path, transforms):
        self.blur_imgs = blur_path
        self.sharp_imgs = sharp_path
        if len(self.blur_imgs) != len(self.sharp_imgs):
            raise ValueError("wrong number of imgs")
        else:
            self.len = len(self.blur_imgs)
        self.transforms = transforms

    def __getitem__(self, index):
        blur_img = cv2.imread(self.blur_imgs[index])
        gt_img = cv2.imread(self.sharp_imgs[index])

        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)  # [0,1]
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)  # [0,1]

        blur, gt = self.transforms(blur_img, gt_img)

        return blur, gt

    def __len__(self):
        return self.len



blur_images = sorted(os.path.join(blur_path, i) for i in os.listdir(blur_path))
sharp_images = sorted(os.path.join(sharp_path, i) for i in os.listdir(sharp_path))

tr_bound = int(0.7 * len(blur_images))
v_bound = int(tr_bound + 0.15 * len(blur_images))

train_blur = blur_images[:tr_bound]
train_sharp = sharp_images[:tr_bound]

val_blur = blur_images[tr_bound:v_bound]
val_sharp = sharp_images[tr_bound:v_bound]

test_blur = blur_images[v_bound:]
test_sharp = sharp_images[v_bound:]

train_dataset = Dataset(train_blur, train_sharp, transforms=train_transforms)
val_dataset = Dataset(val_blur, val_sharp, transforms=val_transforms)
test_dataset = Dataset(test_blur, test_sharp, transforms=val_transforms)

tiny_dataset = Dataset(train_blur, train_sharp, transforms=train_transforms)
tiny_dataset = torch.utils.data.Subset(tiny_dataset, range(0,20))

tiny_dataset_val = Dataset(train_blur, train_sharp, transforms=val_transforms)
tiny_dataset_val = torch.utils.data.Subset(tiny_dataset_val, [0,1])
print(f"train: {len(train_dataset)} pics\n val: {len(val_dataset)} pics\n test: {len(test_dataset)} pics\n ")

random_pic = train_dataset.__getitem__(np.random.randint(0, len(train_dataset)))[0]
h, w = random_pic.shape[-2:]
print(f'shape = {h}, {w}')

r = 0.09 * h
r_outer = 0.7 * h
# f = create_gaussian_filter(h,w,r) * create_gaussian_filter(h,w,r_outer,False)
#f = create_filter(h, w, h * 0.06) * create_filter(h, w, h * 0.3, False)

#################
DEVICE = torch.device("cuda:0")

MODEL = Net(10000).to(DEVICE)

BATCH_SIZE = 10

EPOCHS =25000

LR = 0.0003

OPTIMIZER = torch.optim.Adam(
    params=MODEL.parameters(),
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

#################

train_load = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_load = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_load = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

tiny_load = DataLoader(tiny_dataset, batch_size=BATCH_SIZE, shuffle=True)
tiny_load_val = DataLoader(tiny_dataset_val, batch_size=1, shuffle=False)


SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(
    OPTIMIZER,
    T_max=EPOCHS * len(tiny_load),
    eta_min=1e-5
)

def train_and_val(train_load, val_load,
                  device=DEVICE,
                  model=MODEL,
                  epochs=EPOCHS,
                  optimizer=OPTIMIZER,
                  scheduler=SCHEDULER
                  ):

    checkpoint_path = model_save_dir / 'last.pt'
    if checkpoint_path.exists():
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    train_history = []
    val_history = []

    print(f'{epochs * len(train_load)} grad descents training is started')

    for epoch in range(start_epoch, epochs):
        model.train()
        print(f"\n______EPOCH: {epoch + 1}/{epochs}______")
        train_tqdm = tqdm(train_load, leave=True)
        mean_loss = 0
        for b, (blur, sharp) in enumerate(train_tqdm):
            blur, sharp = blur.to(device), sharp.to(device)
            pred, pred_f, H = model(blur)

            # sharp_mag, _ = fft_img(sharp, True)
            # loss = mix_loss(pred, sharp, pred_mag, sharp_mag, a = 0.7)
            # loss = Charbonnier(pred, sharp)
            loss = Charbonnier_Scharr(pred, sharp, a=0.7)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            mean_loss += loss.item()
            current_mean = mean_loss / (b + 1)
            train_tqdm.set_description(f"TRAIN: current loss = {current_mean:.5f}")
        train_loss = mean_loss / len(train_load)
        train_history.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_tqdm = tqdm(val_load, leave=True)

            mean_loss = 0
            for b, (blur, sharp) in enumerate(val_tqdm):
                blur, sharp = blur.to(device), sharp.to(device)
                pred, pred_f, H = model(blur)

                # sharp_mag, _ = fft_img(sharp, True)
                # loss = mix_loss(pred, sharp, pred_mag, sharp_mag, a = 0.7)
                # loss = Charbonnier(pred, sharp)
                loss = Charbonnier_Scharr(pred, sharp, a=0.7)

                mean_loss += loss.item()
                current_loss = mean_loss / (b + 1)
                val_tqdm.set_description(f"VAL: current loss = {current_loss:.5f}")
            val_loss = mean_loss / len(val_load)
            val_history.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_dir / (MODEL_NAME + f'_best.pt'))
                print('Best model saved!')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
            }, model_save_dir / f'last.pt')

            if (epoch + 1) % 100 == 0:
                plot_learning_metrics(train_history, val_history, fig_name= 'auto_save', show=False)
                print('metrics added to results')

    return train_history, val_history

def plot_learning_metrics(*args, fig_name: str = '', show = True):
    plt.title("Loss")
    plt.plot(args[0], label='train')
    plt.plot(args[1], label='val')
    if show:
        plt.show()
    if not result_save_dir.exists():
        result_save_dir.mkdir()
    plt.savefig(result_save_dir / ('learning_metrics_' + fig_name + '.png'))


if __name__ == "__main__":
    train_history, val_history = train_and_val(tiny_load, tiny_load_val)
    # train_history, val_history = train_and_val(train_load, val_load)

    plot_learning_metrics(train_history, val_history, fig_name= 'final', show=True)