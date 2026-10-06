import math
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torchsummary import summary

import matplotlib.pyplot as plt

from train_model.train import Net, tiny_load_val, tiny_load, train_load

model_save_dir = Path(r'Z:\PYTHON\projects\Motion deblurring\models')
MODEL_NAME = '5 images_best.pt'
model_path = model_save_dir / (MODEL_NAME)
checkpoint_path = model_save_dir / '5 images last.pt'

LAST = False

device = torch.device("cuda:0")

def show_img_from_tensor(pic,label: str = None):
    pic = pic.cpu().squeeze(0).numpy().transpose(1, 2, 0)
    plt.imshow(pic)
    if label is not None:
        plt.title(label)
    plt.show()

def img_from_tensor(pic):
    return pic.cpu().squeeze(0).numpy().transpose(1, 2, 0)

def sobel(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x,sobel_y)
    return sobel

def show_images(blur, sharp, pred, loss, sobel_diff, title):
    blur = img_from_tensor(blur)
    sharp = img_from_tensor(sharp)
    pred = img_from_tensor(pred)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))

    fig.suptitle(title + ',  ' + loss, fontsize=16)

    axes[0, 0].imshow(blur)
    axes[0, 0].set_title('blurred image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(5 * (pred - blur))
    axes[0, 1].set_title('5 * (pred - blur)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(pred)
    axes[1, 0].set_title('deblurred image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sharp)
    axes[1, 1].set_title('sharp image')
    axes[1, 1].axis('off')

    axes[0, 2].imshow(5 * (sharp - blur))
    axes[0, 2].set_title('5 * (sharp - blur)')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(sobel_diff, cmap='gray')
    axes[1, 2].set_title('sobel(sharp) - sobel(blur)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

model = Net(10000).to(device)

if checkpoint_path.exists() and LAST:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f'\nmodel summary:')
#summary(model, input_size=(3,720,1280))

with torch.inference_mode():
    blur_pic, sharp_pic = tiny_load_val.dataset.__getitem__(0)
    blur_pic, sharp_pic = blur_pic.to(device).unsqueeze(0), sharp_pic.to(device).unsqueeze(0)

    pred, pred_f, H = model(blur_pic)
    #pred = model(blur_pic)
    #print(x_f_hat.real.max(),x_f_hat.real.min(),x_f_hat.real.mean())
print(f'pred stats = {pred.mean().item(), pred.max().item(), pred.min().item()}')

mse = nn.MSELoss()(pred, sharp_pic)
stats_str = f'MSE loss = {mse:.5f}, PSNR = {10 * math.log10(1 / mse)}'
print(stats_str)

show_images( blur_pic,sharp_pic,pred,stats_str,(sobel(img_from_tensor(sharp_pic)) - sobel(img_from_tensor(blur_pic))),'Images comparison')


def show_masks(masks,coeffs,title):
    x = masks.shape[1]
    m = int(math.sqrt(x))
    if x % m == 0:
        nrows = m
        ncols = int(x / m)
    else:
        while x % m != 0:
            x += 1
        nrows = m
        ncols = int(x / m)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
    fig.suptitle(title)

    for row in range(nrows):
        for col in range(ncols):
            idx = col + ncols * row
            img = masks[:,idx].cpu().numpy().transpose(1,2,0)
            axes[row, col].imshow(0.33*torch.sum(img, dim=0, keepdim=True))
            axes[row, col].set_title(f"mask {idx + 1}, c = {coeffs[:,idx].item():.5f}")
            axes[row, col].axis('off')

    plt.show()
#num_masks = masks.shape[1] // 2
#show_masks(masks[:,:num_masks],c,'Mag freq masks')
#show_masks(masks[:,num_masks:],c,'Phase freq masks')

#show_img_from_tensor(H.real, 'freq mask // АЧХ смаза')

def normilize_tensor(x):
    x = torch.fft.fftshift(x)
    log_x = torch.log1p(torch.abs(x))
    log_x += log_x.mean()
    log_x = 255.0 * log_x / log_x.max()
    return log_x

#spec_mse = specMSE(sharp_pic,pred) - normilize_tensor(torch.fft.fft2(sharp_pic) f'spec_pred - spec_sharp (spec MSE = {spec_mse.item():.5f})'
#show_img_from_tensor((  normilize_tensor(x_f_hat)   ))