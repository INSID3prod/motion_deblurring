import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt

torch.cuda.empty_cache()

model_path = r'Z:\PYTHON\projects\PyTorch\my restormer\models\my_modes.pth'

blur_path = r'Z:\datasets\gopro_deblur\blur\images renamed'
sharp_path = r'Z:\datasets\gopro_deblur\sharp\images renamed'

device = torch.device("cuda:0")


def Scharr(img: torch.Tensor, blur = True):
    img = T.Grayscale()(img) # [B,1,H,W]
    if blur:
        img = T.GaussianBlur((3, 3), 1e-3)(img)
    kernel_x = torch.tensor([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3],
    ],dtype=img.dtype, device=img.device
    ).unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_x.transpose(-2, -1)
    sobel_x = F.conv2d(img, kernel_x)
    sobel_y = F.conv2d(img, kernel_y)
    return torch.sqrt(sobel_x**2 + sobel_y**2)

def Scharr_loss(pred: torch.Tensor, sharp: torch.Tensor):
    loss = torch.mean(torch.abs(pred - sharp))
    return loss

def Charbonnier(pred: torch.Tensor, sharp: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - sharp
    return torch.mean(torch.sqrt(diff ** 2 + eps ** 2))


def Charbonnier_Scharr(pred: torch.Tensor,
                       sharp: torch.Tensor,
                       a: float) -> torch.Tensor:
    char = Charbonnier(pred, sharp)
    scharr = Scharr_loss(pred, sharp)
    return a * char + (1 - a) * scharr

"""
def mix_loss(pred: torch.Tensor, sharp: torch.Tensor,
             pred_mag: torch.Tensor, sharp_mag: torch.Tensor,
             a: float) -> torch.Tensor:
    freq_loss = nn.MSELoss()(normalize(pred_mag), normalize(sharp_mag))
    space_loss = Charbonnier(pred, sharp)
    return a * space_loss + (1 - a) * freq_loss
"""

if __name__ == 'main':
    def show_img_from_tensor(pic, label: str = None, cmap: str = 'gray'):
        pic = pic.cpu().numpy().squeeze(0).transpose(1, 2, 0)
        plt.imshow(pic, cmap=cmap)
        if label is not None:
            plt.title(label)
        plt.show()


    def img_from_tensor(pic):
        return pic.cpu().squeeze(0).numpy().transpose(1, 2, 0)


    def test_loader(blur_path, sharp_path, num):
        num = str(num) + '.png'
        blur = cv2.imread(os.path.join(blur_path, num))
        sharp = cv2.imread(os.path.join(sharp_path, num))

        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        blur = torch.from_numpy(blur[:384, :384]).float()
        sharp = torch.from_numpy(sharp[:384, :384]).float()

        blur = blur.permute(2, 0, 1) / 255.
        sharp = sharp.permute(2, 0, 1) / 255.

        blur = blur.unsqueeze(0).to(device)
        sharp = sharp.unsqueeze(0).to(device)

        return blur, sharp


    blur_img, sharp_img = test_loader(blur_path, sharp_path, 300)
    print(f'Scharr loss = {Scharr_loss(blur_img, sharp_img).item():.5f}')

    show_img_from_tensor((Scharr(blur_img) - Scharr(sharp_img)) ** 2)
    """
    with torch.inference_mode():
        blur_pic, sharp_pic = test_loader(blur_path,sharp_path,300)

        pred = model(blur_pic)
        #pred = model(blur_pic)
        #print(x_f_hat.real.max(),x_f_hat.real.min(),x_f_hat.real.mean())
    print(f'pred stats = {pred.mean().item(), pred.max().item(), pred.min().item()}')

    loss = nn.MSELoss()(pred, sharp_pic)
    print(f'Loss = {loss:.5f}')

    #print(f'coeffs: {c}')

    show_images( blur_pic,sharp_pic,pred,loss,(sobel(img_from_tensor(sharp_pic)) - sobel(img_from_tensor(blur_pic))),'Images comparison')
    """