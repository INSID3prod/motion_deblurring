import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import v2

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class pre_Unet_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)
        self.DWSconv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.in_conv(x)
        return self.DWSconv(x)


class pre_Unet(nn.Module):
    def __init__(self, layers: int = 4, base_channels: int = 8, out_channels: int = 3):
        super().__init__()
        assert layers > 2

        self.encoders = nn.Sequential()
        self.neck = nn.Sequential()
        self.decoders = nn.Sequential()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.encoders.append(pre_Unet_block(3, base_channels))
        self.decoders.append(pre_Unet_block(base_channels * 2, base_channels))

        for l in range(layers - 2):
            ch = base_channels * 2 ** l
            self.encoders.append(pre_Unet_block(ch, 2 * ch))
            self.decoders.append(pre_Unet_block(4 * ch, 2 * ch))

        self.neck.append(pre_Unet_block(ch * 2, ch * 4))
        self.dropout = nn.Dropout2d(p=0.1)

        for l in range(layers - 1):
            ch = base_channels * 2 ** l
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2, 2),
                LayerNorm2d(ch)
            ))
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(2 * ch, ch, (2, 2), stride=2),
                LayerNorm2d(ch)
            ))

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        input = x
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.neck(x)
        x = self.dropout(x)

        for up, skip, decoder in zip(self.ups[::-1], skips[::-1], self.decoders[::-1]):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, skip.shape[-2:], mode='nearest')
            x = torch.cat((x, skip), dim=1)
            x = decoder(x)

        return input + self.out_conv(x)


class NAF_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1),
            nn.Sigmoid(),
        )
        self.block_1 = nn.Sequential(
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels),
            nn.GELU(),
            self.channel_attention,
            nn.Conv2d(out_channels, out_channels, (1, 1)),
        )
        self.block_2 = nn.Sequential(
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
        )

    def forward(self, x):
        x = self.in_conv(x)
        res_1 = x
        x = self.block_1(res_1)
        res_2 = x + res_1
        x = self.block_2(res_2)
        return x + res_2


class my_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels)
        )
        self.gate = nn.Sequential(
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.Sigmoid()
        )
        self.block = nn.Sequential(
            LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
        )

    def forward(self, x):
        x = self.in_conv(x)
        skip = x
        x = self.gate(x)
        x = x * skip
        skip = x
        x = self.block(x)
        return x + skip


class Encoder_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.basic_block = my_block(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.basic_block(x)
        x = self.max_pool(skip)
        return x, skip


class Neck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.basic_block = my_block(in_channels, out_channels)

    def forward(self, x):
        return self.basic_block(x)


class Decoder_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.transp_conv = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), 2)
        self.basic_block = my_block(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.transp_conv(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')

        x = torch.cat((x, skip), dim=1)  # in_channels
        return self.basic_block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        """
        self.encoders = nn.Sequential()
        self.decoders = nn.Sequential()
        self.neck = nn.Sequential()

        self.encoders.append(Encoder_Block(3,channels))
        for l in range(levels-2):
            self.encoders.append(Encoder_Block(channels * 2**l,channels * 2**(l+1)))
            self.decoders.append(Decoder_Block())
        """

        self.encoder_block_1 = Encoder_Block(9, 16)
        self.encoder_block_2 = Encoder_Block(16, 32)
        self.encoder_block_3 = Encoder_Block(32, 64)
        # self.encoder_block_4 = Encoder_Block(128,256)

        self.neck = Neck(64, 128)
        self.dropout = nn.Dropout2d(p=0.1)

        self.decoder_block_1 = Decoder_Block(128, 64)
        self.decoder_block_2 = Decoder_Block(64, 32)
        self.decoder_block_3 = Decoder_Block(32, 16)
        # self.decoder_block_4 = Decoder_Block(64,32)

        self.conv_2_mask = nn.Conv2d(16, 9, (3, 3), padding=1)

    def forward(self, x):
        x, skip_1 = self.encoder_block_1(x)  # [B, 64, 360, 640]
        x, skip_2 = self.encoder_block_2(x)
        x, skip_3 = self.encoder_block_3(x)
        # x, skip_4 = self.encoder_block_4(x) # [B, 1024, 45, 80]

        x = self.dropout(x)
        x = self.neck(x)

        x = self.decoder_block_1(x, skip_3)
        x = self.decoder_block_2(x, skip_2)
        x = self.decoder_block_3(x, skip_1)
        # x = self.decoder_block_4(x,skip_1)

        x = self.conv_2_mask(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()

    def forward(self, x):
        x_f = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W / 2 + 1]

        x_mag = torch.abs(x_f)
        x_phase = torch.angle(x_f)

        x_mag_log = torch.log1p(x_mag)

        x_f_cat = torch.cat((x_mag_log, torch.sin(x_phase), torch.cos(x_phase)), dim=1)

        H_cat = self.unet(x_f_cat)  # [B, 9, H, W / 2 + 1]
        ch = H_cat.shape[1]
        b = ch // 3

        H_mag_log = H_cat[:, 0:b]
        H_sin = H_cat[:, b:2 * b]
        H_cos = H_cat[:, 2 * b:]

        H_phase = torch.cat((H_sin, H_cos), dim=1)
        H_phase = F.normalize(H_phase, dim=1, eps=1e-6)
        H_sin_norm = H_phase[:, :b]
        H_cos_norm = H_phase[:, b:]

        H_mag = F.softplus(H_mag_log)

        H = H_mag * torch.complex(H_cos_norm, H_sin_norm)
        # H_rgb = H.expand(-1, 3, -1, -1) # torch.cat((H,H,H), dim=1)

        return H, x_f


class Net(nn.Module):
    def __init__(self, SNR: float):
        super().__init__()
        # self.in_conv = nn.Conv2d(3,3,(3,3),padding=1)
        self.out_conv = nn.Conv2d(3, 3, (3, 3), padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.generator = Generator()
        self.pre_unet = pre_Unet(layers=4, base_channels=16, out_channels=3)
        self.SNR = SNR

    def forward(self, x):
        h, w = x.shape[-2:]

        x = self.pre_unet(x)

        skip_2 = x
        # x = self.in_conv(x)
        H, x_f = self.generator(x)

        K = 1 / self.SNR
        W = torch.conj(H) / (torch.abs(H) ** 2 + K)

        pred_f = x_f * W  # deblurred spec

        pred = torch.fft.irfft2(pred_f, s=(h, w), norm='ortho')

        pred = self.out_conv(pred)
        pred = pred + skip_2

        return pred, pred_f, H