"""PANNs MobileNetV1 inference architecture, adapted from the authors' model.

Copyright (c) 2018-2020 Qiuqiang Kong. MIT licensed; see PANNs notices.txt.
Source: github.com/qiuqiangkong/audioset_tagging_cnn, pytorch/models.py.
Training augmentation and initialization are omitted because the complete
pretrained state is loaded strictly before inference. Layer names are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=1024, hop_length=320, win_length=1024, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.AvgPool2d(stride), nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            conv_bn(1, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2),
            conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1),
            conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1),
            conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1))
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, 527, bias=True)

    def forward(self, audio):
        x = self.logmel_extractor(self.spectrogram_extractor(audio))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = torch.mean(self.features(x), dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        x = F.relu_(self.fc1(x))
        return torch.sigmoid(self.fc_audioset(x))
