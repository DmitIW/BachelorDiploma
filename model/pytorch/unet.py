# 3td party import
from fastai.vision import (
    nn,  # torch.nn
    F,   # torch.nn.functional
)
from torch import cat as t_cat
# Internal modules import
from model.pytorch.utility import (
    SavedFeatures
)


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()

        kernel, stride = 3, 2
        up_out = x_out = n_out // 2

        self.conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.conv_1 = nn.Conv2d(n_out, n_out, kernel, padding=kernel // 2)
        self.conv_2 = nn.Conv2d(n_out, n_out, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, prev_x, backbone_x):
        up_x = self.tr_conv(prev_x)
        conv_backbone_x = self.conv(backbone_x)
        cat_x = t_cat([up_x, conv_backbone_x], dim=1)
        conv1_x = F.relu(self.conv_1(cat_x))
        return self.bn(self.conv_2(conv1_x))


class UnetR34(nn.Module):
    def __init__(self, backbone, classes_num=255):
        super().__init__()

        self.backbone = backbone
        self.backbone_outs = [
            SavedFeatures(backbone[i]) for i in [2, 4, 5, 6]
        ]

        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, classes_num, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.backbone(x))

        x = self.up1(
            x, self.backbone_outs[3].feature
        )
        x = self.up2(
            x, self.backbone_outs[2].feature
        )
        x = self.up3(
            x, self.backbone_outs[1].feature
        )
        x = self.up4(
            x, self.backbone_outs[0].feature
        )

        return self.up5(x)

    def close(self):
        for out in self.backbone_outs:
            out.remove()
