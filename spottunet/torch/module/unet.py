from torch import nn

from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.conv import PreActivation2d


class UNet2D(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, n_filters_init=8):
        super().__init__()
        self.n_filters_init = n_filters_init
        n = n_filters_init

        self.init_path = nn.Sequential(
            nn.Conv2d(n_chans_in, n, kernel_size=3, padding=1, bias=False),
            ResBlock2d(n, n, kernel_size=3, padding=1),
            ResBlock2d(n, n, kernel_size=3, padding=1),
            ResBlock2d(n, n, kernel_size=3, padding=1),
        )
        self.shortcut0 = nn.Conv2d(n, n, kernel_size=1, padding=0)

        self.down1 = nn.Sequential(
            PreActivation2d(n, n * 2, kernel_size=2, stride=2, bias=False),
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1),
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1),
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1)
        )
        self.shortcut1 = nn.Conv2d(n * 2, n * 2, kernel_size=1, padding=0)

        self.down2 = nn.Sequential(
            PreActivation2d(n * 2, n * 4, kernel_size=2, stride=2, bias=False),
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1),
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1),
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1)
        )
        self.shortcut2 = nn.Conv2d(n * 4, n * 4, kernel_size=1, padding=0)

        self.bottleneck = nn.Sequential(
            PreActivation2d(n * 4, n * 8, kernel_size=2, stride=2, bias=False),
            ResBlock2d(n * 8, n * 8, kernel_size=3, padding=1),
            ResBlock2d(n * 8, n * 8, kernel_size=3, padding=1),
            ResBlock2d(n * 8, n * 8, kernel_size=3, padding=1),
            nn.ConvTranspose2d(n * 8, n * 4, kernel_size=2, stride=2, bias=False),
        )

        self.up2 = nn.Sequential(
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1),
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1),
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1),
            nn.ConvTranspose2d(n * 4, n * 2, kernel_size=2, stride=2, bias=False),
        )

        self.up1 = nn.Sequential(
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1),
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1),
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1),
            nn.ConvTranspose2d(n * 2, n, kernel_size=2, stride=2, bias=False),
        )

        self.out_path = nn.Sequential(
            ResBlock2d(n, n, kernel_size=3, padding=1),
            ResBlock2d(n, n, kernel_size=3, padding=1),
            ResBlock2d(n, n, kernel_size=3, padding=1),
            PreActivation2d(n, n_chans_out, kernel_size=1),
            nn.BatchNorm2d(n_chans_out)
        )

    def forward(self, x):

        x0 = self.init_path(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        x2_up = self.up2(self.bottleneck(x2) + self.shortcut2(x2))
        x1_up = self.up1(x2_up + self.shortcut1(x1))
        x_out = self.out_path(x1_up + self.shortcut0(x0))

        return x_out
