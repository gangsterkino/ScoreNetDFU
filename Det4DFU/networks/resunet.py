from networks.custom_modules.basic_modules import *
from utils.misc import initialize_weights
class Baseline(nn.Module):
    def __init__(self, img_ch=3, num_classes=7, depth=8, use_residual=True):
        super(Baseline, self).__init__()

        chs = [64, 128, 256, 512, 512]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = EncoderBlock(img_ch, chs[0], depth=depth, use_res=use_residual)
        self.enc2 = EncoderBlock(chs[0], chs[1], depth=depth, use_res=use_residual)
        self.enc3 = EncoderBlock(chs[1], chs[2], depth=depth, use_res=use_residual)
        self.enc4 = EncoderBlock(chs[2], chs[3], depth=depth, use_res=use_residual)
        self.enc5 = EncoderBlock(chs[3], chs[4], depth=depth, use_res=use_residual)

        self.dec4 = DecoderBlock(chs[4], chs[3], use_deconv=use_residual)
        self.decconv4 = EncoderBlock(chs[3] * 2, chs[3], use_res=use_residual)

        self.dec3 = DecoderBlock(chs[3], chs[2], use_deconv=use_residual)
        self.decconv3 = EncoderBlock(chs[2] * 2, chs[2], use_res=use_residual)

        self.dec2 = DecoderBlock(chs[2], chs[1], use_deconv=use_residual)
        self.decconv2 = EncoderBlock(chs[1] * 2, chs[1], use_res=use_residual)

        self.dec1 = DecoderBlock(chs[1], chs[0], use_deconv=use_residual)
        self.decconv1 = EncoderBlock(chs[0] * 2, chs[0], use_res=use_residual)

        self.conv_1x1 = nn.Conv2d(chs[0], num_classes, 1, bias=False)

        initialize_weights(self)

    def forward(self, x):
        # encoding path
        x1 = self.enc1(x)

        x2 = self.maxpool(x1)
        x2 = self.enc2(x2)

        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)

        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)

        x5 = self.maxpool(x4)
        x5 = self.enc5(x5)

        # decoding + concat path
        d4 = self.dec4(x5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.decconv4(d4)

        d3 = self.dec3(d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.decconv3(d3)

        d2 = self.dec2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.decconv2(d2)

        d1 = self.dec1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.decconv1(d1)

        d1 = self.conv_1x1(d1)

        return d1
