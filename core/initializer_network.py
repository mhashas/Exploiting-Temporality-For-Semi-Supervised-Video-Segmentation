import torch
import torch.nn as nn

class InitializerNetwork(nn.Module):

    def __init__(self, args, backbone, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(InitializerNetwork, self).__init__()
        self.num_directions = 2 if args.lstm_bidirectional else 1
        lstm_size = int((min(args.resize) / 2 ** args.num_downs * args.ngf * 8 ))
        self.backbone = backbone(args.num_downs, input_nc, args.ngf, norm_layer)
        self.h_conv = nn.Sequential(*[nn.Conv2d(args.ngf*8, lstm_size, 1), nn.ReLU()])
        self.c_conv = nn.Sequential(*[nn.Conv2d(args.ngf*8, lstm_size, 1), nn.ReLU()])

    def forward(self, x):
        x = self.backbone(x)
        h = self.h_conv(x)
        c = self.c_conv(x)

        h = h.reshape(h.size(0), -1)
        c = c.reshape(c.size(0), -1)

        if self.num_directions == 2:
            h = torch.split(h, int(h.size(1)/2), dim=1)
            c = torch.split(c, int(c.size(1)/2), dim=1)
            h = torch.stack(h)
            c = torch.stack(c)
        else:
            h = torch.unsqueeze(h, 0)
            c = torch.unsqueeze(c, 0)

        return (h,c)