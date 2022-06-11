# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

class md_pointnet(nn.Module):

    def __init__(self, in_channel=3, conv_channels=[128, 128, 256, 512], aggregate_layer=3):
        super(md_pointnet, self).__init__()
        self.conv_channels = conv_channels
        self.aggregate_layer = aggregate_layer
        self.xyz_emb = nn.Conv2d(in_channel, conv_channels[0], kernel_size=1)
        self.xb = nn.BatchNorm2d(conv_channels[0])

        self.conv1 = nn.Conv2d(conv_channels[0]+3, conv_channels[1], kernel_size=1)
        self.conv2 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=1)
        self.conv3 = nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=1)
        self.b1 = nn.BatchNorm2d(conv_channels[1])
        self.b2 = nn.BatchNorm2d(conv_channels[2])
        self.b3 = nn.BatchNorm2d(conv_channels[3])

    def forward(self, input_ft, xyz):
        b, c, h, w = xyz.size()
        emb =F.relu(self.xb(self.xyz_emb(input_ft)), inplace=True)
        xyz_e = torch.cat([xyz, emb], dim=1)

        l1 = F.relu(self.b1(self.conv1(xyz_e)), inplace=True)
        l2 = F.relu(self.b2(self.conv2(l1)), inplace=True)
        l3 = self.b3(self.conv3(l2))

        gl_ft = F.adaptive_max_pool2d(l3, (1, 1))
        gl_ft = F.adaptive_avg_pool2d(gl_ft, (h, w))

        return torch.cat([l3, gl_ft], dim=1)

class md_pointnet_wo_xyz_cat(md_pointnet):
    def __init__(self,
                 in_channel=3,
                 conv_channels=[128, 128, 256, 512],
                 aggregate_layer=3):
        super(md_pointnet_wo_xyz_cat, self).__init__(in_channel,
                                                     conv_channels,
                                                     aggregate_layer)

        self.conv1 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=1)


    def forward(self, input_ft):
        b, c, h, w = input_ft.size()
        xyz_e =F.relu(self.xb(self.xyz_emb(input_ft)), inplace=True)
        l1 = F.relu(self.b1(self.conv1(xyz_e)), inplace=True)
        l2 = F.relu(self.b2(self.conv2(l1)), inplace=True)
        l3 = self.b3(self.conv3(l2))

        gl_ft = F.adaptive_max_pool2d(l3, (1, 1))
        gl_ft = F.adaptive_avg_pool2d(gl_ft, (h, w))

        return torch.cat([l3, gl_ft], dim=1)



class md2_pointnet(nn.Module):
    def __init__(self, in_channel=3, conv_channels=[128, 128, 256, 512, 1024], aggregate_layer=3):
        super(md2_pointnet, self).__init__()
        self.conv_channels = conv_channels
        self.aggregate_layer = aggregate_layer
        self.xyz_emb = nn.Conv2d(in_channel, conv_channels[0], kernel_size=1)
        self.xb = nn.BatchNorm2d(conv_channels[0])

        self.conv1 = nn.Conv2d(conv_channels[0]+3, conv_channels[1], kernel_size=1)
        self.conv2 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=1)
        self.conv3 = nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=1)
        self.conv4 = nn.Conv2d(conv_channels[3], conv_channels[4], kernel_size=1)
        self.b1 = nn.BatchNorm2d(conv_channels[1])
        self.b2 = nn.BatchNorm2d(conv_channels[2])
        self.b3 = nn.BatchNorm2d(conv_channels[3])
        self.b4 = nn.BatchNorm2d(conv_channels[4])

    def forward(self, input_ft, xyz):
        b, c, h, w = xyz.size()
        emb =F.relu(self.xb(self.xyz_emb(input_ft)), inplace=True)
        xyz_e = torch.cat([xyz, emb], dim=1)

        l1 = F.relu(self.b1(self.conv1(xyz_e)), inplace=True)
        l2 = F.relu(self.b2(self.conv2(l1)), inplace=True)
        l3 = F.relu(self.b3(self.conv3(l2)), inplace=True)
        l4 = self.b4(self.conv4(l3))

        gl_ft = F.adaptive_max_pool2d(l4, (1, 1))
        gl_ft = F.adaptive_avg_pool2d(gl_ft, (h, w))

        return torch.cat([l3, gl_ft], dim=1)