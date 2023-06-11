
import torch
import timm

class My_model(torch.nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.backbone = timm.create_model('hrnet_w18_small_v2', pretrained=True, features_only=True)
        final_layer_0 = [
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        ]
        final_layer_1 = [
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        ]
        final_layer_2 = [
            torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        ]
        final_layer_3 = [
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        ]
        headers = [
            torch.nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1, momentum=0.1),
            torch.nn.ReLU(inplace=True)
                   ]
        headers_size = [
            torch.nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1, momentum=0.1),
            torch.nn.ReLU(inplace=True)
                   ]
        # headers = [
        #     torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #     torch.nn.BatchNorm2d(1, momentum=0.1),
        #     torch.nn.ReLU(inplace=True)
        #            ]
        # headers_size = [
        #     torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #     torch.nn.BatchNorm2d(1, momentum=0.1),
        #     torch.nn.ReLU(inplace=True)
        #            ]
        self.header = torch.nn.Sequential(*headers)
        self.header_size = torch.nn.Sequential(*headers_size)
        self.final_layer_0 = torch.nn.Sequential(*final_layer_0)
        self.final_layer_1 = torch.nn.Sequential(*final_layer_1)
        self.final_layer_2 = torch.nn.Sequential(*final_layer_2)
        self.final_layer_3 = torch.nn.Sequential(*final_layer_3)

    def forward(self, x):
        x = self.backbone(x)
        y1 = self.final_layer_0(x[0])
        y2 = self.final_layer_1(x[1])
        # y = y1 + y2
        y3 = self.final_layer_2(x[2])
        y4 = self.final_layer_3(x[3])
        y = torch.cat((y1, y2, y3, y4), dim=1)
        y_location = self.header(y)
        y_size = self.header_size(y)
        return y_location, y_size