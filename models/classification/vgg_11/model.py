import torch
from torch import nn

class VGG_11(nn.Module):
    
    def __init__(self, num_classes: int):
        """
        Модель VGG_11 

        Params: 
            num_classes - количество классов
        """
        super().__init__()
        
        self.layer_feature = nn.Sequential(
            # 224х224х3 -> 112x112х64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 112х112х65 -> 56x56х128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 56х56х128 -> 28x28х256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 28x28x256 -> 14x14x512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 14x14x512 -> 7x7x512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer_classificator = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.layer_feature(x)
        x = torch.flatten(x, 1)
        x = self.layer_classificator(x)
        return x
    
if __name__ == "__main__":
    model = VGG_11(num_class=2)

    x = torch.randn(2,3,224,224)
    output = model(x)

    print("Input size:", x.shape)
    print("Output shape:", output.shape)
