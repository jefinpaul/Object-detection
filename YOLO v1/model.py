import torch
import torch.nn as nn

# Define a CNN block with convolution, batch normalization, and Leaky ReLU
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x

# Define YOLOv1 model
class YOLOv1(nn.Module):
    def __init__(self, architecture_config):
        super(YOLOv1, self).__init__()
        self.layers = self._create_layers(architecture_config)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)  # Assuming input size of 448x448 and feature map size 7x7
        self.fc2 = nn.Linear(4096, 1470)  # 1470 = (7x7x30) where 30 is the number of predictions per grid cell (20 classes + 4 bounding box coordinates + 1 objectness score)

    def _create_layers(self, config):
        layers = []
        in_channels = 3  # Start with 3 input channels for RGB images

        for layer in config:
            if isinstance(layer, tuple):
                kernel_size, num_filters, stride, padding = layer
                layers.append(CNNBlock(in_channels, num_filters, kernel_size, stride, padding))
                in_channels = num_filters
            elif layer == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
            elif isinstance(layer, list):
                for sub_layer in layer:
                    kernel_size, num_filters, stride, padding = sub_layer
                    layers.append(CNNBlock(in_channels, num_filters, kernel_size, stride, padding))
                    in_channels = num_filters
            else:
                raise ValueError(f"Unsupported layer type: {layer}")
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Example usage:
# Define the architecture configuration for YOLOv1
architecture_config = [
    (7, 64, 2, 3),  # Conv layer: 7x7 kernel, 64 filters, stride 2, padding 3
    "M",            # MaxPool layer: 2x2 with stride 2

    (3, 192, 1, 1), # Conv layer: 3x3 kernel, 192 filters, stride 1, padding 1
    "M",            # MaxPool layer: 2x2 with stride 2

    (1, 128, 1, 0), # Conv layer: 1x1 kernel, 128 filters, stride 1, padding 0
    (3, 256, 1, 1), # Conv layer: 3x3 kernel, 256 filters, stride 1, padding 1
    (1, 256, 1, 0), # Conv layer: 1x1 kernel, 256 filters, stride 1, padding 0
    (3, 512, 1, 1), # Conv layer: 3x3 kernel, 512 filters, stride 1, padding 1
    "M",            # MaxPool layer: 2x2 with stride 2

    # Layer 11 - 16:
    [(1, 256, 1, 0),  # 1x1 Conv layer, 256 filters
     (3, 512, 1, 1)], # 3x3 Conv layer, 512 filters, repeated 4 times
    (1, 512, 1, 0),   # Conv layer: 1x1 kernel, 512 filters, stride 1, padding 0
    (3, 1024, 1, 1),  # Conv layer: 3x3 kernel, 1024 filters, stride 1, padding 1
    "M",              # MaxPool layer: 2x2 with stride 2

    # Layer 18 - 23:
    [(1, 512, 1, 0),  # 1x1 Conv layer, 512 filters
     (3, 1024, 1, 1)], # 3x3 Conv layer, 1024 filters, repeated 2 times
    (3, 1024, 1, 1),  # Conv layer: 3x3 kernel, 1024 filters, stride 1, padding 1
    (3, 1024, 2, 1),  # Conv layer: 3x3 kernel, 1024 filters, stride 2, padding 1

    (3, 1024, 1, 1),  # Conv layer: 3x3 kernel, 1024 filters, stride 1, padding 1
    (3, 1024, 1, 1),  # Conv layer: 3x3 kernel, 1024 filters, stride 1, padding 1
]

model = YOLOv1(architecture_config)