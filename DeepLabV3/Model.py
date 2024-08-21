import torch
from torch import nn
import torchvision.models as models

class DeepLabV3(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(DeepLabV3, self).__init__()
        # Load the DeepLabV3 model with a pretrained ResNet101 backbone
        self.deeplab = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)

        # Modify the classifier to output n_classes
        self.deeplab.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

        # Add a softmax activation function to the output
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.deeplab(x)['out']  # Extract the output from the 'out' key
        x = self.output_activation(x)
        return x
