
import torch
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from Config import *

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

#TENSORBOARD
writer = SummaryWriter(DIRECTORY)

## Arrays to store history and metrics
history = {'train_loss': [], 'test_loss': []}
#metrics= {'Accuracy': [], 'Iou': [], 'L1_distance': []}

## Save model
def save_model(model, cp_name):
    torch.save(model.state_dict(), os.path.join(DIRECTORY, cp_name))

## Save Checkpoint model
def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

## Load Checkpoint model
def load_checkpoint(model, optimizer, filepath):
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

## move to device for training
def to_device(x):
    return x.cuda() if torch.cuda.is_available() else x.cpu()

## get device for training
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a tensor for a segmentation trimap.
def tensor_trimap(t):
    # Input: Float tensor with values in [0.0 .. 1.0]
    # Output: Long tensor with values in {0, 1, 2}
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs

class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
    


def close_figures():
    import matplotlib.pyplot as plt
    while plt.get_fignums():
        plt.close(plt.gcf())
    

## Print the test dataset masks 
def print_test_dataset_masks(model, test_pets_targets, test_pets_labels, epoch, save_path, show_plot):
    
    to_device(model.eval())
    predictions = model(to_device(test_pets_targets))
    test_pets_labels = to_device(test_pets_labels)

    #pred = nn.Softmax(dim=1)(predictions)
    pred_labels = predictions.argmax(dim=1)
    # Add a value 1 dimension at dim=1
    pred_labels = pred_labels.unsqueeze(1)
    pred_mask = pred_labels.to(torch.float)

    title = f"Epoch: {epoch:02d}"

    # Close all previously open figures.
    close_figures()
    
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_targets, nrow=5)))
    plt.axis('off')
    plt.title("Original Image")

    fig.add_subplot(3, 1, 2)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_labels.float() / 2.0, nrow=5)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(t2img(torchvision.utils.make_grid(pred_mask / 2.0, nrow=5)))
    plt.axis('off')
    plt.title("Predicted Mask")
    

    #save image training progress
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"epoch_{epoch:02}.png"), format="png", bbox_inches="tight", pad_inches=0.4)
    
    if show_plot is False:
        close_figures()
    else:
        plt.show()

