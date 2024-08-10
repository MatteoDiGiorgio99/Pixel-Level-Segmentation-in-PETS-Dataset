import torch
from torch import nn
import os
import torchvision
import torchvision.transforms as T
from torchsummary import summary
from matplotlib import pyplot as plt
import torchmetrics as TM
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
import time



BACH_SIZE = 4
BACH_SIZE_TEST = 20
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
STEP_SIZE = 10
GAMMA = 0.1

###
DISABLE_SKIP = True
###

DIRECTORY = f"runs/unet_DisableSkip"
DIRECTORY_CHECKPOINTS = f"runs/unet_DisableSkip/checkpoints"
PATIENCE_EARLY_STOPPING=3
HISTORY_PATH = os.path.join(DIRECTORY, 'unet_history.pickle')
#METRICS_PATH = os.path.join(DIRECTORY, 'unet_metrics.pickle')

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

#TENSORBOARD
writer = SummaryWriter(DIRECTORY)

#create a directory
os.makedirs(DIRECTORY, exist_ok=True)
# Create a directory to save the model checkpoints.
os.makedirs(DIRECTORY_CHECKPOINTS, exist_ok=True)



def save_model(model, cp_name):
    torch.save(model.state_dict(), os.path.join(DIRECTORY, cp_name))

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while
# end def

# Oxford IIIT Pets Segmentation dataset loaded via torchvision.
train = os.path.join('oxford-iiit-pet', 'train')
test = os.path.join('oxford-iiit-pet', 'test')
train_dataset = torchvision.datasets.OxfordIIITPet(root=train, split="trainval", target_types="segmentation", download=True)
test_dataset = torchvision.datasets.OxfordIIITPet(root=test, split="test", target_types="segmentation", download=True)


#(train_pets_input, train_pets_target) = train_dataset[0]

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

class PetsDataset(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)
        
        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        # end if
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)
    
# Create a tensor for a segmentation trimap.
# Input: Float tensor with values in [0.0 .. 1.0]
# Output: Long tensor with values in {0, 1, 2}
def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs

transform_dict = args_to_dict(
    pre_transform=T.ToTensor(),
    pre_target_transform=T.ToTensor(),
    common_transform=T.Compose([
        ToDevice(get_device()),
        T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
        # Random Horizontal Flip as data augmentation.
        T.RandomHorizontalFlip(p=0.5),
    ]),
    post_transform=T.Compose([
        # Color Jitter as data augmentation.
        T.ColorJitter(contrast=0.3),
    ]),
    post_target_transform=T.Compose([
        T.Lambda(tensor_trimap),
    ]),
)

# Create the train and test instances of the data loader for the
# Oxford IIIT Pets dataset with random augmentations applied.

pets_train = PetsDataset(
    root=train,
    split="trainval",
    target_types="segmentation",
    download=False,
    **transform_dict,
)
pets_test = PetsDataset(
    root=test,
    split="test",
    target_types="segmentation",
    download=False,
    **transform_dict,
)

#reduce the size of the dataset for testing
#pets_train = torch.utils.data.Subset(pets_train, range(0, 100))
#pets_test = torch.utils.data.Subset(pets_test, range(0, 100))
print(f"Train dataset size: {len(pets_train)}")
print(f"Test dataset size: {len(pets_test)}")

pets_train_dataloader = torch.utils.data.DataLoader(
    pets_train,
    batch_size=BACH_SIZE,
    shuffle=True,
)
pets_test_dataloader = torch.utils.data.DataLoader(
    pets_test,
    batch_size=BACH_SIZE_TEST,
    shuffle=True,
)


(train_pets_inputs, train_pets_targets) = next(iter(pets_train_dataloader))
(test_pets_inputs, test_pets_targets) = next(iter(pets_test_dataloader))


# UNet Model
def conv_layer(input_channels, output_channels):
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU()
    )
    return conv

class UNet(nn.Module):
    def __init__(self, n_classes=3, disable_skip=DISABLE_SKIP):
        super(UNet, self).__init__()
        
        self.disable_skip = disable_skip
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        self.down_1 = conv_layer(3, 64)
        self.down_2 = conv_layer(64, 128)
        self.down_3 = conv_layer(128, 256)
        self.down_4 = conv_layer(256, 512)
        self.down_5 = conv_layer(512, 1024)
        
        # Decoder
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = conv_layer(1024 if not self.disable_skip else 512, 512)
        
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = conv_layer(512 if not self.disable_skip else 256, 256)
        
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = conv_layer(256 if not self.disable_skip else 128, 128)
        
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = conv_layer(128 if not self.disable_skip else 64, 64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, padding=0)
        self.output_activation = nn.Softmax(dim=1)
                
    def forward(self, img):
        x1 = self.down_1(img)  # 256x256
        x2 = self.max_pool(x1)  # 128x128
        x3 = self.down_2(x2)  # 64x64
        x4 = self.max_pool(x3)  # 32x32
        x5 = self.down_3(x4)  # 32x32
        x6 = self.max_pool(x5)  # 16x16
        x7 = self.down_4(x6)  # 16x16
        x8 = self.max_pool(x7)  # 8x8
        x9 = self.down_5(x8)  # 8x8
        
        x = self.up_1(x9)  # 16x16
        if not self.disable_skip:
            x = self.up_conv_1(torch.cat([x, x7], 1))  # 16x16
        else:
            x = self.up_conv_1(x)  # 16x16
        
        x = self.up_2(x)  # 32x32
        if not self.disable_skip:
            x = self.up_conv_2(torch.cat([x, x5], 1))  # 32x32
        else:
            x = self.up_conv_2(x)  # 32x32
        
        x = self.up_3(x)  # 64x64
        if not self.disable_skip:
            x = self.up_conv_3(torch.cat([x, x3], 1))  # 64x64
        else:
            x = self.up_conv_3(x)  # 64x64
        
        x = self.up_4(x)  # 128x128
        if not self.disable_skip:
            x = self.up_conv_4(torch.cat([x, x1], 1))  # 128x128
        else:
            x = self.up_conv_4(x)  # 128x128
        
        x = self.output(x)  # 256x256
        x = self.output_activation(x)
        
        return x

    

history = {'train_loss': [], 'test_loss': []}
#metrics= {'Accuracy': [], 'Iou': [], 'L1_distance': []}
# Create a summary of the model to see the architecture.
m = UNet(n_classes=3)
m.eval()
to_device(m)
summary(m,(3,256,256))   


# Define a custom IoU Metric for validating the model.
def IoUMetric(pred, gt, softmax=False):
    # Run softmax if input is logits.
    if softmax is True:
        pred = nn.Softmax(dim=1)(pred)

    
    # Add the one-hot encoded masks for all 3 output channels
    # (for all the classes) to a tensor named 'gt' (ground truth).
    gt = torch.cat([ (gt == i) for i in range(3) ], dim=1)

    intersection = gt * pred
    union = gt + pred - intersection

    # Compute the sum over all the dimensions except for the batch dimension.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
    
    # Compute the mean over the batch dimension.
    return iou.mean()

class IoULoss(nn.Module):
    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax
    
    # pred => Predictions (logits, B, 3, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        #return 1.0 - IoUMetric(pred, gt, self.softmax)
        # Compute the negative log loss for stable training.
        return -(IoUMetric(pred, gt, self.softmax).log())


def L1DistanceMetric(predictions, ground_truth):
    
    """Calcola la distanza L1 tra le previsioni del modello e le etichette di ground truth.
    
    Args:
        predictions: Predictions (B, 3, H, W)
        ground_truth: Ground Truth Labels (B, 1, H, W)
    
    Returns:
        float: La distanza L1 normalizzata tra le previsioni e le etichette di ground truth."""
    
    #predictions = nn.Softmax(dim=1)(predictions)
    ground_truth = torch.cat([(ground_truth == i) for i in range(3)], dim=1)
    batch_size, num_channels, height, width = predictions.shape
    
    # Calcola la somma delle differenze assolute tra le previsioni e le etichette per ogni pixel
    l1_distance = torch.abs(predictions - ground_truth.to(torch.float)).sum()
    
    # Normalizza la distanza dividendo per il numero totale di pixel
    total_pixels = batch_size * num_channels * height * width
    normalized_l1_distance = l1_distance / total_pixels
    
    return normalized_l1_distance




# Train the model for a single epoch
def train_model(epoch,model, loader, optimizer, criterion):
    
    to_device(model.train())
    running_loss = 0.0
    
    loop = tqdm(enumerate(loader, 0), total=len(loader), leave=False)
    for batch_idx, (inputs, targets) in loop:

        # Write the network graph at epoch 0, batch 0
        if epoch == 1 and batch_idx == 0:
            writer.add_graph(model, inputs)

        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()

    print("Train Loss: {:.4f}".format(running_loss / (batch_idx+1)))
    history['train_loss'].append(running_loss / (batch_idx+1))
    writer.add_scalar('Loss/train',running_loss / (batch_idx+1), epoch)


def prediction_accuracy(predicted_labels,ground_truth_labels):

    #predicted_labels = nn.Softmax(dim=1)(predicted_labels)
    predicted_labels = predicted_labels.argmax(dim=1)
    predicted_labels = predicted_labels.unsqueeze(1)

    eq = ground_truth_labels == predicted_labels
    return eq.sum().item() / predicted_labels.numel()

def evaluate(epoch,model,loader, test_loader, criterion):

    model.eval()
    test_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_l1_distance = 0.0


    with torch.no_grad():

        loop = tqdm(enumerate(test_loader, 0), total=len(loader), leave=False)
        for batch_idx,(data, target) in loop:
            data, target = to_device(data), to_device(target)
            output = model(data)
           
            loss = criterion(output, target)
            test_loss += loss.item()

            accuracy = prediction_accuracy(output, target)
            total_accuracy += accuracy

            custom_iou = IoUMetric(output, target)
            total_iou += custom_iou.item()

            L1_distance_acc = L1DistanceMetric(output, target)
            total_l1_distance += L1_distance_acc.item()


    num_batches = len(test_loader)

    print(f"Test Loss: {test_loss/num_batches:.4f}")
    print(f"Pixel Accuracy: {total_accuracy/num_batches:.4f}")
    print(f"IoU: {total_iou/num_batches:.4f}")
    print(f"L1 Distance: {total_l1_distance/num_batches:.4f}")
    print("---------------------------------")
    history['test_loss'].append(test_loss/num_batches)
    #metrics['Accuracy'].append(total_accuracy / num_batches)
    #metrics['Iou'].append(total_iou / num_batches)
    #metrics['L1_distance'].append(total_l1_distance / num_batches)
  

    writer.add_scalar('Loss/val', test_loss / num_batches, epoch)
    writer.add_scalar('Accuracy/val', total_accuracy / num_batches, epoch )
    writer.add_scalar('IoU/val', total_iou / num_batches, epoch )
    writer.add_scalar('L1_distance/val', total_l1_distance / num_batches, epoch )

    return test_loss / num_batches


    
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

  

save_path = os.path.join(DIRECTORY, "unet_training_progress_images")
os.makedirs(save_path, exist_ok=True)
print_test_dataset_masks(m, test_pets_inputs, test_pets_targets, epoch=0, save_path=None, show_plot=True)


# Optimizer and Learning Rate Scheduler.
to_device(m)
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
#scheduler=None
criterion = IoULoss()

def train_loop(model, loader,test_loader, test_data, optimizer, scheduler, save_path,checkpoint_path=None):
    test_inputs, test_targets = test_data
    best_loss = float('inf')
    start_epoch=1

    if checkpoint_path is not None:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")
        start_epoch += 1

    for i in range(start_epoch,EPOCHS+1):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(epoch,model, loader, optimizer,criterion)
        with torch.inference_mode():
            # Display the plt in the final training epoch.
            val_loss = evaluate(epoch,model,loader, test_loader, criterion)
            print_test_dataset_masks(model, test_inputs, test_targets, epoch=epoch, save_path=save_path, show_plot=(epoch == EPOCHS))

        # Check early stopping condition
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Epochs without improvement: {epochs_no_improve}")
            if epochs_no_improve == PATIENCE_EARLY_STOPPING:
                print("Early stopping!")
                break

        #save checkpoint every 5 epochs or last epoch
        #if epoch % 5 == 0 or epoch == EPOCHS:
        #    print(f"Saving checkpoint at epoch {epoch}")
        #    save_checkpoint(model, optimizer, epoch, os.path.join(DIRECTORY_CHECKPOINTS, f"checkpoint_{epoch}.pth"))

        if scheduler is not None:
            scheduler.step()

    #save history and metrics
    print("Saving history...")
    
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history, f,protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Saving metrics...")

    #with open(METRICS_PATH, 'wb') as f:
    #    pickle.dump(metrics, f,protocol=pickle.HIGHEST_PROTOCOL)
    
    writer.close()
    print("Training complete!")


save_path = os.path.join(DIRECTORY, "unet_training_progress_images")

#checkpoint_path = os.path.join(DIRECTORY_CHECKPOINTS, 'checkpoint_25.pth')
print("Starting training...")
train_loop(m, pets_train_dataloader,pets_test_dataloader,(test_pets_inputs, test_pets_targets), optimizer, scheduler, save_path,None)

#save the final model
save_model(m, f"unet_model_DisableSkip.pth")



