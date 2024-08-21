import torch
import os
import torchvision
import torchvision.transforms as T
from torchsummary import summary
import pickle
import torchvision.models.segmentation as models


from Config import *
from Utils import *
from Dataset import *
from Model import *
from MetricLoss import *
from Train import *


#create a directory
os.makedirs(DIRECTORY, exist_ok=True)
# Create a directory to save the model checkpoints.
os.makedirs(DIRECTORY_CHECKPOINTS, exist_ok=True)

# Oxford IIIT Pets Segmentation dataset loaded via torchvision.
train = os.path.join('oxford-iiit-pet', 'train')
test = os.path.join('oxford-iiit-pet', 'test')
train_dataset = torchvision.datasets.OxfordIIITPet(root=train, split="trainval", target_types="segmentation", download=True)
test_dataset = torchvision.datasets.OxfordIIITPet(root=test, split="test", target_types="segmentation", download=True)

#(train_pets_input, train_pets_target) = train_dataset[0]

### DATA AUGMENTATION ###
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

## DATALOADER ##
# Create the data loaders for the Oxford IIIT Pets dataset.
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

# Create a summary of the model to see the architecture.
#m = UNet(n_classes=3)
#m.eval()
#to_device(m)
#summary(m,(3,256,256))   


m = DeepLabV3(n_classes=3, pretrained=True) # type: ignore
m.eval()
to_device(m)


## Display the first batch of the training dataset.
save_path = os.path.join(DIRECTORY, "deeplabv3_training_progress_images")
os.makedirs(save_path, exist_ok=True)
print_test_dataset_masks(m, test_pets_inputs, test_pets_targets, epoch=0, save_path=None, show_plot=True)


# Optimizer and Learning Rate Scheduler.
to_device(m)
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
#scheduler=None
criterion = IoULoss() ## Loss function


def train_loop(model, loader,test_loader, test_data, optimizer, scheduler, save_path,checkpoint_path=None):
    test_inputs, test_targets = test_data
    best_loss = float('inf')
    start_epoch=1
    epochs_no_improve = 0

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
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Saving checkpoint at epoch {epoch}")
            save_checkpoint(model, optimizer, epoch, os.path.join(DIRECTORY_CHECKPOINTS, f"checkpoint_{epoch}.pth"))

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

save_path = os.path.join(DIRECTORY, "deeplabv3_training_progress_images")

#checkpoint_path = os.path.join(DIRECTORY_CHECKPOINTS, 'checkpoint_25.pth')
print("Starting training...")
train_loop(m, pets_train_dataloader,pets_test_dataloader,(test_pets_inputs, test_pets_targets), optimizer, scheduler, save_path,None)

#save the final model
save_model(m, f"DeepLabV3_model.pth")



