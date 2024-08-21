
import torch
from tqdm import tqdm
from Utils import *
from MetricLoss import IoUMetric, L1DistanceMetric , prediction_accuracy


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

    # Write the loss to the tensorboard
    writer.add_scalar('Loss/train',running_loss / (batch_idx+1), epoch)

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
  
    # Write the metrics to the tensorboard
    writer.add_scalar('Loss/val', test_loss / num_batches, epoch)
    writer.add_scalar('Accuracy/val', total_accuracy / num_batches, epoch )
    writer.add_scalar('IoU/val', total_iou / num_batches, epoch )
    writer.add_scalar('L1_distance/val', total_l1_distance / num_batches, epoch )

    return test_loss / num_batches