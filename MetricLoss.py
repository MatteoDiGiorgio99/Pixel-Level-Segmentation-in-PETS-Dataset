import torch
import torch.nn as nn

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

# Define a custom IoU Loss for training the model.
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
        
    '''Calculates L1 distance between model predictions and ground truth labels.
    
    Args:
        predictions: Predictions (B, 3, H, W)
        ground_truth: Ground Truth Labels (B, 1, H, W)
    
    Returns:
        float: The normalised L1 distance between predictions and ground truth labels.'''
    
    #predictions = nn.Softmax(dim=1)(predictions)
    ground_truth = torch.cat([(ground_truth == i) for i in range(3)], dim=1)
    batch_size, num_channels, height, width = predictions.shape
    
    # Calculate the sum of absolute differences between predictions and labels for each pixel
    l1_distance = torch.abs(predictions - ground_truth.to(torch.float)).sum()
    
    # Normalizes the distance by dividing by the total number of pixels
    total_pixels = batch_size * num_channels * height * width
    normalized_l1_distance = l1_distance / total_pixels
    
    return normalized_l1_distance


def prediction_accuracy(predicted_labels,ground_truth_labels):

    #predicted_labels = nn.Softmax(dim=1)(predicted_labels)
    predicted_labels = predicted_labels.argmax(dim=1)
    predicted_labels = predicted_labels.unsqueeze(1)

    eq = ground_truth_labels == predicted_labels
    return eq.sum().item() / predicted_labels.numel()