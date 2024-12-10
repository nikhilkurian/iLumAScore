import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import dataset_sampling2
from rce_loss import SSGCELoss
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`"
)

# Define fully connected layer
class FullyConnected(nn.Module):
    def __init__(self, model, num_ftrs, num_classes):
        super(FullyConnected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_3 = self.fc_4(x)
        return out_3

# Helper function for one-hot encoding
def one_hot(arr_, num_classes,device):
    """Convert labels to one-hot encoding."""
    b = arr_.size()[0]
    out_ = torch.zeros(b, num_classes).to(device)
    out_[range(b), arr_.cpu().numpy()] = 1
    return out_

# Main test function
def test_model(test_fold, device='cuda:0', batch_size=80, num_classes=2, num_workers=4, pos_class='luma', loss_criterion='rce'):
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    test_patients_txt = f'../data/Dataloader/{test_fold}_test.txt'
    
    # Load pre-trained DenseNet121 model
    model_loc = '../data/SavedModel/final'
    model = models.densenet121(pretrained=True)
    model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    num_ftrs = model.classifier.in_features
    model_final = FullyConnected(model.features, num_ftrs, num_classes)
    model_final = model_final.to(device)
    model_final = nn.DataParallel(model_final)
    model_final.load_state_dict(torch.load(model_loc))
    model = model_final
    sm = nn.Softmax(dim=1)

    # Select loss criterion
    if loss_criterion == 'rce':
        criterion = SSGCELoss()

    # Transformation for test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = dataset_sampling2.hetero_test(
        transforms=test_transform, 
        txt_path=test_patients_txt,
        pos_class=pos_class
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    total_nluma = 0
    total_luma = 0

    # Test the model
    with torch.no_grad():
        for images, labels, paths in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, one_hot(labels, num_classes,device))
            test_loss += loss.item() * images.size(0)
            _, out_labels = torch.max(preds, 1)
            counts = torch.bincount(out_labels, minlength=2)
            total_nluma += counts[0].item()
            total_luma += counts[1].item()
            test_acc += torch.sum(out_labels == labels)

    test_loss /= len(test_dataset)
    test_acc = test_acc.double() / len(test_dataset)
    iluma = total_luma/(total_luma+total_nluma)
    print(f'Non_LumA Count: {total_nluma}')
    print(f'LumA Count: {total_luma}')
    print(f'Calculated Heterogeneity Score for {test_fold} case from histology images, iLumA:{iluma} ')

# Main function for argument parsing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_folder", type=str, required=True, help="Path to the test patients text file")
    args = parser.parse_args()

    test_model(
        test_fold=args.test_folder
    )

if __name__ == "__main__":
    main()
