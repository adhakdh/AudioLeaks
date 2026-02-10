




import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



scenario_name = "tmp"
data_dir = f"{scenario_name}_mel_split"
batch_size = 32
num_epochs = 60
learning_rate = 1e-3
num_workers = 0
patience = 5
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    
    set_seed(seed)

    
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ]),
    }

    
    image_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(data_dir, split),
            transform=data_transforms[split],
        )
        for split in ["train", "val"]
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        for split in ["train", "val"]
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print("Loaded ResNet18 with new torchvision weights API")
    except ImportError:
        model = models.resnet18(pretrained=True)
        print("Loaded ResNet18 with legacy pretrained=True API")

    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    

    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    best_val_acc = 0.0
    no_improve_epochs = 0

    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                if phase == "train":
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc.item() > best_val_acc:
                    best_val_acc = epoch_acc.item()
                    no_improve_epochs = 0
                    torch.save(model.state_dict(),
                               "mel_classifier_resnet18_best.pth")
                    print(f"✔ Best model updated (Val Acc = {best_val_acc:.4f})")
                else:
                    no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("\nEarly stopping triggered.")
            break

    
    print("\nTraining finished.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
