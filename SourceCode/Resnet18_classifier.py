import os
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch


def evaluate(model, dataloader, dataset_size, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    return epoch_loss, epoch_acc


def build_model(num_classes):
    try:
        model = models.resnet18(weights=None)
        print("Loaded ResNet18 with random initialization")
    except Exception:
        model = models.resnet18(pretrained=False)
        print("Loaded ResNet18 with random initialization (legacy API)")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model



class NpyMelDatasetSingleSplit(Dataset):
    def __init__(self, root_dir, out_size=(224, 224)):
        self.root_dir = Path(root_dir)
        self.out_size = out_size

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.class_names = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        if not self.class_names:
            raise ValueError(f"No class folders found in: {self.root_dir}")

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            npy_files = sorted(class_dir.glob("*.npy"))
            for f in npy_files:
                self.samples.append((f, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No .npy files found under: {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def _normalize_array(self, arr):
        mean = arr.mean()
        std = arr.std()
        if std < 1e-6:
            std = 1.0
        arr = (arr - mean) / std
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        arr = np.load(file_path)

        if arr.ndim != 2:
            raise ValueError(f"Invalid array shape {arr.shape} in {file_path}, expected a 2D array")

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        if not np.isfinite(arr).all():
            raise ValueError(f"NaN or Inf found in {file_path}")

        arr = self._normalize_array(arr)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=self.out_size,
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.squeeze(0)
        tensor = tensor.repeat(3, 1, 1)

        return tensor, label


def make_single_split_dataloaders(root_data_dir, batch_size, num_workers, device):
    train_dir = os.path.join(root_data_dir, "train")
    val_dir = os.path.join(root_data_dir, "val")

    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise RuntimeError(f"Val directory not found: {val_dir}")

    train_dataset = NpyMelDatasetSingleSplit(train_dir)
    val_dataset = NpyMelDatasetSingleSplit(val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return train_dataset, val_dataset, train_loader, val_loader



def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_training_header(
    checkpoint_root,
    device,
    split_modes_to_run,
    batch_size,
    num_epochs,
    learning_rate,
    weight_decay,
    num_workers,
):
    print("Checkpoint root:", os.path.abspath(checkpoint_root))
    print("Device:", device)
    print("Split modes to run:", split_modes_to_run)
    print("Epochs:", num_epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Weight decay:", weight_decay)
    print("Num workers:", num_workers)
    print("Normalize mode:", "zscore")
    print("Label smoothing:", 0.1)
    print("Seed:", 42)



def train_single_split_mode(
    split_mode,
    root_dir_map,
    checkpoint_root,
    batch_size,
    num_epochs,
    learning_rate,
    weight_decay,
    num_workers,
    device,
    patience=5,
):
    if split_mode not in root_dir_map:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    root_data_dir = root_dir_map[split_mode]
    if not os.path.isdir(root_data_dir):
        raise RuntimeError(f"Data directory not found for split_mode '{split_mode}': {root_data_dir}")

    save_dir = os.path.join(checkpoint_root, split_mode)
    os.makedirs(save_dir, exist_ok=True)

    best_save_path = os.path.join(save_dir, f"{split_mode}_best.pth")
    summary_path = os.path.join(save_dir, f"{split_mode}_summary.json")

    print("\n" + "#" * 90)
    print("Running split mode:", split_mode)
    print("Root data directory:", os.path.abspath(root_data_dir))
    print("Checkpoint path:", os.path.abspath(best_save_path))
    print("#" * 90)

    # 固定随机种子（lib 原函数内部就是 42）
    set_seed()

    train_dataset, val_dataset, train_loader, val_loader = make_single_split_dataloaders(
        root_data_dir=root_data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Normalize mode:", "zscore")
    print("Label smoothing:", 0.1)
    print("Seed:", 42)

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # =========================
        # train
        # =========================
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects / len(train_dataset)

        # =========================
        # val
        # =========================
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            dataset_size=len(val_dataset),
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.6g}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(current_lr),
        })

        # =========================
        # best model + early stop
        # =========================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve_epochs = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "scenario_name": split_mode,
                    "best_val_acc": float(best_val_acc),
                    "best_epoch": best_epoch,
                    "history": history,
                    "seed": 42,
                    "label_smoothing": 0.1,
                    "normalize_mode": "zscore",
                },
                best_save_path,
            )

            print(f"======= Best model updated (Val Acc = {best_val_acc:.4f})")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered. Best epoch: {best_epoch}, Best val acc: {best_val_acc:.4f}")
            break

    summary = {
        "split_mode": split_mode,
        "root_data_dir": os.path.abspath(root_data_dir),
        "checkpoint_path": os.path.abspath(best_save_path),
        "device": str(device),
        "num_epochs_requested": num_epochs,
        "num_epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "normalize_mode": "zscore",
        "label_smoothing": 0.1,
        "seed": 42,
        "early_stopping_patience": patience,
        "history": history,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_save_path}")
    print(f"Summary saved to: {summary_path}")

    return summary, summary_path



def save_combined_summary(checkpoint_root, all_mode_results):
    combined_summary_path = os.path.join(checkpoint_root, "all_selected_modes_summary.json")
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_mode_results, f, indent=2, ensure_ascii=False)
    return combined_summary_path



def print_training_footer_single_split(all_mode_results, combined_summary_path):
    print("\n" + "=" * 90)
    print("All requested split modes finished.")
    for mode in all_mode_results:
        print(
            "{} | best_val_acc {:.4f}".format(
                mode,
                all_mode_results[mode]["best_val_acc"],
            )
        )
    print("Combined summary saved to:", os.path.abspath(combined_summary_path))




split_modes_to_run = ["appidentity"]

batch_size = 32
num_epochs = 30
learning_rate = 1e-3
weight_decay = 1e-4
num_workers = 4
patience = 5

checkpoint_root = "single_split_checkpoints"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir_map = {mode: f"{mode}_mel_npy_split" for mode in split_modes_to_run}

    os.makedirs(checkpoint_root, exist_ok=True)

    print_training_header(
        checkpoint_root=checkpoint_root,
        device=device,
        split_modes_to_run=split_modes_to_run,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
    )

    all_mode_results = {}

    for split_mode in split_modes_to_run:
        summary, summary_path = train_single_split_mode(
            split_mode=split_mode,
            root_dir_map=root_dir_map,
            checkpoint_root=checkpoint_root,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_workers=num_workers,
            device=device,
            patience=patience,
        )

        print("\nCompleted split mode:", split_mode)
        print("Best val accuracy: {:.4f}".format(summary["best_val_acc"]))
        print("Summary saved to:", os.path.abspath(summary_path))

        all_mode_results[split_mode] = {
            "best_val_acc": summary["best_val_acc"],
            "summary_file": os.path.abspath(summary_path),
        }

    combined_summary_path = save_combined_summary(checkpoint_root, all_mode_results)
    print_training_footer_single_split(all_mode_results, combined_summary_path)


if __name__ == "__main__":
    main()