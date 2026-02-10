import os
import re
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from openpyxl.utils import get_column_letter
import pandas as pd
import torch.nn.functional as F


scenario_name = "tmp"
data_dir = f"{scenario_name}_mel_split/val"
model_path = "classifier_resnet18.pth"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


topk = 5

def load_model(num_classes):
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def sanitize_sheet_name(name):
    """
    Excel sheet name constraints:
    - Max length 31
    - Cannot contain: : \ / ? * [ ]
    """
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    name = name.strip()
    if not name:
        name = "Sheet"
    return name[:31]


def main():
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    class_names = dataset.classes
    num_classes = len(class_names)
    print("cat:", class_names)

    
    sample_paths = [p for (p, _) in dataset.samples]
    sample_true_idxs = [y for (_, y) in dataset.samples]

    
    model = load_model(num_classes)

    
    all_preds = []
    all_labels = []

    
    grouped_rows = {name: [] for name in class_names}

    
    per_label_stats = {
        name: {"total": 0, "top1": 0, "top3": 0, "top5": 0}
        for name in class_names
    }

    global_idx = 0
    k_eff = min(topk, num_classes)

    with torch.no_grad():
        for inputs, labels in dataloader:
            bsz = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)               
            probs = F.softmax(outputs, dim=1)     
            _, top_idxs = torch.topk(probs, k=k_eff, dim=1)  

            
            preds = top_idxs[:, 0]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            
            batch_paths = sample_paths[global_idx:global_idx + bsz]
            batch_true_idxs = sample_true_idxs[global_idx:global_idx + bsz]

            for i in range(bsz):
                fname = os.path.basename(batch_paths[i])

                true_idx = int(batch_true_idxs[i])
                true_name = class_names[true_idx]

                
                top_pred_list = top_idxs[i].cpu().tolist()

                
                per_label_stats[true_name]["total"] += 1
                if true_idx == top_pred_list[0]:
                    per_label_stats[true_name]["top1"] += 1
                if true_idx in top_pred_list[:min(3, len(top_pred_list))]:
                    per_label_stats[true_name]["top3"] += 1
                if true_idx in top_pred_list[:min(5, len(top_pred_list))]:
                    per_label_stats[true_name]["top5"] += 1

                
                correct = 1 if top_pred_list[0] == true_idx else 0
                row = {
                    "file_name": fname,
                    "correct": correct,
                }

                
                for k_i in range(k_eff):
                    cls_idx = int(top_pred_list[k_i])
                    cls_name = class_names[cls_idx]
                    row[f"top{k_i+1}_class"] = cls_name

                grouped_rows[true_name].append(row)

            global_idx += bsz

    
    print("\n===== report =====")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    
    print("\n===== matrix =====")
    print(confusion_matrix(all_labels, all_preds))

    
    print("\n===== Per-label Top-k Accuracy (Recall@k) =====")
    print(f"{'Label':<25} {'Top1':>10} {'Top3':>10} {'Top5':>10} {'Total':>10}")
    print("-" * 70)

    top1_list, top3_list, top5_list = [], [], []
    total_all = 0
    hit1_all = 0
    hit3_all = 0
    hit5_all = 0

    for tmp_i, label in enumerate(class_names):
        s = per_label_stats[label]
        total = s["total"]
        if total == 0:
            
            continue

        top1 = s["top1"] / total
        top3 = s["top3"] / total
        top5 = s["top5"] / total

        print(f"{tmp_i+1:<2} {label:<20} {top1:>10.4f} {top3:>10.4f} {top5:>10.4f} {total:>10d}")

        
        top1_list.append(top1)
        top3_list.append(top3)
        top5_list.append(top5)

        
        total_all += total
        hit1_all += s["top1"]
        hit3_all += s["top3"]
        hit5_all += s["top5"]

    if top1_list:
        macro_top1 = sum(top1_list) / len(top1_list)
        macro_top3 = sum(top3_list) / len(top3_list)
        macro_top5 = sum(top5_list) / len(top5_list)

        micro_top1 = hit1_all / total_all if total_all else 0.0
        micro_top3 = hit3_all / total_all if total_all else 0.0
        micro_top5 = hit5_all / total_all if total_all else 0.0

        print("-" * 70)
        print(f"{'MACRO-AVG':<25} {macro_top1:>10.4f} {macro_top3:>10.4f} {macro_top5:>10.4f} {'-':>10}")  
        print(f"{'MICRO-AVG':<25} {micro_top1:>10.4f} {micro_top3:>10.4f} {micro_top5:>10.4f} {total_all:>10d}")   


if __name__ == "__main__":
    main()
