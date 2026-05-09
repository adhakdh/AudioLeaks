# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lib import (
    NpyMelDataset,
    build_model,
    load_model_from_checkpoint,
)

# =========================
# Config
# =========================
input_dir = "robust_factors"
model_path = "single_split_checkpoints/appidentity/appidentity_best.pth"

batch_size = 32
num_workers = 0
topk = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_per_folder_report = False   #  True


SCENARIO_CONFIG = {
    "volume": ["50", "75", "100"],
    "movement": ["head", "body"],
    "noise": ["ac", "voice"],
}


def evaluate_one_scenario(data_dir, model, topk=5):
    dataset = NpyMelDataset(
        data_dir,
        out_size=(224, 224),
        normalize_mode="zscore",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    class_names = dataset.class_names
    num_classes = len(class_names)

    model.eval()

    total_samples = 0
    total_top1_correct = 0
    total_top3_correct = 0

    per_class_total = [0 for _ in range(num_classes)]
    per_class_top1_correct = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    raise ValueError("error")
            else:
                raise ValueError("error")

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)

            # Top-1
            pred_top1 = logits.argmax(dim=1)
            top1_correct_mask = pred_top1.eq(targets)

            # Top-3
            k3 = min(3, logits.shape[1])
            top3_pred = logits.topk(k3, dim=1).indices
            top3_correct_mask = top3_pred.eq(targets.unsqueeze(1)).any(dim=1)

            batch_size_now = targets.size(0)
            total_samples += batch_size_now
            total_top1_correct += top1_correct_mask.sum().item()
            total_top3_correct += top3_correct_mask.sum().item()

            for class_idx in range(num_classes):
                class_mask = targets.eq(class_idx)
                class_count = class_mask.sum().item()
                if class_count > 0:
                    per_class_total[class_idx] += class_count
                    per_class_top1_correct[class_idx] += (
                        top1_correct_mask[class_mask].sum().item()
                    )

    if total_samples == 0:
        raise ValueError(f"{data_dir}")

    top1_accuracy = total_top1_correct / total_samples
    top3_accuracy = total_top3_correct / total_samples

    per_class_accuracy = {}
    for class_idx, class_name in enumerate(class_names):
        total_i = per_class_total[class_idx]
        correct_i = per_class_top1_correct[class_idx]
        acc_i = (correct_i / total_i) if total_i > 0 else 0.0
        per_class_accuracy[class_name] = {
            "correct": correct_i,
            "total": total_i,
            "accuracy": acc_i,
        }

    results = {
        "summary_metrics": {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "num_samples": total_samples,
        },
        "per_class_accuracy": per_class_accuracy,
    }

    return results, class_names


def print_classification_report(results, scenario_label):
    print(f"\n===== Classification Report: {scenario_label} =====")

    summary = results["summary_metrics"]
    print(f"num_samples: {summary['num_samples']}")
    print(f"top1_accuracy: {summary['top1_accuracy']:.4f}")
    print(f"top3_accuracy: {summary['top3_accuracy']:.4f}")

    print("\nPer-class Top-1 Accuracy:")
    print("class_name, correct, total, accuracy")
    for class_name, stats in results["per_class_accuracy"].items():
        print(
            f"{class_name}, "
            f"{stats['correct']}, "
            f"{stats['total']}, "
            f"{stats['accuracy']:.4f}"
        )


def main():
    first_type = next(iter(SCENARIO_CONFIG))
    first_sub = SCENARIO_CONFIG[first_type][0]

    first_data_dir = f"{input_dir}/{first_type}/{first_sub}_mel_npy"
    first_dataset = NpyMelDataset(
        first_data_dir,
        out_size=(224, 224),
        normalize_mode="zscore",
    )

    model = load_model_from_checkpoint(
        model_path=model_path,
        dataset_class_names=first_dataset.class_names,
        build_model_fn=build_model,
        device=device,
    )

    print("===== Scenarios To Evaluate =====")
    for scenario_type, names in SCENARIO_CONFIG.items():
        print(f"{scenario_type}: {names}")
    print()

    print("===== Overall Accuracy =====")
    print("scenario_type/sub Top-1 Acc. Top-3 Acc.")

    for scenario_type, scenario_name_list in SCENARIO_CONFIG.items():
        for scenario_name in scenario_name_list:
            data_dir = f"{input_dir}/{scenario_type}/{scenario_name}_mel_npy"
            scenario_label = f"{scenario_type} {scenario_name}"

            results, _ = evaluate_one_scenario(data_dir, model, topk=topk)

            top1 = results["summary_metrics"]["top1_accuracy"]
            top3 = results["summary_metrics"]["top3_accuracy"]

            print(f"{scenario_label} {top1:.4f} {top3:.4f}")

            if print_per_folder_report:
                print_classification_report(results, scenario_label)


if __name__ == "__main__":
    main()