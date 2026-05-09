# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lib import (
    NpyMelDataset,
    build_model,
    load_checkpoint_safely,
    load_model_from_checkpoint,
    evaluate_topk_classification,
    save_single_test_eval_to_excel,
)

# =========================
# Config
# =========================
scenario_name = "appidentity"

data_dir = f"../../Datasets/{scenario_name}_split/test"
model_path = f"single_split_checkpoints/{scenario_name}/{scenario_name}_best.pth"

batch_size = 32
num_workers = 4
topk = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
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

    print("Classes:", class_names)
    print("Input mel shape:", getattr(dataset, "expected_shape", None))
    print("Samples:", len(dataset))
    print()

    model = load_model_from_checkpoint(
        model_path=model_path,
        dataset_class_names=class_names,
        build_model_fn=build_model,
        device=device,
    )

    results = evaluate_topk_classification(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        device=device,
        topk=topk,
    )


    print("===== Classification Report =====")
    print(results["classification_report_text"])
    print()

    print("===== Confusion Matrix =====")
    print(results["confusion_matrix_df"].values)
    print()


    print("===== Overall Accuracy =====")
    print(f"Top-1 Accuracy: {results['summary_metrics']['top1_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {results['summary_metrics']['top3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['summary_metrics']['top5_accuracy']:.4f}")
    print()


if __name__ == "__main__":
    main()