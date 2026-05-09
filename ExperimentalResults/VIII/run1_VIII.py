# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib import (
    NpyMelDataset,
    build_model,
    build_root_dir_map,
    get_fold_names,
    get_checkpoint_path,
    load_model_for_fold,
    evaluate_classification_metrics,
    summarize_mode_results,
    save_eval_results_to_excel,
)

# =========================
# Config
# =========================
split_modes_to_run = ["gamelevel","vrchatroom","gamebehavior",  "keyboard",  "smallroom"]

root_dir_map = build_root_dir_map(split_modes_to_run)

checkpoint_root = "cross_protocol_checkpoints"
output_xlsx = os.path.join(checkpoint_root, "selected_modes_test_metrics.xlsx")

batch_size = 32
num_workers = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    all_rows = []
    all_summary_rows = []

    print("Device:", device)
    print("Split modes to evaluate:", split_modes_to_run)
    print("Checkpoint root:", os.path.abspath(checkpoint_root))
    print("Output xlsx:", os.path.abspath(output_xlsx))
    print("Normalize mode:", "zscore")
    print("Model input size:", (3, 224, 224))
    print()

    for split_mode in split_modes_to_run:
        data_root = root_dir_map[split_mode]
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Data directory not found for split_mode={split_mode}: {data_root}")

        fold_names = get_fold_names(data_root)
        if not fold_names:
            raise RuntimeError(f"No fold directories found under: {data_root}")

        mode_rows = []

        for fold_name in fold_names:
            print(f"Evaluating {split_mode} / {fold_name} ...")

            test_dir = os.path.join(data_root, fold_name, "test")
            if not os.path.isdir(test_dir):
                raise FileNotFoundError(f"Test directory not found: {test_dir}")

            dataset = NpyMelDataset(
                test_dir,
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
            model_path = get_checkpoint_path(checkpoint_root, split_mode, fold_name)
            model = load_model_for_fold(
                model_path=model_path,
                dataset_class_names=class_names,
                build_model_fn=build_model,
                device=device,
            )

            metrics = evaluate_classification_metrics(model, dataloader, device)

            row = {
                "split_mode": split_mode,
                "fold_name": fold_name,
                "test_samples": len(dataset),
                "num_classes": len(class_names),
                "model_input_shape": str((3, 224, 224)),
                "normalize_mode": "zscore",
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
                "checkpoint_path": os.path.abspath(model_path),
            }

            mode_rows.append(row)
            all_rows.append(row)

        df_mode = pd.DataFrame(mode_rows)
        df_mode = df_mode.sort_values(
            by="fold_name",
            key=lambda s: s.map(
                lambda x: int(str(x)[4:]) if str(x).startswith("fold") and str(x)[4:].isdigit() else 999999
            )
        ).reset_index(drop=True)

        summary_row = summarize_mode_results(df_mode)
        all_summary_rows.append(summary_row)

    df_all = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(all_summary_rows)

    df_all = df_all[
        [
            "split_mode",
            "fold_name",
            "test_samples",
            "num_classes",
            "model_input_shape",
            "normalize_mode",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "checkpoint_path",
        ]
    ]

    df_summary = df_summary[
        [
            "split_mode",
            "num_folds",
            "accuracy_mean",
            "accuracy_std",
            "precision_macro_mean",
            "precision_macro_std",
            "recall_macro_mean",
            "recall_macro_std",
            "f1_macro_mean",
            "f1_macro_std",
            "precision_weighted_mean",
            "precision_weighted_std",
            "recall_weighted_mean",
            "recall_weighted_std",
            "f1_weighted_mean",
            "f1_weighted_std",
        ]
    ]

    print("=" * 100)
    print("Combined summary")
    print("=" * 100)
    print(
        df_summary[["split_mode", "num_folds", "accuracy_mean", "accuracy_std","f1_macro_mean","f1_macro_std"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("=" * 100)
    print("XLSX saved to:", os.path.abspath(output_xlsx))

if __name__ == "__main__":
    main()