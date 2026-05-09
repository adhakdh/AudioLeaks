# -*- coding: utf-8 -*-
import os
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lib import (
    print_inference_header,
    build_time_to_decision_dataloader,
    load_segmentlevel_model_from_checkpoint,
    infer_all_segments_by_session,
    evaluate_time_to_decision_random_majority,
    summarize_time_to_decision_repeats,
    save_time_to_decision_results_to_excel,
)

# ========= Parameters ==========
scenario_name = "time_to_decision_majority_vote"

# Use the same test split as 5.2
data_dir = "../../Datasets/appidentity_split/test"

# Use the trained segment-level model from 5.2
model_path = "single_split_checkpoints/appidentity/appidentity_best.pth"

batch_size = 60
num_workers = 4

# Decision windows in seconds
decision_windows_sec = [2, 4, 6, 8, 10, 20, 30, 40, 60]

# Segment length used in 5.2
segment_len_sec = 2

# Repeat random window sampling
num_repeats = 50
random_seed = 2020

# Output Excel
out_xlsx = f"{scenario_name}_summary.xlsx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print_inference_header(
        device=device,
        data_dir=data_dir,
        model_path=model_path,
        out_xlsx=out_xlsx,
        decision_windows_sec=decision_windows_sec,
        num_repeats=num_repeats,
        random_seed=random_seed,
    )

    dataset, dataloader = build_time_to_decision_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    class_names = dataset.class_names
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Input mel shape:", dataset.expected_shape)
    print("Model input shape: (3, 224, 224)")
    print("Num segment files:", len(dataset))

    model = load_segmentlevel_model_from_checkpoint(
        model_path=model_path,
        num_classes=num_classes,
        class_names=class_names,
        device=device,
    )

    session_cache = infer_all_segments_by_session(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        device=device,
    )
    print("Num sessions found:", len(session_cache))

    detail_df, all_reports, all_cms = evaluate_time_to_decision_random_majority(
        session_cache=session_cache,
        class_names=class_names,
        decision_windows_sec=decision_windows_sec,
        num_repeats=num_repeats,
        segment_len_sec=segment_len_sec,
        random_seed=random_seed,
    )

    summary_df = summarize_time_to_decision_repeats(detail_df)

    print("\n" + "=" * 70)
    print("window_sec  top1_accuracy_mean  top1_accuracy_std  top3_accuracy_mean top3_accuracy_std macro_f1_mean  macro_f1_std")
    for _, row in summary_df.iterrows():
        print(
            f"{int(row['window_sec']):>3} "
            f"{row['top1_accuracy_mean']:.4f}  {row['top1_accuracy_std']:.4f} "
            f"{row['top3_accuracy_mean']:.4f}  {row['top3_accuracy_std']:.4f} "
            f"{row['macro_f1_mean']:.4f} {row['macro_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()