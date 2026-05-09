# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lib import (
    build_model,
    load_checkpoint_safely,
)


# ===============================
# Open-set evaluation script
# 6-known / 2-unknown
# ===============================

# ========= Parameters ==========
checkpoint_dir = "openset_6known_2unknown_checkpoints"
data_dir = "../../Datasets//appidentity_split/test"

evaluate_mode = "all_existing"

checkpoint_path = ""   # only used when evaluate_mode == "single"

batch_size = 32
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= Dataset ==========
class NpyMelDatasetOpenSet(Dataset):
    def __init__(self, root_dir, known_classes, unknown_classes, out_size=(224, 224)):
        self.root_dir = Path(root_dir)
        self.out_size = out_size

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.known_classes = list(known_classes)
        self.unknown_classes = list(unknown_classes)
        self.all_eval_classes = self.known_classes + self.unknown_classes

        for c in self.all_eval_classes:
            if not (self.root_dir / c).exists():
                raise FileNotFoundError(f"Class folder not found: {self.root_dir / c}")

        self.class_to_idx_known = {name: idx for idx, name in enumerate(self.known_classes)}
        self.samples = []

        for class_name in self.known_classes:
            class_dir = self.root_dir / class_name
            npy_files = sorted(class_dir.glob("*.npy"))
            for f in npy_files:
                self.samples.append((f, 1, self.class_to_idx_known[class_name], class_name))

        for class_name in self.unknown_classes:
            class_dir = self.root_dir / class_name
            npy_files = sorted(class_dir.glob("*.npy"))
            for f in npy_files:
                self.samples.append((f, 0, -1, class_name))

        if not self.samples:
            raise ValueError("No evaluation samples found.")

        first_arr = np.load(self.samples[0][0])
        if first_arr.ndim != 2:
            raise ValueError(
                f"Expected a 2D mel array, but got shape {first_arr.shape} in {self.samples[0][0]}"
            )
        self.expected_shape = first_arr.shape

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
        file_path, is_known, closed_set_label, class_name = self.samples[idx]
        arr = np.load(file_path)

        if arr.ndim != 2:
            raise ValueError(f"Invalid array shape {arr.shape} in {file_path}, expected 2D")

        if arr.shape != self.expected_shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: got {arr.shape}, expected {self.expected_shape}"
            )

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

        return tensor, is_known, closed_set_label, class_name, str(file_path)


def compute_fpr95(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    valid = np.where(tpr >= 0.95)[0]
    if len(valid) == 0:
        return np.nan
    return float(np.min(fpr[valid]))


def evaluate_one_checkpoint(ckpt_path):
    checkpoint = load_checkpoint_safely(ckpt_path, device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint format not supported: {ckpt_path}")

    known_classes = checkpoint.get("known_classes", None)
    unknown_classes = checkpoint.get("unknown_classes", None)
    class_names = checkpoint.get("class_names", None)

    if known_classes is None or unknown_classes is None or class_names is None:
        raise ValueError(
            f"Checkpoint missing required metadata (known_classes / unknown_classes / class_names): {ckpt_path}"
        )

    if list(known_classes) != list(class_names):
        raise ValueError(
            f"Checkpoint inconsistency: known_classes != class_names\n"
            f"known_classes={known_classes}\n"
            f"class_names={class_names}"
        )

    dataset = NpyMelDatasetOpenSet(
        root_dir=data_dir,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        out_size=(224, 224),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    y_known_binary = []   # 1 = known, 0 = unknown
    energy_scores = []    # logsumexp(logits)

    with torch.no_grad():
        for inputs, is_known, _, _, _ in dataloader:
            inputs = inputs.to(device, non_blocking=True)

            outputs = model(inputs)
            batch_energy = torch.logsumexp(outputs, dim=1)

            batch_energy = batch_energy.cpu().numpy()
            is_known = np.array(is_known)

            for i in range(len(batch_energy)):
                y_known_binary.append(int(is_known[i]))
                energy_scores.append(float(batch_energy[i]))

    y_known_binary = np.array(y_known_binary)
    energy_scores = np.array(energy_scores)

    n_known = int((y_known_binary == 1).sum())
    n_unknown = int((y_known_binary == 0).sum())

    if n_known == 0 or n_unknown == 0:
        raise ValueError(
            f"Need both known and unknown samples for evaluation, got n_known={n_known}, n_unknown={n_unknown}"
        )

    auroc_energy = float(roc_auc_score(y_known_binary, energy_scores))
    fpr95_energy = compute_fpr95(y_known_binary, energy_scores)

    result = {
        "checkpoint_name": Path(ckpt_path).name,
        "split_id": checkpoint.get("split_id", None),
        "unknown_classes": ", ".join(unknown_classes),
        "num_known_samples": n_known,
        "num_unknown_samples": n_unknown,
        "auroc_energy": auroc_energy,
        "fpr95_energy": fpr95_energy,
        "checkpoint_path": os.path.abspath(ckpt_path),
    }

    return result


def get_checkpoint_list():
    if evaluate_mode == "single":
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be set when evaluate_mode == 'single'")
        return [checkpoint_path]

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.pth"))
    if not ckpts:
        raise ValueError(f"No .pth files found in {ckpt_dir}")
    return [str(p) for p in ckpts]


def main():
    print("Device:", device)
    print("Test directory:", os.path.abspath(data_dir))
    print("Evaluate mode:", evaluate_mode)

    ckpt_list = get_checkpoint_list()
    print("Checkpoints to evaluate:", len(ckpt_list))

    all_results = []
    failed_results = []

    for ckpt in ckpt_list:
        try:
            result = evaluate_one_checkpoint(ckpt)
            all_results.append(result)

        except Exception as e:
            print(f"FAILED on checkpoint: {ckpt}")
            print("Error:", repr(e))
            failed_results.append({
                "checkpoint_name": Path(ckpt).name,
                "checkpoint_path": os.path.abspath(ckpt),
                "error": repr(e),
            })


    print("\n" + "=" * 90)
    print("Finished.")
    print("Successful evaluations:", len(all_results))
    print("Failed evaluations    :", len(failed_results))

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(by=["split_id", "checkpoint_name"], na_position="last")
        print("\n===== Current Summary =====")
        print(df[["split_id", "checkpoint_name", "unknown_classes", "auroc_energy", "fpr95_energy"]].to_string(index=False))

        print("\n===== Mean over evaluated checkpoints =====")
        mean_auroc = df["auroc_energy"].mean()
        std_auroc = df["auroc_energy"].std()

        print(f"Mean AUROC (Energy)  : {mean_auroc:.4f}")
        print(f"Std  AUROC (Energy)  : {std_auroc:.4f}")
        print(f"Energy AUROC         : {mean_auroc:.4f} ± {std_auroc:.4f}")


if __name__ == "__main__":
    main()