# -*- coding: utf-8 -*-
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib import NpyMelDataset
from model import build_torch_model

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False



scenario_name = "appidentity"
data_dir =  "../../Datasets/appidentity_split"
model_dir = "trained_models"
output_xlsx = "model_test_results.xlsx"
output_csv = "model_test_results.csv"

selected_models = [ "svm", "simplecnn", "mobilenetv2", "crnn"]  

batch_size = 32
num_workers = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compute_metrics_from_probs(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)

    top1 = accuracy_score(y_true, y_pred)

    k = min(3, y_prob.shape[1])
    topk_indices = np.argsort(-y_prob, axis=1)[:, :k]
    top3 = np.mean([y_true[i] in topk_indices[i] for i in range(len(y_true))])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "Top-1 Acc": top1,
        "Top-3 Acc": top3,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }



def dataset_to_numpy(dataset):
    xs = []
    ys = []

    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x.numpy().reshape(-1))
        ys.append(y)

    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def evaluate_svm(test_dataset):
    model_path = os.path.join(model_dir, f"{scenario_name}_mel_svm_best_npy.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SVM model file not found: {model_path}")

    if HAS_JOBLIB:
        obj = joblib.load(model_path)
    else:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)

    pipeline = obj["pipeline"]
    class_names = obj["class_names"]
    model_input_shape = tuple(obj.get("model_input_shape", (3, 224, 224)))

    if class_names != test_dataset.class_names:
        raise ValueError(
            f"SVM class names mismatch:\n"
            f"model={class_names}\n"
            f"test={test_dataset.class_names}"
        )

    if model_input_shape != (3, 224, 224):
        raise ValueError(
            f"SVM model_input_shape mismatch:\n"
            f"model={model_input_shape}\n"
            f"expected={(3, 224, 224)}"
        )

    X_test, y_test = dataset_to_numpy(test_dataset)

    y_prob = pipeline.predict_proba(X_test)
    metrics = compute_metrics_from_probs(y_test, y_prob)
    return metrics


def evaluate_torch_model(model_name, test_dataset):
    model_path = os.path.join(model_dir, f"{scenario_name}_mel_{model_name}_best_npy.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    model_input_shape = tuple(checkpoint.get("model_input_shape", (3, 224, 224)))

    if class_names != test_dataset.class_names:
        raise ValueError(
            f"{model_name} class names mismatch:\n"
            f"model={class_names}\n"
            f"test={test_dataset.class_names}"
        )

    if model_input_shape != (3, 224, 224):
        raise ValueError(
            f"{model_name} model_input_shape mismatch:\n"
            f"model={model_input_shape}\n"
            f"expected={(3, 224, 224)}"
        )

    model = build_torch_model(model_name, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics_from_probs(y_true, y_prob)
    return metrics


def main():

    test_dataset = NpyMelDataset(
        os.path.join(data_dir, "test"),
        out_size=(224, 224),
        normalize_mode="zscore",
    )

    results = []

    for model_name in selected_models:
        model_name = model_name.lower()
        if model_name == "svm":
            metrics = evaluate_svm(test_dataset)
        elif model_name in ["simplecnn", "mobilenetv2", "crnn"]:
            metrics = evaluate_torch_model(model_name, test_dataset)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        row = {"Model": model_name}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("Final results")
    print("=" * 70)
    print(df)


if __name__ == "__main__":
    main()