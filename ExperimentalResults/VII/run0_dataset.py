# -*- coding: utf-8 -*-
import os
import re
import json
import shutil

# =========================
# Config
# =========================
split_modes_to_run = [ "htc", "quest3", "pico","quest2"]  # "htc", "quest3", "pico"

overwrite = True
save_mode = "hardlink"              # "symlink" / "copy" / "hardlink"

# 现在文件名格式:
# <class_name>__<session_id>_segXXXXXX.npy
# 例如:
# htc_animal__1_seg000000.npy
filename_pattern = re.compile(r"^(.+?)__(\d+)_seg(\d{6})\.npy$")

save_dir = "cross_fold"
os.makedirs(save_dir, exist_ok=True)
# =========================


def list_classes(root_dir):
    
    classes = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    return sorted(classes)


def list_npy_files(class_dir):
    
    files = [
        f for f in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(".npy")
    ]
    return sorted(files)


def parse_filename(filename):
    m = filename_pattern.match(filename)
    if m is None:
        raise ValueError("Invalid filename format: {}".format(filename))

    class_name = m.group(1)
    session_id = int(m.group(2))
    seg_id = int(m.group(3))

    return class_name, session_id, seg_id


def collect_all_sessions(input_dir, classes):
    
    sessions = set()

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        files = list_npy_files(cls_dir)

        for f in files:
            parsed_class, session_id, seg_id = parse_filename(f)

            if parsed_class != cls:
                raise ValueError(
                    "Class name mismatch between directory and filename:\n"
                    "directory={}\nfilename={}".format(cls, f)
                )

            sessions.add(session_id)

    return sorted(sessions)


def get_fold_definitions(session_list):
    folds = []
    for i, test_session in enumerate(session_list, start=1):
        train_sessions = [s for s in session_list if s != test_session]
        folds.append({
            "fold_name": f"fold{i}",
            "train_sessions": train_sessions,
            "test_session": test_session,
        })
    return folds


def save_one_file(src_path, dst_path):
    
    if os.path.lexists(dst_path):
        os.remove(dst_path)

    if save_mode == "copy":
        shutil.copy2(src_path, dst_path)
    elif save_mode == "symlink":
        os.symlink(src_path, dst_path)
    elif save_mode == "hardlink":
        os.link(src_path, dst_path)
    else:
        raise ValueError("Unknown save_mode: {}".format(save_mode))


def save_files(file_list, src_dir, dst_dir):
    
    os.makedirs(dst_dir, exist_ok=True)

    for filename in file_list:
        src_path = os.path.abspath(os.path.join(src_dir, filename))
        dst_path = os.path.join(dst_dir, filename)
        save_one_file(src_path, dst_path)


def process_one_input_dir(input_dir):
    input_dir_abs = os.path.abspath(input_dir)
    output_dir = os.path.join(save_dir, os.path.basename(input_dir))
    output_dir_abs = os.path.abspath(output_dir)

    if not os.path.isdir(input_dir_abs):
        raise RuntimeError("Input directory does not exist: {}".format(input_dir_abs))

    if save_mode not in {"copy", "symlink", "hardlink"}:
        raise RuntimeError("save_mode must be one of: copy, symlink, hardlink")

    if os.path.exists(output_dir_abs):
        if overwrite:
            shutil.rmtree(output_dir_abs)
        else:
            raise RuntimeError("Output directory already exists: {}".format(output_dir_abs))

    os.makedirs(output_dir_abs, exist_ok=True)

    classes = list_classes(input_dir_abs)
    if not classes:
        raise RuntimeError("No class subdirectories found under: {}".format(input_dir_abs))

    session_list = collect_all_sessions(input_dir_abs, classes)

    if len(session_list) != 4:
        raise RuntimeError(
            "Expected exactly 4 sessions for 4-fold split, but found: {}".format(session_list)
        )

    folds = get_fold_definitions(session_list)

    all_folds_summary = {
        "input_dir": input_dir_abs,
        "output_dir": output_dir_abs,
        "split_mode": "session_4fold",
        "save_mode": save_mode,
        "sessions": session_list,
        "num_folds": len(folds),
        "folds": []
    }

    print("Input directory: {}".format(input_dir_abs))
    print("Output directory: {}".format(output_dir_abs))
    print("Split mode: session_4fold")
    print("Save mode: {}".format(save_mode))
    print("Sessions: {}".format(session_list))
    print("Number of folds: {}".format(len(folds)))
    print("-" * 80)

    for fold in folds:
        fold_name = fold["fold_name"]
        train_sessions = set(fold["train_sessions"])
        test_session = fold["test_session"]

        fold_dir = os.path.join(output_dir_abs, fold_name)
        train_dir = os.path.join(fold_dir, "train")
        test_dir = os.path.join(fold_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        fold_record = {
            "fold_name": fold_name,
            "split_mode": "session_4fold",
            "train_sessions": sorted(list(train_sessions)),
            "test_session": test_session,
            "classes": {}
        }

        total_train_files = 0
        total_test_files = 0

        for cls in classes:
            cls_dir = os.path.join(input_dir_abs, cls)
            files = list_npy_files(cls_dir)

            train_files = []
            test_files = []

            for f in files:
                parsed_class, session_id, seg_id = parse_filename(f)

                if parsed_class != cls:
                    raise ValueError(
                        "Class name mismatch between directory and filename:\n"
                        "directory={}\nfilename={}".format(cls, f)
                    )

                if session_id in train_sessions:
                    train_files.append(f)
                elif session_id == test_session:
                    test_files.append(f)
                else:
                    raise ValueError(
                        "Session {} in file {} is not assigned to train/test".format(session_id, f)
                    )

            cls_train_dir = os.path.join(train_dir, cls)
            cls_test_dir = os.path.join(test_dir, cls)

            save_files(train_files, cls_dir, cls_train_dir)
            save_files(test_files, cls_dir, cls_test_dir)

            total_train_files += len(train_files)
            total_test_files += len(test_files)

            fold_record["classes"][cls] = {
                "train_count": len(train_files),
                "test_count": len(test_files),
                "train_files": train_files,
                "test_files": test_files,
            }

            print(
                "[{}] class [{}] | train {} | test {}".format(
                    fold_name, cls, len(train_files), len(test_files)
                )
            )

        fold_record["summary"] = {
            "train_total_files": total_train_files,
            "test_total_files": total_test_files,
        }

        split_record_path = os.path.join(fold_dir, "split_record.json")
        with open(split_record_path, "w", encoding="utf-8") as f:
            json.dump(fold_record, f, indent=2, ensure_ascii=False)

        all_folds_summary["folds"].append({
            "fold_name": fold_name,
            "train_sessions": sorted(list(train_sessions)),
            "test_session": test_session,
            "train_total_files": total_train_files,
            "test_total_files": total_test_files,
            "split_record": split_record_path
        })

        print(
            "[{}] done | train_sessions {} | test_session {} | train {} | test {}".format(
                fold_name,
                sorted(list(train_sessions)),
                test_session,
                total_train_files,
                total_test_files
            )
        )
        print("-" * 80)

    summary_path = os.path.join(output_dir_abs, "all_folds_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_folds_summary, f, indent=2, ensure_ascii=False)

    print("All folds created successfully for:", input_dir)
    print("Summary file: {}".format(summary_path))
    print()


def main():
    for split_mode in split_modes_to_run:
        input_dir = f"crossdevice/{split_mode}_mel_npy"
        process_one_input_dir(input_dir)


if __name__ == "__main__":
    main()