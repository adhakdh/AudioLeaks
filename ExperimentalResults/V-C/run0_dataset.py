# -*- coding: utf-8 -*-
import os
import re
import json
import shutil

# =========================
# Config
# =========================
input_dir = "../../Datasets/appidentity" 

split_modes_to_run = ["day", "session", "user"]

overwrite = True
save_mode = "hardlink"   # "symlink" / "copy" / "hardlink"

output_root_name = "cross_fold"

target_users = None

filename_pattern = re.compile(r"^(.+?)_(user\d+)_(\d+)_(seg\d+)\.npy$")
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
    user_name = m.group(2)
    session_id = int(m.group(3))
    seg_id = m.group(4)

    return class_name, user_name, session_id, seg_id


def natural_user_key(user_name):
    
    m = re.match(r"^user(\d+)$", user_name)
    if m:
        return int(m.group(1))
    return user_name


def session_to_day(session_id):
    if session_id in [1, 2]:
        return "day1"
    elif session_id in [3, 4]:
        return "day2"
    elif session_id in [5, 6]:
        return "day3"
    else:
        raise ValueError("Unsupported session_id for day split: {}".format(session_id))


def session_to_halfday(session_id):
    if session_id % 2 == 1:
        return "am"
    return "pm"


def collect_all_users(input_dir_abs, classes):
    
    users = set()

    for cls in classes:
        cls_dir = os.path.join(input_dir_abs, cls)
        files = list_npy_files(cls_dir)

        for f in files:
            parsed_class, user_name, session_id, seg_id = parse_filename(f)
            if parsed_class != cls:
                raise ValueError(
                    "Class name mismatch between directory and filename:\n"
                    "directory={}\nfilename={}".format(cls, f)
                )
            users.add(user_name)

    return sorted(users, key=natural_user_key)


def get_fold_definitions(split_mode, user_list=None):
    if split_mode == "day":
        groups = ["day1", "day2", "day3"]
        folds = []
        for i, test_group in enumerate(groups, start=1):
            train_groups = [g for g in groups if g != test_group]
            folds.append({
                "fold_name": "fold{}".format(i),
                "train_groups": train_groups,
                "test_group": test_group,
            })
        return folds

    elif split_mode == "session":
        groups = ["am", "pm"]
        folds = []
        for i, test_group in enumerate(groups, start=1):
            train_groups = [g for g in groups if g != test_group]
            folds.append({
                "fold_name": "fold{}".format(i),
                "train_groups": train_groups,
                "test_group": test_group,
            })
        return folds

    elif split_mode == "user":
        if user_list is None or len(user_list) == 0:
            raise ValueError("user_list is required when split_mode='user'")

        folds = []
        for i, test_group in enumerate(user_list, start=1):
            train_groups = [u for u in user_list if u != test_group]
            folds.append({
                "fold_name": "fold{}".format(i),
                "train_groups": train_groups,
                "test_group": test_group,
            })
        return folds

    else:
        raise ValueError("Unsupported split_mode: {}".format(split_mode))


def get_sample_group(split_mode, user_name, session_id):
    
    if split_mode == "day":
        return session_to_day(session_id)
    elif split_mode == "session":
        return session_to_halfday(session_id)
    elif split_mode == "user":
        return user_name
    else:
        raise ValueError("Unsupported split_mode: {}".format(split_mode))


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


def run_one_split_mode(input_dir_abs, classes, split_mode, output_root_abs):
    output_dir = os.path.join(output_root_abs, "{}_mel_npy".format(split_mode))

    if split_mode not in {"day", "session", "user"}:
        raise RuntimeError("split_mode must be one of: day, session, user")

    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError("Output directory already exists: {}".format(output_dir))

    os.makedirs(output_dir, exist_ok=True)

    if split_mode == "user":
        if target_users is None:
            user_list = collect_all_users(input_dir_abs, classes)
        else:
            user_list = sorted(target_users, key=natural_user_key)

        if len(user_list) < 2:
            raise RuntimeError("At least 2 users are required for user split")
    else:
        user_list = None

    folds = get_fold_definitions(split_mode, user_list)

    all_folds_summary = {
        "input_dir": input_dir_abs,
        "output_dir": output_dir,
        "output_root": output_root_abs,
        "split_mode": split_mode,
        "save_mode": save_mode,
        "users": user_list if user_list is not None else [],
        "num_folds": len(folds),
        "folds": []
    }

    print("=" * 100)
    print("Input directory: {}".format(input_dir_abs))
    print("Output root: {}".format(output_root_abs))
    print("Output directory: {}".format(output_dir))
    print("Split mode: {}".format(split_mode))
    print("Save mode: {}".format(save_mode))
    if user_list is not None:
        print("Users: {}".format(user_list))
    print("Number of folds: {}".format(len(folds)))
    print("-" * 80)

    for fold in folds:
        fold_name = fold["fold_name"]
        train_groups = set(fold["train_groups"])
        test_group = fold["test_group"]

        fold_dir = os.path.join(output_dir, fold_name)
        train_dir = os.path.join(fold_dir, "train")
        test_dir = os.path.join(fold_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        fold_record = {
            "fold_name": fold_name,
            "split_mode": split_mode,
            "train_groups": sorted(list(train_groups)),
            "test_group": test_group,
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
                parsed_class, user_name, session_id, seg_id = parse_filename(f)

                if parsed_class != cls:
                    raise ValueError(
                        "Class name mismatch between directory and filename:\n"
                        "directory={}\nfilename={}".format(cls, f)
                    )

                sample_group = get_sample_group(split_mode, user_name, session_id)

                if sample_group in train_groups:
                    train_files.append(f)
                elif sample_group == test_group:
                    test_files.append(f)
                else:
                    raise ValueError(
                        "Sample group {} in file {} is not assigned to train/test".format(sample_group, f)
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
                "[{}][{}] class [{}] | train {} | test {}".format(
                    split_mode, fold_name, cls, len(train_files), len(test_files)
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
            "train_groups": sorted(list(train_groups)),
            "test_group": test_group,
            "train_total_files": total_train_files,
            "test_total_files": total_test_files,
            "split_record": split_record_path
        })

        print(
            "[{}][{}] done | train_groups {} | test_group {} | train {} | test {}".format(
                split_mode,
                fold_name,
                sorted(list(train_groups)),
                test_group,
                total_train_files,
                total_test_files
            )
        )
        print("-" * 80)

    summary_path = os.path.join(output_dir, "all_folds_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_folds_summary, f, indent=2, ensure_ascii=False)

    print("[{}] All folds created successfully.".format(split_mode))
    print("[{}] Summary file: {}".format(split_mode, summary_path))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir_abs = os.path.abspath(os.path.join(script_dir, input_dir))
    output_root_abs = os.path.join(script_dir, output_root_name)

    if not os.path.isdir(input_dir_abs):
        raise RuntimeError("Input directory does not exist: {}".format(input_dir_abs))

    if save_mode not in {"copy", "symlink", "hardlink"}:
        raise RuntimeError("save_mode must be one of: copy, symlink, hardlink")

    if not split_modes_to_run:
        raise RuntimeError("split_modes_to_run is empty")

    for mode in split_modes_to_run:
        if mode not in {"day", "session", "user"}:
            raise RuntimeError("Invalid split mode in split_modes_to_run: {}".format(mode))

    classes = list_classes(input_dir_abs)
    if not classes:
        raise RuntimeError("No class subdirectories found under: {}".format(input_dir_abs))

    if os.path.exists(output_root_abs):
        if overwrite:
            shutil.rmtree(output_root_abs)
        else:
            raise RuntimeError("Output root already exists: {}".format(output_root_abs))

    os.makedirs(output_root_abs, exist_ok=True)

    print("Script directory: {}".format(script_dir))
    print("Input directory: {}".format(input_dir_abs))
    print("Output root: {}".format(output_root_abs))
    print("Split modes to run: {}".format(split_modes_to_run))

    for split_mode in split_modes_to_run:
        run_one_split_mode(input_dir_abs, classes, split_mode, output_root_abs)


if __name__ == "__main__":
    main()