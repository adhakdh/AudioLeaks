# -*- coding: utf-8 -*-
import os
import shutil
import random
import json


input_dir = "../../Datasets/appidentity"    
output_dir = f"{input_dir}_split"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random_seed = 42                           
overwrite = True                           
save_mode = "hardlink"                      
# ==========================================


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


def extract_session_id(filename):
    base = os.path.splitext(filename)[0]

    if "_seg" in base:
        return base.split("_seg")[0]

    raise ValueError("Invalid filename format, '_seg' not found: {}".format(filename))


def group_files_by_session(files):

    session_dict = {}
    for f in files:
        session_id = extract_session_id(f)
        session_dict.setdefault(session_id, []).append(f)

    # 保证每个 session 内文件顺序稳定
    for session_id in session_dict:
        session_dict[session_id].sort()

    return session_dict


def split_sessions(session_ids, train_ratio, val_ratio, test_ratio, rng):

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = session_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return [], [], []

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    while n_train + n_val + n_test < n:
        n_train += 1
    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 0:
            n_train -= 1
        elif n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    if n >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        if n_test == 0:
            n_test = 1

        while n_train + n_val + n_test > n:
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

    elif n == 2:
        n_train, n_val, n_test = 1, 0, 1

    elif n == 1:
        n_train, n_val, n_test = 1, 0, 0

    train_sessions = shuffled[:n_train]
    val_sessions = shuffled[n_train:n_train + n_val]
    test_sessions = shuffled[n_train + n_val:n_train + n_val + n_test]

    return train_sessions, val_sessions, test_sessions


def collect_files_from_sessions(session_dict, session_list):
    
    files = []
    for session_id in session_list:
        files.extend(session_dict[session_id])
    return sorted(files)


def save_files(file_list, src_dir, dst_dir):
    
    os.makedirs(dst_dir, exist_ok=True)

    for f in file_list:
        src_path = os.path.abspath(os.path.join(src_dir, f))
        dst_path = os.path.join(dst_dir, f)

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


def main():
    input_dir_abs = os.path.abspath(input_dir)
    output_dir_abs = os.path.abspath(output_dir)

    train_dir = os.path.join(output_dir_abs, "train")
    val_dir = os.path.join(output_dir_abs, "val")
    test_dir = os.path.join(output_dir_abs, "test")
    split_record_path = os.path.join(output_dir_abs, "split_record.json")

    if not os.path.isdir(input_dir_abs):
        raise RuntimeError("Input directory does not exist: {}".format(input_dir_abs))

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise RuntimeError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if save_mode not in {"copy", "symlink", "hardlink"}:
        raise RuntimeError("save_mode must be one of: copy, symlink, hardlink")

    if os.path.exists(output_dir_abs):
        if overwrite:
            shutil.rmtree(output_dir_abs)
        else:
            raise RuntimeError("Output directory already exists: {}".format(output_dir_abs))

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = list_classes(input_dir_abs)
    if not classes:
        raise RuntimeError("No class subdirectories found under: {}".format(input_dir_abs))

    rng = random.Random(random_seed)

    split_record = {
        "input_dir": input_dir_abs,
        "output_dir": output_dir_abs,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "split_unit": "session",
        "save_mode": save_mode,
        "classes": {}
    }

    total_train_files = 0
    total_val_files = 0
    total_test_files = 0

    total_train_sessions = 0
    total_val_sessions = 0
    total_test_sessions = 0

    for cls in classes:
        cls_dir = os.path.join(input_dir_abs, cls)
        files = list_npy_files(cls_dir)

        if not files:
            print("Skip empty class: {}".format(cls))
            continue

        session_dict = group_files_by_session(files)
        session_ids = sorted(session_dict.keys())

        train_sessions, val_sessions, test_sessions = split_sessions(
            session_ids, train_ratio, val_ratio, test_ratio, rng
        )

        train_files = collect_files_from_sessions(session_dict, train_sessions)
        val_files = collect_files_from_sessions(session_dict, val_sessions)
        test_files = collect_files_from_sessions(session_dict, test_sessions)

        cls_train_dir = os.path.join(train_dir, cls)
        cls_val_dir = os.path.join(val_dir, cls)
        cls_test_dir = os.path.join(test_dir, cls)

        save_files(train_files, cls_dir, cls_train_dir)
        save_files(val_files, cls_dir, cls_val_dir)
        save_files(test_files, cls_dir, cls_test_dir)

        total_train_files += len(train_files)
        total_val_files += len(val_files)
        total_test_files += len(test_files)

        total_train_sessions += len(train_sessions)
        total_val_sessions += len(val_sessions)
        total_test_sessions += len(test_sessions)

        split_record["classes"][cls] = {
            "total_files": len(files),
            "total_sessions": len(session_ids),

            "train_session_count": len(train_sessions),
            "val_session_count": len(val_sessions),
            "test_session_count": len(test_sessions),

            "train_file_count": len(train_files),
            "val_file_count": len(val_files),
            "test_file_count": len(test_files),

            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "test_sessions": test_sessions,

            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files
        }

        print(
            "label [{}]: total_files {}, total_sessions {}, "
            "train_sessions {}, val_sessions {}, test_sessions {}, "
            "train_files {}, val_files {}, test_files {}".format(
                cls,
                len(files),
                len(session_ids),
                len(train_sessions),
                len(val_sessions),
                len(test_sessions),
                len(train_files),
                len(val_files),
                len(test_files)
            )
        )

    with open(split_record_path, "w", encoding="utf-8") as f:
        json.dump(split_record, f, indent=2, ensure_ascii=False)

    print("\nDone")
    print("Input directory: {}".format(input_dir_abs))
    print("Output directory: {}".format(output_dir_abs))
    print("Save mode: {}".format(save_mode))
    print("Train sessions: {}".format(total_train_sessions))
    print("Validation sessions: {}".format(total_val_sessions))
    print("Test sessions: {}".format(total_test_sessions))
    print("Train files: {}".format(total_train_files))
    print("Validation files: {}".format(total_val_files))
    print("Test files: {}".format(total_test_files))
    print("Split record: {}".format(split_record_path))


if __name__ == "__main__":
    main()