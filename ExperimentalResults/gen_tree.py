#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

EXCLUDE_DIRS = {'.git', '__pycache__', '.idea', '.vscode'}
STOP_DIRS = {'tmp_mel_split', 'tmp_mel'}
PNG_EXT = '.png'

def print_tree(root, prefix=""):
    entries = [
        e for e in os.listdir(root)
        if e not in EXCLUDE_DIRS
    ]
    entries.sort()

    png_files = [e for e in entries if e.lower().endswith(PNG_EXT)]
    other_entries = [e for e in entries if not e.lower().endswith(PNG_EXT)]

    for i, entry in enumerate(other_entries):
        path = os.path.join(root, entry)
        is_last = (i == len(other_entries) - 1 and not png_files)
        connector = "└── " if is_last else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            if entry in STOP_DIRS:
                continue

            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    print(os.path.basename(os.getcwd()) + "/")
    print_tree(os.getcwd())
