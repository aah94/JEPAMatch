# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Prepare Tiny-ImageNet-200 for semilearn.datasets.cv_datasets.imagenet.get_imagenet
(dataset='tiny_imagenet'), which expects an ImageFolder-style layout:

    <data_dir>/tiny_imagenet/train/<wnid>/*.JPEG
    <data_dir>/tiny_imagenet/val/<wnid>/*.JPEG

The official download (http://cs231n.stanford.edu/tiny-imagenet-200.zip) already
has train images under train/<wnid>/images/*.JPEG, which ImageFolder-style loaders
find fine (subfolders are walked recursively). Its val split, however, is flat
(val/images/*.JPEG) with a separate val_annotations.txt mapping filename -> wnid,
which is NOT directly usable -- this script reorganizes val/ into val/<wnid>/*.JPEG
(via symlinks, so it's fast and uses no extra disk) and symlinks train/ into place.

Usage:
    python preprocess/preprocess_tiny_imagenet.py --data_dir ./data [--download]

With --download, the raw archive is fetched and extracted automatically if not
already present at <data_dir>/tiny-imagenet-200-raw. Otherwise, point --raw_dir at
an already-extracted tiny-imagenet-200 folder.
"""
import argparse
import os
import shutil
import zipfile

DOWNLOAD_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_and_extract(raw_dir):
    parent = os.path.dirname(raw_dir.rstrip("/"))
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        import urllib.request
        print(f"Downloading {DOWNLOAD_URL} -> {zip_path}")
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)

    print(f"Extracting {zip_path} -> {parent}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(parent)
    # the archive extracts to <parent>/tiny-imagenet-200
    extracted = os.path.join(parent, "tiny-imagenet-200")
    if extracted != raw_dir:
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
        shutil.move(extracted, raw_dir)


def build_train_split(raw_dir, out_dir):
    """Symlink raw_dir/train/<wnid>/images/*.JPEG classes into out_dir/train/<wnid>/."""
    src_train = os.path.join(raw_dir, "train")
    dst_train = os.path.join(out_dir, "train")
    os.makedirs(dst_train, exist_ok=True)

    wnids = sorted(d for d in os.listdir(src_train) if os.path.isdir(os.path.join(src_train, d)))
    for wnid in wnids:
        src_images = os.path.join(src_train, wnid, "images")
        dst_class_dir = os.path.join(dst_train, wnid)
        if os.path.islink(dst_class_dir) or os.path.exists(dst_class_dir):
            continue
        os.symlink(os.path.abspath(src_images), dst_class_dir)
    print(f"train: linked {len(wnids)} classes -> {dst_train}")


def build_val_split(raw_dir, out_dir):
    """Reorganize the flat raw_dir/val/images/*.JPEG into out_dir/val/<wnid>/*.JPEG."""
    src_val = os.path.join(raw_dir, "val")
    src_images = os.path.join(src_val, "images")
    annotations_path = os.path.join(src_val, "val_annotations.txt")
    dst_val = os.path.join(out_dir, "val")
    os.makedirs(dst_val, exist_ok=True)

    filename_to_wnid = {}
    with open(annotations_path, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            filename, wnid = fields[0], fields[1]
            filename_to_wnid[filename] = wnid

    made_dirs = set()
    for filename, wnid in filename_to_wnid.items():
        class_dir = os.path.join(dst_val, wnid)
        if wnid not in made_dirs:
            os.makedirs(class_dir, exist_ok=True)
            made_dirs.add(wnid)
        dst_link = os.path.join(class_dir, filename)
        if not os.path.islink(dst_link) and not os.path.exists(dst_link):
            os.symlink(os.path.abspath(os.path.join(src_images, filename)), dst_link)

    print(f"val: linked {len(filename_to_wnid)} images into {len(made_dirs)} classes -> {dst_val}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="semilearn data root (matches the yaml config's data_dir)")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="path to an already-extracted tiny-imagenet-200 folder "
                             "(default: <data_dir>/tiny-imagenet-200-raw)")
    parser.add_argument("--download", action="store_true",
                        help="download + extract the official archive into --raw_dir if missing")
    args = parser.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.data_dir, "tiny-imagenet-200-raw")
    out_dir = os.path.join(args.data_dir, "tiny_imagenet")

    if args.download and not os.path.isdir(raw_dir):
        download_and_extract(raw_dir)

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"{raw_dir} not found. Extract tiny-imagenet-200.zip there first, "
            f"or pass --download to fetch it automatically."
        )

    os.makedirs(out_dir, exist_ok=True)
    build_train_split(raw_dir, out_dir)
    build_val_split(raw_dir, out_dir)
    print(f"Done. dataset: tiny_imagenet, data_dir: {args.data_dir} (i.e. {out_dir})")


if __name__ == "__main__":
    main()
