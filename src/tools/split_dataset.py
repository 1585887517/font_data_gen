import os
import random
import shutil


def copy_or_link(src, dst):

    if os.path.exists(dst):
        os.remove(dst)

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_dataset(
    img_dir,
    mask_dir,
    out_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    clean=False
):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # ==================================================
    # 1. collect files
    # ==================================================
    files = [
        f for f in os.listdir(img_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    files.sort()
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # ==================================================
    # 2. create folders
    # ==================================================
    for split in splits.keys():
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # ==================================================
    # 3. helper function
    # ==================================================
    def move_and_write(split_name, file_list):

        txt_path = os.path.join(out_dir, f"{split_name}.txt")

        with open(txt_path, "w") as f:

            for name in file_list:

                img_src = os.path.join(img_dir, name)
                base_name = os.path.splitext(name)[0]
                mask_name = f"{base_name}.png"
                mask_src = os.path.join(mask_dir, mask_name)

                img_dst = os.path.join(out_dir, "images", split_name, name)
                mask_dst = os.path.join(out_dir, "labels", split_name, mask_name)

                # hardlink first: same filesystem dataset split is almost free
                copy_or_link(img_src, img_dst)
                copy_or_link(mask_src, mask_dst)

                # write txt (PaddleSeg format)
                f.write(f"images/{split_name}/{name} labels/{split_name}/{mask_name}\n")

        print(f"[{split_name}] -> {len(file_list)} samples")

    # ==================================================
    # 4. process all splits
    # ==================================================
    move_and_write("train", train_files)
    move_and_write("val", val_files)
    move_and_write("test", test_files)

    print("✅ dataset split done:", out_dir)
