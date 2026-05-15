import os
import cv2
import numpy as np
import random
from datasets import load_dataset
from tools.logger import Logger


class HandwritingLoader:
    """
    工业级 handwriting loader（RAW + RGBA 双层结构）
    """

    def __init__(self, cfg):

        self.cfg = cfg

        # ==================================================
        # 🚀 RAW 数据（不可变）
        # ==================================================
        self.iam_raw = self._load_images(cfg.IAM_DIR, "iam")
        self.casia_raw = self._load_images(cfg.CASIA_DIR, "casia")

        # ==================================================
        # 🚀 RGBA 数据（overlay专用）
        # ==================================================
        self.iam_rgba = self._prepare_rgba(
            name="iam",
            raw_files=self.iam_raw,
            out_dir=cfg.IAM_RGBA_DIR
        )

        self.casia_rgba = self._prepare_rgba(
            name="casia",
            raw_files=self.casia_raw,
            out_dir=cfg.CASIA_RGBA_DIR
        )


    # ==================================================
    # 🚀 load raw images
    # ==================================================
    def _load_images(self, path, dataset_name):

        os.makedirs(path, exist_ok=True)

        files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # ==================================================
        # 1. 本地已有数据
        # ==================================================
        if len(files) > 0:
            return files

        # ==================================================
        # 2. fallback dataset
        # ==================================================

        dataset_name_map = {
            "iam": "Teklia/IAM-line",
            "casia": "Teklia/CASIA-HWDB2-line"
        }

        dataset = load_dataset(
            dataset_name_map.get(dataset_name, "Teklia/IAM-line")
        )


        all_items = []

        # ==================================================
        # 3. 遍历所有 split（关键升级点）
        # ==================================================
        for split_name, split_data in dataset.items():

            Logger.info(f"[load] processing split={split_name}, size={len(split_data)}")

            for i, item in enumerate(split_data):

                img = item["image"]

                save_path = os.path.join(
                    path,
                    f"{dataset_name}_{split_name}_{i}.png"
                )

                img.save(save_path)

                all_items.append(save_path)

            Logger.info(f"[load] finished split={split_name}")

        # ==================================================
        # 4. return all files
        # ==================================================
        Logger.info(f"[load] total images saved: {len(all_items)}")

        return sorted(all_items)


    # ==================================================
    # 🚀 RAW → RGBA（不改变原数据）
    # ==================================================
    def _prepare_rgba(self, name, raw_files, out_dir):

        os.makedirs(out_dir, exist_ok=True)

        rgba_list = []

        # ==================================================
        # 🚀 已存在RGBA则直接读取
        # ==================================================
        existing = [
            os.path.join(out_dir, f)
            for f in sorted(os.listdir(out_dir))
            if f.endswith(".png")
        ]

        if len(existing) > 0:
            return existing


        # ==================================================
        # 🚀 RAW → RGBA
        # ==================================================
        for i, f in enumerate(raw_files):

            img = cv2.imread(f, cv2.IMREAD_COLOR)

            if img is None:
                Logger.warn(f"Failed to read: {f}")
                continue

            # ==================================================
            # 1. 灰度化
            # ==================================================
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            # ==================================================
            # 2. 白底去除（核心）
            # ==================================================
            _, alpha = cv2.threshold(
                gray,
                235,
                255,
                cv2.THRESH_BINARY_INV
            )

            # ==================================================
            # 3. RGBA合并
            # ==================================================
            b, g, r = cv2.split(img)

            rgba = cv2.merge([b, g, r, alpha])

            # ==================================================
            # 4. 保存RGBA
            # ==================================================
            save_path = os.path.join(
                out_dir,
                f"{name}_{i}.png"
            )

            cv2.imwrite(save_path, rgba)

            rgba_list.append(save_path)

        Logger.info(f"{name} RGBA generated: {len(rgba_list)}")

        return rgba_list    


    # ==================================================
    # 🚀 关键：按 source 取数据（1:1控制点）
    # ==================================================
    def overlay_by_source(self, img, mask, source, allow_overlap=False):

     
        files = (
            self.iam_rgba if source == "iam"
            else self.casia_rgba
        )

        if not files:
            raise RuntimeError(f"{source} RGBA is empty")

        f = random.choice(files)

        hw = cv2.imread(f, cv2.IMREAD_UNCHANGED)

        if hw is None or hw.shape[-1] != 4:
            return img, mask

        # ==================================================
        # 1. split
        # ==================================================
        rgb = cv2.cvtColor(hw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        alpha = hw[:, :, 3].astype(np.float32) / 255.0

        ys, xs = np.where(alpha > 0)
        if len(xs) == 0 or len(ys) == 0:
            return img, mask

        pad = 4
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(alpha.shape[1], int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(alpha.shape[0], int(ys.max()) + pad + 1)
        rgb = rgb[y0:y1, x0:x1]
        alpha = alpha[y0:y1, x0:x1]

        # ==================================================
        # 2. degradation (optional realism)
        # ==================================================
        alpha *= random.uniform(0.62, 1.0)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        ink_palette = random.choice([
            np.array([135, 24, 42], dtype=np.float32),
            np.array([35, 42, 120], dtype=np.float32),
            np.array([20, 20, 20], dtype=np.float32),
            np.array([95, 44, 60], dtype=np.float32),
        ])
        rgb = rgb * random.uniform(0.15, 0.45) + ink_palette * random.uniform(0.55, 0.85)

        # ==================================================
        # 3. size safety (IMPORTANT FIX)
        # ==================================================
        H, W = img.shape[:2]
        h, w = rgb.shape[:2]

        target_width = random.uniform(0.22, 0.96) * W
        target_height = random.uniform(0.05, 0.20) * H
        max_scale = min(H / h, W / w, 2.2)
        desired_scale = min(target_width / w, target_height / h)
        scale = min(max_scale, max(0.28, desired_scale * random.uniform(0.85, 1.25)))

        if abs(scale - 1.0) > 0.05:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            h, w = new_h, new_w

        # ==================================================
        # 4. SAFE POSITION (FIXED BOUNDARY BUG)
        # ==================================================
        placement = random.random()
        if placement < 0.45:
            x_min = int(W * 0.04)
            x_max = int(W * 0.72)
            y_min = int(H * 0.18)
            y_max = int(H * 0.78)
        elif placement < 0.75:
            x_min = int(W * 0.02)
            x_max = int(W * 0.45)
            y_min = int(H * 0.55)
            y_max = int(H * 0.90)
        else:
            x_min = 0
            x_max = max(0, W - w)
            y_min = 0
            y_max = max(0, H - h)

        x_low = min(max(0, x_min), max(0, W - w))
        x_high = max(x_low, min(max(0, x_max), max(0, W - w)))
        y_low = min(max(0, y_min), max(0, H - h))
        y_high = max(y_low, min(max(0, y_max), max(0, H - h)))

        # ==================================================
        # 5. avoid overlap with printed text and existing handwriting
        # ==================================================
        max_attempts = 32
        success = False
        
        # 决定是否启用“邻近”模式
        force_proximity = random.random() < getattr(self.cfg, "HANDWRITING_PROXIMITY_PROB", 0.0)

        for _ in range(max_attempts):
            x = random.randint(x_low, x_high)
            y = random.randint(y_low, y_high)

            actual_h = min(h, H - y)
            actual_w = min(w, W - x)
            if actual_h <= 0 or actual_w <= 0:
                continue

            alpha_crop = alpha[:actual_h, :actual_w]
            text_region = alpha_crop > 0.08
            if not np.any(text_region):
                continue

            existing_region = mask[y:y+actual_h, x:x+actual_w]
            
            # 如果启用了邻近模式，我们需要该区域周围已经有东西（类 1 或类 2）
            if force_proximity:
                # 检查 20 像素范围内的邻近区域
                y_near_min = max(0, y - 20)
                y_near_max = min(H, y + actual_h + 20)
                x_near_min = max(0, x - 20)
                x_near_max = min(W, x + actual_w + 20)
                near_mask = mask[y_near_min:y_near_max, x_near_min:x_near_max]
                if not np.any(near_mask > 0):
                    continue

            if not allow_overlap:
                if np.any(existing_region[text_region] > 0):
                    continue

            success = True
            break

        if not success:
            return img, mask

        rgb = rgb[:actual_h, :actual_w]
        alpha = alpha[:actual_h, :actual_w]
        roi = img[y:y+actual_h, x:x+actual_w].astype(np.float32)

        # ==================================================
        # 6. blending
        # ==================================================
        alpha_3 = np.stack([alpha]*3, axis=-1)
        blended = alpha_3 * rgb + (1 - alpha_3) * roi
        img[y:y+actual_h, x:x+actual_w] = blended.astype(np.uint8)

        # ==================================================
        # 7. mask (must match SAME SHAPE)
        # ==================================================
        mask[y:y+actual_h, x:x+actual_w][text_region] = 2

        return img, mask
