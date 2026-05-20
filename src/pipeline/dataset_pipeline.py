import random
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from tools.logger import Logger
from tools.augmentations import Augmentations as A
from tools.io_utils import IOUtils as IO
from tools.split_dataset import split_dataset

_CFG = None
_GEN = None
_HW = None
_TL = None


def init_worker(cfg, gen_cls, hw_cls, text_loader_cls=None):
    global _CFG, _GEN, _HW, _TL

    _CFG = cfg
    _TL = text_loader_cls(cfg) if text_loader_cls else None
    _GEN = gen_cls(cfg, _TL)
    _HW = hw_cls(cfg)


def class_ratios(mask):

    total = max(1, mask.size)
    return {
        "foreground": float((mask > 0).sum()) / total,
        "printed": float((mask == 1).sum()) / total,
        "handwriting": float((mask == 2).sum()) / total,
    }


def overlay_handwriting_until_added(img, mask, hw, source, allow_overlap, max_attempts):

    for _ in range(max_attempts):
        before_count = int((mask == 2).sum())
        next_img, next_mask = hw.overlay_by_source(
            img,
            mask,
            source,
            allow_overlap=allow_overlap
        )
        after_count = int((next_mask == 2).sum())

        if after_count > before_count:
            return next_img, next_mask, True

        img, mask = next_img, next_mask

    return img, mask, False


def sample_meets_requirements(mask, task_mode, cfg):

    ratios = class_ratios(mask)

    if task_mode == "printed_only":
        return ratios["printed"] >= cfg.MIN_PRINTED_RATIO

    if task_mode == "handwriting_only":
        return ratios["handwriting"] >= cfg.MIN_HANDWRITING_RATIO

    return (
        ratios["foreground"] >= cfg.MIN_FOREGROUND_RATIO
        and ratios["printed"] >= cfg.MIN_PRINTED_RATIO
        and ratios["handwriting"] >= cfg.MIN_HANDWRITING_RATIO
    )


# ==================================================
# 🚀 worker（必须在全局，避免 multiprocessing pickle 问题）
# ==================================================
def worker(task):

    import random
    import numpy as np

    seed = task["seed"]
    random.seed(seed)
    np.random.seed(seed % (2**32))

    cfg = _CFG
    gen = _GEN
    hw = _HW

    task_mode = task.get("dataset_mode", cfg.DATASET_MODE)
    original_mode = cfg.DATASET_MODE
    cfg.DATASET_MODE = task_mode

    overlay_count = 0
    allow_overlap = False
    if task_mode == "both":
        overlay_min, overlay_max = cfg.HANDWRITING_OVERLAYS_PER_IMAGE
        overlay_count = random.randint(overlay_min, overlay_max)
    elif task_mode == "both_overlap":
        overlay_min, overlay_max = cfg.HANDWRITING_OVERLAYS_PER_IMAGE
        overlay_count = random.randint(overlay_min, overlay_max)
        allow_overlap = True
    elif task_mode == "handwriting_only":
        overlay_count = random.randint(5, 10)

    max_rebuild_attempts = getattr(cfg, "MAX_REBUILD_ATTEMPTS", 8)
    max_overlay_attempts = max(4, overlay_count * 8)
    extra_count = 0
    while True:
        img, mask = gen.build()

        for _ in range(overlay_count):
            img, mask, _ = overlay_handwriting_until_added(
                img,
                mask,
                hw,
                task["source"],
                allow_overlap,
                max_overlay_attempts
            )

        if sample_meets_requirements(mask, task_mode, cfg):
            break

        extra_count += 1
        if extra_count >= max_rebuild_attempts:
            break

    cfg.DATASET_MODE = original_mode

    r = random.random()
    if r < 0.35:
        img, mask = A.rotate(img, mask)
    elif r < 0.70:
        img, mask = A.perspective(img, mask)
    
    # 随机增加凹凸或弯曲形变
    if random.random() < 0.4:
        if random.random() < 0.5:
            img, mask = A.mesh_distortion(img, mask)
        else:
            img, mask = A.bending_warp(img, mask)

    if cfg.ENABLE_SCAN_NOISE:
        img, mask = A.scan_noise(img, mask, phone_effects=cfg.ENABLE_PHONE_EFFECTS)

    IO.save(img, mask, task["name"], cfg)

    return task["id"]


# ==================================================
# 🚀 Pipeline
# ==================================================
class DatasetPipeline:

    def __init__(self, cfg, gen_cls, hw_cls, text_loader_cls=None):

        self.cfg = cfg
        self.gen_cls = gen_cls
        self.hw_cls = hw_cls
        self.text_loader_cls = text_loader_cls

        os.makedirs(cfg.OUTPUT_IMG, exist_ok=True)
        os.makedirs(cfg.OUTPUT_MASK, exist_ok=True)

    # ==================================================
    # 🚀 run entry
    # ==================================================
    def run(self):

        total = self.cfg.NUM_SAMPLES

        iam_n = total // 2
        casia_n = total - iam_n

        if self.cfg.CLEAN_OUTPUT:
            self._clean_outputs()

        tasks = self._build_tasks(iam_n, casia_n)

        worker_count = min(self.cfg.NUM_WORKERS, max(1, len(tasks)))

        Logger.info(f"[pipeline] total={len(tasks)} workers={worker_count}")

        # ==================================================
        # 🚀 multiprocessing execution
        # ==================================================
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_worker,
            initargs=(self.cfg, self.gen_cls, self.hw_cls, self.text_loader_cls)
        ) as executor:

            list(tqdm(
                executor.map(worker, tasks),
                total=len(tasks),
                desc="generating"
            ))

        # ==================================================
        # 🚀 dataset split
        # ==================================================
        split_dataset(
            img_dir=self.cfg.OUTPUT_IMG,
            mask_dir=self.cfg.OUTPUT_MASK,
            out_dir=self.cfg.OUTPUT_DIR,
            train_ratio=self.cfg.TRAIN_RATIO,
            val_ratio=self.cfg.VAL_RATIO,
            test_ratio=self.cfg.TEST_RATIO,
            seed=self.cfg.RANDOM_SEED,
            clean=True
        )

        # ==================================================
        # 🚀 remove original images and masks after split
        # ==================================================
        for path in [self.cfg.OUTPUT_IMG, self.cfg.OUTPUT_MASK]:
            if os.path.isdir(path):
                shutil.rmtree(path)
                Logger.info(f"[pipeline] removed {path}")

        Logger.info("[pipeline] done")

    # ==================================================
    # 🚀 build tasks
    # ==================================================
    def _build_tasks(self, iam_n, casia_n):

        tasks = []
        rng = random.Random(self.cfg.RANDOM_SEED)

        # --------------------------------------------------
        # iam tasks
        # --------------------------------------------------
        for i in range(iam_n):

            tasks.append({
                "id": f"iam_{i}",
                "source": "iam",
                "seed": rng.randint(0, 10**18),
                "name": f"iam_{i}",
            })

        # --------------------------------------------------
        # casia tasks
        # --------------------------------------------------
        for i in range(casia_n):

            tasks.append({
                "id": f"casia_{i}",
                "source": "casia",
                "seed": rng.randint(0, 10**18),
                "name": f"casia_{i}",
            })

        for task in tasks:
            task["dataset_mode"] = self.cfg.DATASET_MODE

        rng.shuffle(tasks)

        return tasks

    def _clean_outputs(self):

        for path in [
            self.cfg.OUTPUT_IMG,
            self.cfg.OUTPUT_MASK,
            getattr(self.cfg, "OUTPUT_MASK_VIS", ""),
            self.cfg.OUTPUT_DIR,
        ]:
            if path and os.path.isdir(path):
                shutil.rmtree(path)
            if path:
                os.makedirs(path, exist_ok=True)

        Logger.info("[pipeline] cleaned previous generated output")
