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


# ==================================================
# 🚀 worker（必须在全局，避免 multiprocessing pickle 问题）
# ==================================================
def worker(task):

    import random
    import numpy as np

    random.seed(task["seed"])
    np.random.seed(task["seed"])

    cfg = _CFG
    gen = _GEN
    hw = _HW

    img, mask = gen.build()

    overlay_count = 0
    if cfg.DATASET_MODE == "both":
        overlay_min, overlay_max = cfg.HANDWRITING_OVERLAYS_PER_IMAGE
        overlay_count = random.randint(overlay_min, overlay_max)
    elif cfg.DATASET_MODE == "handwriting_only":
        # 为 handwriting_only 模式增加更多手写叠加
        overlay_count = random.randint(5, 10)

    for _ in range(overlay_count):
        img, mask = hw.overlay_by_source(img, mask, task["source"])

    min_foreground = (
        cfg.MIN_HANDWRITING_RATIO
        if cfg.DATASET_MODE == "handwriting_only"
        else cfg.MIN_FOREGROUND_RATIO
    )

    extra_count = 0
    while (
        (mask > 0).mean() < min_foreground
        and extra_count < 6
    ):
        # 如果前景不够，重新生成文档并重新叠加手写，否则最终可能只剩下背景
        img, mask = gen.build()
        for _ in range(overlay_count):
            img, mask = hw.overlay_by_source(img, mask, task["source"])
        extra_count += 1

    r = random.random()

    if r < 0.45:
        img, mask = A.rotate(img, mask)
    elif r < 0.9:
        img, mask = A.perspective(img, mask)

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
                "seed": rng.randint(0, 10**9),
                "name": f"iam_{i}",
            })

        # --------------------------------------------------
        # casia tasks
        # --------------------------------------------------
        for i in range(casia_n):

            tasks.append({
                "id": f"casia_{i}",
                "source": "casia",
                "seed": rng.randint(0, 10**9),
                "name": f"casia_{i}",
            })

        rng.shuffle(tasks)

        return tasks

    def _clean_outputs(self):

        for path in [
            self.cfg.OUTPUT_IMG,
            self.cfg.OUTPUT_MASK,
            self.cfg.OUTPUT_DIR,
        ]:
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        Logger.info("[pipeline] cleaned previous generated output")
