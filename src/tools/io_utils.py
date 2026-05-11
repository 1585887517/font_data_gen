import os
import cv2

from tools.logger import Logger


class IOUtils:

    @staticmethod
    def save(img, mask, idx, cfg):

        # ==================================================
        # 🚀 remove extension safely
        # ==================================================
        base_name = os.path.splitext(str(idx))[0]

        # ==================================================
        # 🚀 build save path
        # ==================================================
        img_path = os.path.join(
            cfg.OUTPUT_IMG,
            f"{base_name}{cfg.OUTPUT_IMAGE_EXT}"
        )

        mask_path = os.path.join(
            cfg.OUTPUT_MASK,
            f"{base_name}{cfg.OUTPUT_MASK_EXT}"
        )

        # ==================================================
        # 🚀 save image
        # ==================================================
        ok1 = cv2.imwrite(
            img_path,
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), cfg.IMAGE_JPEG_QUALITY]
        )

        ok2 = cv2.imwrite(
            mask_path,
            mask,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
        )

        # ==================================================
        # 🚀 logging
        # ==================================================
        if ok1 and ok2:
            pass
        else:
            Logger.error(f"[save failed] {base_name}")
