import time
import os
from configs.config import Config, PROJECT_ROOT
from generators.document_generator import DocumentGenerator
from generators.handwriting_loader import HandwritingLoader
from generators.text_loader import TextLoader
from pipeline.dataset_pipeline import DatasetPipeline
from tools.dataset_metadata import build_dataset_metadata
from tools.logger import Logger


def main():

    Logger.info("Start Dataset Factory")

    cfg = Config()

    if cfg.DATASET_MODE == "rotate":
        rotate_dir = os.path.join(PROJECT_ROOT, "output/rotate")
        if os.path.isdir(rotate_dir) and not os.listdir(rotate_dir):
            os.rmdir(rotate_dir)
        
        modes = ["printed_only", "handwriting_only", "both", "both_overlap"]
        total_duration = 0
        for mode in modes:
            Logger.info(f"[rotate] Starting {mode} mode")
            cfg.DATASET_MODE = mode
            cfg.OUTPUT_ROOT = os.path.join(PROJECT_ROOT, f"output/{mode}")
            cfg.OUTPUT_IMG = os.path.join(cfg.OUTPUT_ROOT, "images")
            cfg.OUTPUT_MASK = os.path.join(cfg.OUTPUT_ROOT, "masks")
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_ROOT, cfg.DATASET_NAME)
            
            cfg.init_dirs()
            cfg.print_config()

            pipe = DatasetPipeline(cfg, DocumentGenerator, HandwritingLoader, TextLoader)

            start_time = time.time()
            pipe.run()
            duration = time.time() - start_time
            total_duration += duration
            build_dataset_metadata(cfg)

            Logger.info(f"[rotate] {mode} mode done. Duration: {duration:.2f} seconds")
        
        Logger.info(f"[rotate] All modes done. Total duration: {total_duration:.2f} seconds")
    else:
        cfg.init_dirs()
        cfg.print_config()

        pipe = DatasetPipeline(cfg, DocumentGenerator, HandwritingLoader, TextLoader)

        start_time = time.time()
        pipe.run()
        duration = time.time() - start_time
        build_dataset_metadata(cfg)

        Logger.info(f"All done. Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
