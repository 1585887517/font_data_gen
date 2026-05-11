import time

from configs.config import Config
from generators.document_generator import DocumentGenerator
from generators.handwriting_loader import HandwritingLoader
from pipeline.dataset_pipeline import DatasetPipeline
from tools.dataset_metadata import build_dataset_metadata
from tools.logger import Logger


def main():

    Logger.info("Start Dataset Factory")

    cfg = Config()
    cfg.init_dirs()
    cfg.print_config()

    pipe = DatasetPipeline(cfg, DocumentGenerator, HandwritingLoader)

    start_time = time.time()
    pipe.run()
    duration = time.time() - start_time
    build_dataset_metadata(cfg)

    Logger.info(f"All done. Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
