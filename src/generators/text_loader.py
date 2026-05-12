import os
import random
from collections import defaultdict


class TextLoader:
    """Load local text corpus files from the project data directory."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.texts = []
        self.category_texts = defaultdict(list)
        self._load_texts()

    def _load_texts(self):
        os.makedirs(self.cfg.TEXT_ROOT, exist_ok=True)

        for root, _, files in os.walk(self.cfg.TEXT_ROOT):
            category = os.path.relpath(root, self.cfg.TEXT_ROOT)
            if category == ".":
                category = "general"
            for name in sorted(files):
                if name.lower().endswith(self.cfg.TEXT_FILE_EXTENSIONS):
                    path = os.path.join(root, name)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                text = line.strip()
                                if text:
                                    self.texts.append(text)
                                    self.category_texts[category].append(text)
                    except OSError:
                        continue

    def sample(self, category=None):
        if category:
            category = category.lower()
            if category in self.category_texts and self.category_texts[category]:
                return random.choice(self.category_texts[category])

        if self.texts:
            return random.choice(self.texts)

        return None
