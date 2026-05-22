# ============================================================
# FILE: src/datasets/elpv_dataset.py
# ============================================================

from collections import Counter

import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset

from elpv_dataset.utils import load_dataset


class ELPVDataset(Dataset):

    def __init__(self, transform=None, binary_classification=True):

        # ----------------------------------------------------
        # LOAD ELPV DATASET
        # ----------------------------------------------------

        images, proba, types = load_dataset()

        self.images = images
        self.proba = proba
        self.types = types

        self.transform = transform

        self.binary_classification = binary_classification

        # ----------------------------------------------------
        # BUILD LABELS
        # ----------------------------------------------------

        self.labels = self._build_labels()

        # ----------------------------------------------------
        # BUILD IMAGE PATHS
        # ----------------------------------------------------

        self.image_paths = self._build_image_paths()

    # ========================================================
    # BUILD LABELS
    # ========================================================

    def _build_labels(self):

        labels = []

        for p in self.proba:

            # ------------------------------------------------
            # BINARY CLASSIFICATION
            # ------------------------------------------------
            #
            # 0 = NO DEFECT
            # 1 = DEFECT
            #
            # ------------------------------------------------

            if self.binary_classification:

                label = 0 if p == 0 else 1

            else:

                label = p

            labels.append(label)

        return labels

    # ========================================================
    # BUILD IMAGE PATHS
    # ========================================================

    def _build_image_paths(self):

        image_paths = []

        for idx in range(len(self.images)):

            image_paths.append(f"ELPV_image_{idx:05d}.png")

        return image_paths

    # ========================================================
    # LEN
    # ========================================================

    def __len__(self):

        return len(self.images)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        image = self.images[idx]

        label = self.labels[idx]

        image_path = self.image_paths[idx]

        # ----------------------------------------------------
        # NUMPY -> PIL
        # ----------------------------------------------------

        image = Image.fromarray(np.uint8(image))

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        image = image.convert("RGB")

        # ----------------------------------------------------
        # TRANSFORMS
        # ----------------------------------------------------

        if self.transform:

            image = self.transform(image)

        # ----------------------------------------------------
        # LABEL -> TENSOR
        # ----------------------------------------------------

        label = torch.tensor(label, dtype=torch.long)

        # ----------------------------------------------------
        # RETURN DICTIONARY
        # ----------------------------------------------------

        return {"image": image, "label": label, "path": image_path}

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    def get_statistics(self):

        class_counts = Counter(self.labels)

        total_samples = len(self.labels)

        stats = {
            "total_samples": total_samples,
            "no_defect": class_counts.get(0, 0),
            "defect": class_counts.get(1, 0),
        }

        return stats
