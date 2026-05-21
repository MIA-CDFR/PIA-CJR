# ============================================================
# FILE: src/models/densenet_model.py
# ============================================================

import torch.nn as nn

from torchvision.models import (
    densenet121,
    DenseNet121_Weights
)


def build_densenet121(

    num_classes=2,

    pretrained=True,

    dropout=0.3
):

    # --------------------------------------------------------
    # LOAD PRETRAINED MODEL
    # --------------------------------------------------------

    if pretrained:

        weights = DenseNet121_Weights.DEFAULT

    else:

        weights = None

    model = densenet121(
        weights=weights
    )

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    in_features = model.classifier.in_features

    # --------------------------------------------------------
    # REPLACE CLASSIFIER
    # --------------------------------------------------------

    model.classifier = nn.Sequential(

        nn.Dropout(p=dropout),

        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model