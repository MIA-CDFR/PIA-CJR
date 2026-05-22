# ============================================================
# FILE: src/models/efficientnet_model.py
# ============================================================

import torch.nn as nn

from torchvision.models import (

    efficientnet_b0,

    EfficientNet_B0_Weights
)


def build_efficientnet_b0(

    num_classes=2,

    pretrained=True,

    dropout=0.30
):

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    if pretrained:

        weights = (
            EfficientNet_B0_Weights.DEFAULT
        )

    else:

        weights = None

    model = efficientnet_b0(
        weights=weights
    )

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    in_features = (
        model.classifier[1].in_features
    )

    # --------------------------------------------------------
    # REPLACE CLASSIFIER
    # --------------------------------------------------------

    model.classifier = nn.Sequential(

        nn.Dropout(
            p=dropout
        ),

        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model