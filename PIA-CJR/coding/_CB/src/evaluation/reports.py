# src/evaluation/reports.py

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


def save_classification_report(
    y_true, y_pred, class_names, txt_output_path, csv_output_path
):

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    txt_output_path = Path(txt_output_path)
    csv_output_path = Path(csv_output_path)

    txt_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    df = pd.DataFrame(report_dict).transpose()

    df.to_csv(csv_output_path)

    return report_dict
