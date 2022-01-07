from typing import List
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import matplotlib

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", "-f", default="./importances.json", type=str)
    args = ap.parse_args()

    plt.rcParams.update(
        {
            "font.size": 20,
            # "font.family": "Crimson Pro",
        }
    )

    with open(args.file, "r") as input_file:
        importances = json.load(input_file)

    titles: List[str] = ["Balanced Accuracy", "Sensitivity", "Specificity"]
    kde = sns.histplot(
        data=[float(val[1]) for val in importances["weighted_importances"]],
        # kind="kde",
        color=matplotlib.cm.get_cmap("plasma")(0.3),
        fill=True,
        label="means TP",
        # bw_adjust=0.1,
        stat="count",
        binwidth=0.00001,
    )
    kde.set_xlabel("Feature Importance")
    plt.title(
        "Distribution of Weighted Importance",
    )
    plt.show()
