from typing import List
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file", "-f", default="./ratio_preserving_sample_1000.csv", type=str
    )
    args = ap.parse_args()

    csv_path: str = args.file
    titles: List[str] = ["Balanced Accuracy", "Sensitivity", "Specificity"]
    data = pd.read_csv(csv_path, names=titles)
    kde = sns.displot(data, kind="kde", fill=True)
    plt.show()
