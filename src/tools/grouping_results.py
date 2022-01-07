import argparse
import re
import os
from typing import Tuple
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import sklearn.metrics as smetrics


def setup_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files",
        nargs="+",
        type=str,
        help="CLFR save files [json]. Naming scheme should follow <timestamp>_<grouping>, where the position of <grouping> is determined by the surrounding underscores.",
    )

    return ap.parse_args()


def parse_filename(filename: str) -> Tuple[str, dict]:
    basename = os.path.basename(filename)
    pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<group_prefix>_*)(?P<group_name>[^_]*)(?P<group_postfix>_*)\.json"
    p: re.Pattern = re.compile(pattern)
    groups = re.match(p, basename)
    return basename, groups.groupdict()


def accuracy_list(clfr):
    bas = []
    for run in clfr["classification_runs"]:
        target = [int(case["pcr"]) for case in run["verification_cases"]]
        predictions = run["predictions"]
        ba = smetrics.balanced_accuracy_score(target, predictions, adjusted=False)
        bas.append(ba)
    return bas


def plot_by_level_sns(clfrs, titles, variance):
    def _fontsize(count: int, min_count: int, base_size: int = 12):
        factor: float = (min_count / count + 1) / 2
        return base_size * factor

    scaling_factor = 1.5

    min_count = min([len(clfrs[level].keys()) for level in clfrs])

    # figs, axes = plt.subplots(ncols=len(clfrs.keys()))
    figs = []
    axes = []
    for _ in clfrs:
        fig, ax = plt.subplots()
        figs.append(fig)
        axes.append(ax)

    for level, groups in enumerate(clfrs):
        accuracy_data = [
            (group[0], clfrs[groups][group[0]]["balanced_accuracy"], group[1])
            for group in zip(clfrs[groups], variance)
        ]
        accuracy_frame = pd.DataFrame(
            accuracy_data,
            columns=["Group Name", "Balanced Accuracy", "Balanced Accuracy Variance"],
        )

        accuracy_frame.sort_values("Balanced Accuracy", inplace=True)

        print(accuracy_frame)

        b = sns.barplot(
            x=accuracy_frame["Balanced Accuracy"],
            y=accuracy_frame["Group Name"],
            orient="h",
            ax=axes[level],
            palette="plasma",
            xerr=accuracy_frame["Balanced Accuracy Variance"],
        )
        axes[level].set_title(titles[level])
        axes[level].set_xlim([0, 0.7])
        axes[level].axvline(x=0.5, color="#F0F0F080")
        b.set_yticklabels(
            b.get_yticklabels(),
            rotation=30,
            fontsize=_fontsize(
                accuracy_frame["Group Name"].size,
                min_count,
                base_size=12 * scaling_factor,
            ),
        )
        # figs[level].set_figwidth(4.5 * scaling_factor)
        figs[level].set_figheight(16 * scaling_factor)
        # figs[level].suptitle("Balanced Accuracy by Feature Grouping")
        figs[level].tight_layout()
        figs[level].savefig(f"grouping_accuracies_{level}.png", pad_inches=0.1, dpi=300)
    # plt.show()


if __name__ == "__main__":
    args = setup_args()

    clfrs: dict = {}
    variance = []

    for filename in args.files:
        basename, name_info = parse_filename(filename)
        group_level = len(name_info["group_prefix"])
        if group_level not in clfrs:
            clfrs[group_level] = {}
        with open(filename, "r") as input_file:
            clfrs[group_level][name_info["group_name"]] = json.load(input_file)
        al = accuracy_list(clfrs[group_level][name_info["group_name"]])
        sns.histplot(data=al)
        plt.show()
        variance.append(np.var(al))

        # print(
        #     f"{group_level} - {name_info['group_name']:48} - {clfrs[group_level][name_info['group_name']]['balanced_accuracy']}"
        # )

    plot_by_level_sns(
        clfrs,
        [
            "Grouping by Image Filter",
            "Grouping by Feature Class",
            "Grouping by Feature Name",
        ],
        variance,
    )
