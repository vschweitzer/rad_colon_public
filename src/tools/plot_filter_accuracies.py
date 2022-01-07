import json
import argparse
import re
from typing import Optional, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as smetrics
import matplotlib


def filter_run_info(runs: dict):
    balanced_accuracies: List[float] = []
    feature_count: int = len(
        runs["classification_runs"][0]["feature_filters"][0]["filter_args"][0]
    )
    for run in runs["classification_runs"]:
        balanced_accuracies.append(run_info(run))

    balanced_accuracy_variance: float = np.var(balanced_accuracies)
    balanced_accuracy_average: float = np.average(balanced_accuracies)
    var_min = balanced_accuracy_average - balanced_accuracy_variance / 2
    var_max = balanced_accuracy_average + balanced_accuracy_variance / 2
    print(
        f"Mean: {np.mean(balanced_accuracies):<013.10} / Var: {np.var(balanced_accuracies):<013.10}"
    )
    return [
        balanced_accuracy_average,
        balanced_accuracy_variance,
        feature_count,
        var_min,
        var_max,
    ]


def run_info(run: dict):
    predictions: List[int] = run["predictions"]
    targets: List[int] = [
        int(test_case["pcr"]) for test_case in run["verification_cases"]
    ]
    balanced_accuracy = smetrics.balanced_accuracy_score(targets, predictions)
    return balanced_accuracy


def print_bar(value: float, val_min: float, val_max: float, bars: int = 100):
    value_scaled = value - val_min
    max_scaled = val_max - val_min
    bar_count = int((value_scaled / max_scaled) * bars)
    print("=" * bar_count)


graph_data: list = []

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "runs",
        nargs="+",
        type=str,
        help="Classification Run Info files. Expected to be named <timestamp>_<filter_level>.json.",
    )
    args = ap.parse_args()

    plt.rcParams.update(
        {
            "font.size": 20,
            # "font.family": "Crimson Pro",
        }
    )

    accuracy_max: Optional[float] = None
    accuracy_min: Optional[float] = None
    runs: dict = {}
    for run in args.runs:
        pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.*?)\.json"
        filename: str = run
        # Hardcoded for expected name pattern
        match = re.search(pattern, filename)

        if match is None:
            print(f'Could not parse filename "{filename}"')
            continue

        try:
            timestamp: str = match.group(1)
            filter_level_str: str = match.group(2)
            filter_level: float = float(filter_level_str)
        except ValueError as ex:
            print(f"Could not convert filename: {ex}")
            continue
        with open(filename, "r") as input_file:
            # print(f"Loading {filename}...")
            try:
                runs[filter_level] = json.load(input_file)
            except json.decoder.JSONDecodeError:
                continue

    for filter_level in sorted(runs):
        graph_data.append([filter_level] + filter_run_info(runs[filter_level]))
    graph_frame: pd.DataFrame = pd.DataFrame(
        graph_data,
        columns=[
            "Filter Level",
            "Balanced Accuracy Average",
            "Balanced Accuracy Variance",
            "Feature Count",
            "Variance Min",
            "Variance Max",
        ],
    )
    print(graph_frame)
    fig, ax = plt.subplots()
    second_ax = ax.twinx()
    second_ax.set_yscale("log")
    average_plot = ax.errorbar(
        graph_frame["Filter Level"],
        graph_frame["Balanced Accuracy Average"],
        yerr=graph_frame["Balanced Accuracy Variance"],
        color=matplotlib.cm.get_cmap("plasma")(0.75),
        capsize=2.5,
    )
    # average_plot.set_label("Average of Balanced Accuracy")
    # variance_plot = ax.fill_between(
    #     graph_frame["Filter Level"],
    #     graph_frame["Variance Min"],
    #     graph_frame["Variance Max"],
    #     color="#FFB000AA",
    # )
    # variance_plot.set_label("Variance of Balanced Accuracy")
    feature_count_plot = second_ax.plot(
        graph_frame["Filter Level"],
        graph_frame["Feature Count"],
        color=matplotlib.cm.get_cmap("plasma")(0.0),
    )
    # feature_count_plot.set_label("Feature Count")

    plt.title("Balanced Accuracy over Feature Importance Filter Level")
    second_ax.legend(["Feature Count"], loc="upper right")
    ax.legend(
        ["Average of Balanced Accuracy", "Variance of Balanced Accuracy"],
        loc="lower left",
    )

    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel(
        "Filter Level (Only features of Importance >= Feature Level are used)"
    )
    second_ax.set_ylabel("Feature Count")

    ax.grid(color=matplotlib.cm.get_cmap("plasma")(0.75))
    second_ax.grid(linestyle="--", color=matplotlib.cm.get_cmap("plasma")(0.00))

    plt.show()
