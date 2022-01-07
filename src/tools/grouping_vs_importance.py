import json
import argparse
import os
import re
from typing import Tuple, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects


import statsmodels.api as sm

# https://stackoverflow.com/questions/22852244/how-to-get-the-numerical-fitting-results-when-plotting-a-regression-in-seaborn/22852265#22852265
def simple_regplot(
    x, y, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None
):
    """Draw a regression line with error interval."""
    ax = plt.gca() if ax is None else ax

    # calculate best-fit line and interval
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit).fit()

    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)

    # draw the fit line and error interval
    ci_kws = {} if ci_kws is None else ci_kws
    ax.fill_between(
        eval_x[:, 1],
        pred.predicted_mean - n_std * pred.se_mean,
        pred.predicted_mean + n_std * pred.se_mean,
        alpha=0.5,
        **ci_kws,
    )
    line_kws = {} if line_kws is None else line_kws
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)

    # draw the scatterplot
    scatter_kws = {} if scatter_kws is None else scatter_kws
    ax.scatter(x, y, **scatter_kws)

    return fit_results


def parse_filename(filename: str) -> Tuple[str, dict]:
    basename = os.path.basename(filename)
    pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<group_prefix>_*)(?P<group_name>[^_]*)(?P<group_postfix>_*)\.json"
    p: re.Pattern = re.compile(pattern)
    groups = re.match(p, basename)
    return basename, groups.groupdict()


def setup_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("importances", type=str, help="Dictionary of feature importances")
    ap.add_argument("groupings", nargs="+", type=str, help="Grouping performanes")
    return ap.parse_args()


def get_average_importance(
    level: int, name: str, importances: List[float], avg: str = "mean"
):
    avg_functions = {"mean": np.mean, "median": np.median}
    pattern_str: str = f"^{r'.*?_' * level}{re.escape(name)}.*$"
    pattern: re.Pattern = re.compile(pattern_str)

    # print(pattern_str)

    filtered_keys = filter(
        lambda x: (re.match(pattern, x) is not None), importances.keys()
    )

    filtered_features = [importances[key] for key in filtered_keys]

    return avg_functions[avg](filtered_features), len(filtered_features)


def plot_ratios(
    ratios,
    level: int,
    avg: str = "mean",
    subplot=None,
    title=None,
):
    ratio_data = []
    for group in ratios[level]:
        ratio_data.append(
            (
                group,
                ratios[level][group][avg],
                ratios[level][group]["balanced_accuracy"],
                ratios[level][group]["feature_count"],
            )
        )
    avg_key = f"Average Importance ({avg})"
    ratio_frame = pd.DataFrame(
        ratio_data,
        columns=["Grouping Name", avg_key, "Balanced Accuracy", "Feature Count"],
    )

    # reg_data = simple_regplot(
    #     ratio_frame[avg_key],
    #     ratio_frame["Balanced Accuracy"],
    #     n_std=2,
    #     n_pts=100,
    #     ax=subplot,
    #     scatter_kws={
    #         "s": ratio_frame["Feature Count"],
    #         "cmap": "plasma",
    #         "c": abs(ratio_frame["Balanced Accuracy"] - 0.5) / ratio_frame[avg_key],
    #         "color": None,
    #         "alpha": 0.75,
    #     },
    #     line_kws=None,
    #     ci_kws=None,
    # )

    regp = sns.regplot(
        data=ratio_frame,
        x=avg_key,
        y="Balanced Accuracy",
        scatter_kws={
            "s": ratio_frame["Feature Count"],
            "cmap": "plasma",
            "c": abs(ratio_frame["Balanced Accuracy"] - 0.5) / ratio_frame[avg_key],
            "color": None,
        },
        color=matplotlib.cm.get_cmap("plasma")(0.0),
        # line_kws={"palette": "plasma"},
        ax=subplot,
        ci=95,
    )
    # regp.set_xticklabels(regp.get_xticklabels(), rotation=30)
    if title is not None:
        subplot.title(title)
    vertical_offset = 2 * np.mean(ratio_frame.loc[:, avg_key])
    if len(ratio_frame) < 20:
        for entry in ratio_frame.itertuples(index=False):
            subplot.text(
                entry[1],
                entry[2] + vertical_offset,
                entry[0],
                horizontalalignment="center",
                size="xx-small",
            ).set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="white", alpha=0.5),
                    path_effects.Normal(),
                ]
            )
    # print(reg_data.summary())

    x = ratio_frame[avg_key]
    y = ratio_frame["Balanced Accuracy"]
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit).fit()

    return ratio_frame, fit_results
    # plt.show()


if __name__ == "__main__":
    args = setup_args()
    grouping_performances = {}
    ratios = {}

    avg_functions = ["mean", "median"]

    with open(args.importances, "r") as ifile:
        importances = json.load(ifile)

    for grouping in args.groupings:
        base_name, group_info = parse_filename(grouping)
        group_name = group_info["group_name"]
        group_index = len(group_info["group_prefix"])

        if group_index not in grouping_performances:
            grouping_performances[group_index] = {}
        with open(grouping, "r") as grouping_file:
            grouping_performances[group_index][group_name] = json.load(grouping_file)

        if group_index not in ratios:
            ratios[group_index] = {}
        ratios[group_index][group_name] = {}
        for avg in avg_functions:
            (
                ratios[group_index][group_name][avg],
                ratios[group_index][group_name]["feature_count"],
            ) = get_average_importance(group_index, group_name, importances, avg=avg)

        ratios[group_index][group_name]["balanced_accuracy"] = grouping_performances[
            group_index
        ][group_name]["balanced_accuracy"]

    fig = plt.figure(figsize=(11, 9))
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.88, bottom=0.195, left=0.06, right=0.985, hspace=0.2, wspace=0.2
    )
    subfigs = fig.subfigures(nrows=3, ncols=1)
    suptitles = ["By Input Filter", "By Feature Group", "By Feature Extraction Method"]
    # fig, ax = plt.subplots(nrows=3, ncols=2)

    for level, subfig in enumerate(subfigs):
        subfig.suptitle(suptitles[level])

        level_frame = None

        axs = subfig.subplots(nrows=1, ncols=2)
        subfig.subplots_adjust(bottom=0.195)
        for j, avg in enumerate(avg_functions):
            axes = axs[j]
            axes.tick_params(labelrotation=15)
            axes.grid(visible=True, linestyle="--")
            new_frame, fit_results = plot_ratios(ratios, level, avg=avg, subplot=axes)
            print(f"level {level} / avg {avg}")
            print(fit_results.summary())
            if level_frame is None:
                level_frame = new_frame
            else:
                cols_to_use = new_frame.columns.difference(level_frame.columns)
                level_frame = pd.merge(
                    level_frame,
                    new_frame[cols_to_use],
                    left_index=True,
                    right_index=True,
                    how="outer",
                )
                # level_frame.join(new_frame)
            print("+" * 50)
        level_frame.to_csv(f"accuracy_vs_importance_level_{level}.csv", index=False)
        print("/" * 50)

    # Doesn't work :c
    # fig.suptitle(
    #     "Balanced Accuracy in relation to Average Importance", fontweight="bold"
    # )
    plt.savefig("grouping_vs_importance.png", dpi=600)
    plt.show()
