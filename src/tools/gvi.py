import argparse
import statsmodels.api as sm
import json
from typing import Tuple, List
import numpy as np
import pandas as pd
import os
import re
import patsy
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def get_importances(path):
    with open(args.importances, "r") as ifile:
        return json.load(ifile)


class Grouping:
    importances: dict
    _grouping: dict
    info: dict

    @property
    def ba(self):
        return self._grouping["balanced_accuracy"]

    @property
    def level(self):
        return len(self.info["group_prefix"])

    @property
    def name(self):
        return self.info["group_name"]

    def _load_grouping(self, path):
        base_name, group_info = parse_filename(path)
        self.info = group_info
        with open(path, "r") as grouping_file:
            grouping_performance = json.load(grouping_file)
        self._grouping = grouping_performance

    def _get_importances(self, importances):
        pattern_str: str = f"^{r'.*?_' * self.level}{re.escape(self.name)}.*$"
        pattern: re.Pattern = re.compile(pattern_str)

        filtered_keys = list(
            filter(lambda x: (re.match(pattern, x) is not None), importances.keys())
        )

        filtered_features = [importances[key] for key in filtered_keys]

        self.importances = dict(
            [list(pair) for pair in zip(filtered_keys, filtered_features)]
        )

    def average_importance(self, function="mean"):
        avgs = {"mean": np.mean, "median": np.median}

        return avgs[function](list(self.importances.values()))

    def __init__(self, path, importances):
        importances
        self._load_grouping(path)
        self._get_importances(importances)


def setup_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("importances", type=str, help="Dictionary of feature importances")
    ap.add_argument("groupings", nargs="+", type=str, help="Grouping performanes")
    return ap.parse_args()


def parse_filename(filename: str) -> Tuple[str, dict]:
    basename = os.path.basename(filename)
    pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<group_prefix>_*)(?P<group_name>[^_]*)(?P<group_postfix>_*)\.json"
    p: re.Pattern = re.compile(pattern)
    groups = re.match(p, basename)
    return basename, groups.groupdict()


def load_groupings(paths, importances):
    levels = {}

    for grouping_path in args.groupings:
        grouping = Grouping(grouping_path, importances)
        if grouping.level not in levels:
            levels[grouping.level] = {}
        levels[grouping.level][grouping.name] = grouping

    return levels


class Predictor:
    def predict_accuracy_by_importance(self, data, x, y):
        y, X = patsy.dmatrices(f"{y} ~ {x}", data=data, return_type="dataframe")
        model = sm.OLS(y, X)
        model_res = model.fit()
        return model_res, X

    def get_outliers(
        self,
    ):
        below = []
        above = []
        for expected, bounds, name in zip(self.expected, self.ci_bounds(), self.names):
            if expected < bounds[0]:
                below.append(name)
            elif expected > bounds[1]:
                above.append(name)

        return below, above

    def predicted_line(self):
        x_min = min(self.where)
        x_max = max(self.where)

        x_coords = [x_min, x_max]
        x_line_constant = sm.add_constant(x_coords)
        y_coords = self.p.predict(x_line_constant)
        return x_coords, y_coords

    def ci_bounds(self, alpha=0.05):
        predictor = self.p
        points = self.X
        bounds = []
        for where, actual in zip(self.where, self.expected):
            frame = predictor.get_prediction((1.0, where)).summary_frame(alpha=alpha)
            bounds.append(
                [float(frame["mean_ci_lower"]), float(frame["mean_ci_upper"])]
            )
        return bounds

    def __init__(self, data, x, y, name):
        self.names = data[name]
        self.where = data[x]
        self.expected = data[y]
        self.p, self.X = self.predict_accuracy_by_importance(data, x, y)


def plot_points(data, x, y, size, ax):
    ax.scatter(
        data[x], data[y], data[size], c=abs(data[y] - 0.5) / data[y], cmap="plasma"
    )


def label_points(data, x, y, name, outliers, ax, skip_if_more=20):
    if len(data[x]) > skip_if_more:
        return
    for cx, cy, cname in zip(data[x], data[y], data[name]):
        style = ["normal", "italic"][int(cname in outliers)]
        ax.text(
            cx,
            cy,
            cname,
            horizontalalignment="center",
            size="xx-small",
            fontstyle=style,
        ).set_path_effects(
            [
                path_effects.Stroke(linewidth=1, foreground="white", alpha=0.5),
                path_effects.Normal(),
            ]
        )


def fill_ci(x, lower, upper, ax, alpha=0.25, c_offset=0):
    ax.fill_between(
        x, lower, upper, color=plt.get_cmap("plasma")(c_offset), alpha=alpha
    )


def plot_prediction(x, y, ax, c_offset=0):
    ax.plot(x, y, color=plt.get_cmap("plasma")(c_offset))


def plot_ratios(grouping_frame, ax):
    p = Predictor(grouping_frame, "Importance", "Accuracy", "Name")
    bounds = p.ci_bounds()

    ci_frame = pd.DataFrame(
        zip(level_i, [b[0] for b in bounds], [b[1] for b in bounds]),
        columns=["Importance", "CI_Lower", "CI_Upper"],
    )
    ci_frame.sort_values("Importance", inplace=True)

    fill_ci(ci_frame["Importance"], ci_frame["CI_Lower"], ci_frame["CI_Upper"], ax)
    plot_points(grouping_frame, "Importance", "Accuracy", "Size", ax)
    plot_prediction(*p.predicted_line(), ax)
    below, above = p.get_outliers()
    print(f"Below: {', '.join(below)}")
    print(f"Above: {', '.join(above)}")
    label_points(grouping_frame, "Importance", "Accuracy", "Name", below + above, ax)


if __name__ == "__main__":
    args = setup_args()
    feature_importances = get_importances(args.importances)
    levels = load_groupings(args.groupings, feature_importances)

    fig = plt.figure(figsize=(11, 9))
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.88, bottom=0.195, left=0.06, right=0.985, hspace=0.2, wspace=0.2
    )
    subfigs = fig.subfigures(nrows=3, ncols=1)
    suptitles = ["By Input Filter", "By Feature Group", "By Feature Extraction Method"]

    for level, subfig in enumerate(subfigs):
        subfig.suptitle(suptitles[level])
        axs = subfig.subplots(nrows=1, ncols=2)
        subfig.subplots_adjust(bottom=0.195)

        for avg_index, avg in enumerate(["mean", "median"]):
            print(f"Level {level}/{avg}")

            ax = axs[avg_index]
            ax.tick_params(labelrotation=15)
            ax.grid(visible=True, linestyle="--")
            ax.set_xlabel(f"Average Importance ({avg})")
            ax.set_ylabel(f"Balanced Accuracy")

            level_i = [
                levels[level][g].average_importance(function=avg) for g in levels[level]
            ]
            level_a = [levels[level][g].ba for g in levels[level]]
            grouping_frame = pd.DataFrame(
                zip(
                    level_i,
                    level_a,
                    levels[level],
                    [len(levels[level][g].importances) for g in levels[level]],
                ),
                columns=["Importance", "Accuracy", "Name", "Size"],
            )

            plot_ratios(grouping_frame, ax)

    plt.savefig("grouping_vs_importance_new.png", dpi=600)
    plt.show()
