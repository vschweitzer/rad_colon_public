from typing import List, Dict, Any, Tuple
import functools
import pickle
import argparse
import pprint
import json
import re


class FeatureFilter:
    def __init__(
        self, filter_function, filter_args: list = [], filter_kwargs: dict = {}
    ):
        """
        Helper class to simplify feature filter management.
        """
        self.filter_function = filter_function
        self.filter_args = filter_args
        self.filter_kwargs = filter_kwargs

    def evaluate(self, feature):
        return self.filter_function(feature, *self.filter_args, **self.filter_kwargs)

    def run(self, features: dict) -> dict:
        return filter_features(
            features,
            self.filter_function,
            args=self.filter_args,
            kwargs=self.filter_kwargs,
        )

    def _str_dict(self):
        dict_representation: dict = {}
        dict_representation["filter_function_str"] = self.filter_function.__name__
        dict_representation["filter_args"] = self.filter_args
        dict_representation["filter_kwargs"] = self.filter_kwargs
        return dict_representation

    def __str__(self):
        return json.dumps(self._str_dict())


def filter_features(
    feature_vector: Dict[str, Any], filter_function, args: list = [], kwargs: dict = {}
):
    feature_list: List[Tuple[str, Any]] = list(feature_vector.items())
    # filtered_features: List[Tuple[str, Any]] = [
    #     key for key in feature_list if filter_function(key[0], *args, **kwargs)
    # ]
    filtered_features: List[Tuple[str, Any]] = []
    for feature in feature_list:
        if filter_function(feature[0], *args, **kwargs):
            filtered_features.append(feature)
    return dict(filtered_features)


def all_features(feature: str, args: list = [], kwargs: dict = {}) -> bool:
    """
    Returns (true for) all filters
    """
    return True


def feature_startswith(
    feature: str, start_string: str, args: list = [], kwargs: dict = {}
) -> bool:
    return feature.startswith(start_string)


def feature_regex(feature: str, pattern: str):
    return re.fullmatch(pattern, feature) is not None


def feature_in(feature: str, include_list: List[str]):
    return feature in include_list


def feature_not_in(feature: str, exclude_list: List[str]):
    return not feature_in(feature, exclude_list)


def filter_and(feature, filters: List[FeatureFilter]):
    return all([filter_function.evaluate(feature) for filter_function in filters])


def filter_or(feature, filters: List[FeatureFilter]):
    return any([filter_function.evaluate(feature) for filter_function in filters])


filter_dummy: FeatureFilter = FeatureFilter(all_features)
"""
A filter not filtering anything.
"""

if __name__ == "__main__":

    def setup_args():
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--type",
            "-t",
            type=str,
            help="Choose between pickle and JSON file.",
            choices=["json", "pickle"],
        )
        ap.add_argument(
            "file",
            type=str,
            help="Pickled or JSON feature save file to load features from.",
        )
        args = ap.parse_args()
        return args

    def setup_pprint():
        pp = pprint.PrettyPrinter(indent=4)
        return pp

    args: argparse.Namespace = setup_args()
    pp: pprint.PrettyPrinter = setup_pprint()
    # f: FeatureFilter = FeatureFilter(feature_startswith, filter_args=["original"])
    pattern: re.Pattern = re.compile(r"^.*?_glcm_.*$")
    f: FeatureFilter = FeatureFilter(feature_regex, filter_kwargs={"pattern": pattern})
    with open(args.file, "rb") as input_file:
        if args.type == "pickle":
            features = pickle.load(input_file)
        elif args.type == "json":
            features = json.load(input_file)
        else:
            raise NotImplementedError(f'Save type "{args.type}" not supported.')
    pp.pprint("Original:")
    pp.pprint(features)
    pp.pprint("Filtered:")
    pp.pprint(f.run(features))
