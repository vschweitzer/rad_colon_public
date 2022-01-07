from typing import List, Tuple, Callable, Optional, Dict
import sklearn.ensemble as skensemble
import sklearn.metrics as smetrics
import json
import csv
import pprint
import datetime
import hashlib
import numpy
import logging

import test_case_collection
import test_case
import feature_filters
import logging_util
import classification_run


class ClassificationRunCollection:
    choice_function: Optional[Callable]
    classification_runs: List[classification_run.ClassificationRun]
    iterations: int

    def __init__(
        self,
        tcc: test_case_collection.TestCaseCollection,
        iterations: int = 100,
        choice_function=None,
        train_test_split: int = 70,
        split_percent: bool = True,
        feature_filter: feature_filters.FeatureFilter = feature_filters.filter_dummy,
    ):
        self._weighted_importances = None
        self._weighted_importances_raw = None
        self._saved_str_dict = None
        self.classification_runs = []
        self.choice_function = choice_function
        self.iterations = iterations
        self.train_test_split = train_test_split
        self.split_percent = split_percent

        for iteration in range(iterations):
            logging_util.log_wrapper(
                f"Classification Run {iteration:03}", logging.DEBUG
            )
            print(".", end="", flush=True)
            self.classification_runs.append(
                classification_run.ClassificationRun(
                    tcc,
                    choice_function,
                    random_seed=iteration,
                    feature_filters=[feature_filter],
                )
            )
        print("")

    def _member_average(self, key: str):
        return (
            sum(
                [
                    getattr(classification_run, key)
                    for classification_run in self.classification_runs
                ]
            )
            / self.iterations
        )

    def _method_average(self, key: str):
        return (
            sum(
                [
                    getattr(classification_run, key)()
                    for classification_run in self.classification_runs
                ]
            )
            / self.iterations
        )

    @property
    def balanced_accuracy(self):
        """
        Returns average over all balanced accuracies (not adjusted).
        """

        return self._method_average("balanced_accuracy")

    @property
    def sensitivity(self):
        """
        Returns average over all sensitivities.
        """
        return self._member_average("sensitivity")

    @property
    def specificity(self):
        """
        Returns average over all balanced specificities.
        """
        return self._member_average("specificity")

    @property
    def weighted_importances_raw(self):
        """
        Returns importance of features weighted by the balanced accuracy of each
        classification run.
        """
        # feature_importances: List[Tuple[str, float]] = []
        if self._weighted_importances_raw is None:
            feature_names: List[str] = None
            feature_values = None
            first_run: bool = False
            for classification_run in self.classification_runs:
                (
                    run_feature_names,
                    run_feature_values,
                ) = classification_run._importances_raw
                run_feature_values_scaled = (
                    run_feature_values
                    * classification_run.balanced_accuracy(adjust=True)
                    / self.iterations
                )
                if feature_names is None:
                    feature_names = run_feature_names
                if feature_values is None:
                    feature_values = run_feature_values_scaled

                if feature_names != run_feature_names:
                    raise IndexError(
                        f"Unequal feature name: {feature_importances[0]} != {feature[0]}"
                    )

                feature_values += run_feature_values_scaled
            self._weighted_importances_raw = (feature_names, feature_values)
        return self._weighted_importances_raw

    @property
    def weighted_importances(self):
        return list(
            zip(
                self.weighted_importances_raw[0], list(self.weighted_importances_raw[1])
            )
        )

    def _str_dict(self):
        if self._saved_str_dict is not None:
            return self._saved_str_dict
        dict_representation: dict = {}
        dict_representation["classification_runs"] = [
            cr._str_dict() for cr in self.classification_runs
        ]
        dict_representation["balanced_accuracy"] = self.balanced_accuracy
        dict_representation["sensitivity"] = self.sensitivity
        dict_representation["specificity"] = self.specificity
        dict_representation["weighted_importances"] = self.weighted_importances
        dict_representation["iterations"] = self.iterations
        self._saved_str_dict = dict_representation
        return dict_representation

    def __str__(self):
        str_dict: dict = self._str_dict()
        return json.dumps(str_dict)

    def save_info(self, name: Optional[str] = "clfr"):
        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # hasher = hashlib.sha256()
        # hasher.update(self.__str__().encode("utf-8"))
        # save_info_hash = hasher.digest().hex()
        name = f"{timestamp}_{name}.json"
        with open(name, "w") as save_info_file:
            json.dump(self._str_dict(), save_info_file)
