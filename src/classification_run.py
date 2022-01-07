from typing import List, Tuple, Callable, Optional, Dict
import sklearn.ensemble as skensemble
import sklearn.metrics as smetrics
import json
import csv
import pprint

import test_case_collection
import test_case
import feature_filters
import logging_util


class ClassificationRun:
    """
    Class meant to act as a wrapper around sklearn's RandomForestClassifier
    """

    clf: skensemble.RandomForestClassifier
    tcc: test_case_collection.TestCaseCollection

    training_cases: List[test_case.TestCase]
    verification_cases: List[test_case.TestCase]
    training_data: dict
    verification_data: dict

    predictions: List[int]

    true_positive: float
    false_positive: float
    true_negative: float
    false_negative: float

    filters: List[feature_filters.FeatureFilter]

    sample_n: bool
    sample_percent: bool

    def balanced_accuracy(self, adjust: bool = False) -> float:
        return smetrics.balanced_accuracy_score(
            self.verification_data["target"], self.predictions, adjusted=adjust
        )

    @property
    def specificity(self):
        return self.true_negative / (self.true_negative + self.false_positive)

    @property
    def sensitivity(self):
        return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def _importances_raw(self):
        return self.training_data["data_columns"], self.clf.feature_importances_

    @property
    def importances(self):
        names, values = self._importances_raw
        return list(zip(names, list(values)))

    def sample(
        self,
        choice_function=None,
        n: int = 70,
        percent: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        choice_function: None (default => default_sample_function)
        """
        if choice_function is None:
            choice_function = self.tcc.default_sample_function
        return choice_function(count=n, percent=percent, random_seed=random_seed)

    def __init__(
        self,
        tcc: test_case_collection.TestCaseCollection,
        choice_function,
        args: List = [],
        kwargs: dict = {},
        random_seed: int = 0,
        feature_filters: List[feature_filters.FeatureFilter] = [],
        sample_n: int = 70,
        sample_percent: bool = True,
    ):
        """
        * tcc - Test Case Collection
        * choice_function - Function to select Test Cases from Test Case Collection.
        * args/kwargs - args/kwargs for choice function.
        * random_seed - Seed for functions based on randomness.
        * feature_filters - Filters to exclude filters. If multiple are given, they will be chained with AND operations.
        * sample_n - How many feature to choose in the "sample" method.
        * sample_percent - Determines if sample_n is treated as percent or as an absolute amount.
        """
        self.random_seed = random_seed
        self._saved_str_dict = None
        self.filters = feature_filters
        self.clf: skensemble.RandomForestClassifier = skensemble.RandomForestClassifier(
            n_estimators=1000, n_jobs=-1, random_state=random_seed
        )
        self.tcc = tcc
        self.training_cases = self.sample(
            random_seed=random_seed, choice_function=choice_function
        )
        self.verification_cases = self.tcc.all_but(self.training_cases)

        self.training_data: dict = tcc.get_sklearn_data(
            self.training_cases, feature_filters=self.filters
        )
        self.verification_data: dict = tcc.get_sklearn_data(
            self.verification_cases, feature_filters=self.filters
        )

        self.clf.fit(self.training_data["data"], self.training_data["target"])
        self.predictions: List[int] = self.clf.predict(self.verification_data["data"])
        self.prediction_probabilities: List[float] = self.clf.predict_proba(
            self.verification_data["data"]
        )
        (
            self.true_negative,
            self.false_positive,
            self.false_negative,
            self.true_positive,
        ) = smetrics.confusion_matrix(
            self.verification_data["target"], self.predictions
        ).ravel()

    def _str_dict(self):
        if self._saved_str_dict is not None:
            return self._saved_str_dict
        dict_representation: dict = {}
        dict_representation["tcc"] = self.tcc._str_dict()
        dict_representation["feature_filters"] = [
            feature_filter._str_dict() for feature_filter in self.filters
        ]
        dict_representation["training_cases"] = [
            training_case._str_dict() for training_case in self.training_cases
        ]
        dict_representation["verification_cases"] = [
            verification_case._str_dict()
            for verification_case in self.verification_cases
        ]
        dict_representation["predictions"] = [
            int(prediction) for prediction in self.predictions
        ]
        dict_representation["prediction_probabilities"] = [
            [float(proba) for proba in prediction]
            for prediction in self.prediction_probabilities
        ]
        dict_representation["random_seed"] = self.random_seed
        return dict_representation

    def __str__(self):
        return json.dumps(self._str_dict())
