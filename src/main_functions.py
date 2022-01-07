from typing import List, Tuple, Callable, Optional, Dict
import sklearn.ensemble as skensemble
import sklearn.metrics as smetrics
import csv
import pprint
import time
import logging
import re

import test_case
import test_case_collection
import util
import classification_run_collection
import feature_filters
import logging_util


def single_level_auc(
    tcc: test_case_collection.TestCaseCollection,
    filter_level: int,
    iterations: int = 100,
):

    # Run without filter to determine features to filter out
    filter_clfr: classification_run_collection.ClassificationRunCollection = (
        classification_run_collection.ClassificationRunCollection(
            tcc, iterations=iterations
        )
    )
    filter_clfr.save_info("filter_clfr")

    filter_value = filter_level
    high_value_features: List[str] = [
        feature[0]
        for feature in filter_clfr.weighted_importances
        if feature[1] >= filter_value
    ]
    # Filter out low-importance features
    low_value_filter: feature_filters.FeatureFilter = feature_filters.FeatureFilter(
        feature_filters.feature_in, filter_args=[high_value_features]
    )
    # Run with filter
    # Get ROC and AUC from this
    clfr: classification_run_collection.ClassificationRunCollection = (
        classification_run_collection.ClassificationRunCollection(
            tcc, iterations=iterations, feature_filter=low_value_filter
        )
    )

    clfr.save_info(f"constant_level_{filter_level}")

    pass


def importance_filter_ramp(
    tcc: test_case_collection.TestCaseCollection,
    filter_iterations: int = 100,
    iterations: int = 100,
    filter_steps: int = 100,
):
    performance_timer_start: int = time.perf_counter_ns()
    pp = pprint.PrettyPrinter()

    logging_util.log_wrapper(
        f"Timing - Setup: {time.perf_counter_ns() - performance_timer_start} ns",
        loglevel=logging.DEBUG,
    )

    # Run without filter to determine features to filter out
    filter_clfr: classification_run_collection.ClassificationRunCollection = (
        classification_run_collection.ClassificationRunCollection(
            tcc, iterations=filter_iterations
        )
    )
    filter_clfr.save_info("filter_clfr")

    min_importance: float = 0.0
    max_importance: float = max(filter_clfr.weighted_importances_raw[1])
    filter_step_size: float = (max_importance - min_importance) / filter_steps

    logging_util.log_wrapper(
        f"Timing - Filter Classification Run: {time.perf_counter_ns() - performance_timer_start} ns",
        loglevel=logging.DEBUG,
    )

    for filter_index in range(filter_steps):
        filter_value: float = filter_step_size * filter_index
        logging_util.log_wrapper(f"Filter {filter_index} - {filter_value}")
        # Get rough list of important features
        high_value_features: List[str] = [
            feature[0]
            for feature in filter_clfr.weighted_importances
            if feature[1] >= filter_value
        ]

        # Filter out low-importance features
        low_value_filter: feature_filters.FeatureFilter = feature_filters.FeatureFilter(
            feature_filters.feature_in, filter_args=[high_value_features]
        )

        logging_util.log_wrapper(
            f"Timing - Filter {filter_index} Creation: {time.perf_counter_ns() - performance_timer_start} ns",
            loglevel=logging.DEBUG,
        )

        # Run with filter
        clfr: classification_run_collection.ClassificationRunCollection = (
            classification_run_collection.ClassificationRunCollection(
                tcc, iterations=iterations, feature_filter=low_value_filter
            )
        )

        logging_util.log_wrapper(
            f"Timing - Filter {filter_index} Main Run: {time.perf_counter_ns() - performance_timer_start} ns",
            loglevel=logging.DEBUG,
        )

        pp.pprint("Balanced accuracy: " + pp.pformat(clfr.balanced_accuracy))
        pp.pprint("Sensitivity: " + pp.pformat(clfr.sensitivity))
        pp.pprint("Specificy: " + pp.pformat(clfr.specificity))
        # pp.pprint("Importances: ")
        # pp.pprint(clfr.weighted_importances)

        logging_util.log_wrapper(
            f"Timing - Filter {filter_index} Info Dump: {time.perf_counter_ns() - performance_timer_start} ns",
            loglevel=logging.DEBUG,
        )

        clfr.save_info(f"{filter_value}")

        logging_util.log_wrapper(
            f"Timing - Filter {filter_index} Info Save: {time.perf_counter_ns() - performance_timer_start} ns",
            loglevel=logging.DEBUG,
        )


def group_by_level(
    tcc: test_case_collection.TestCaseCollection,
    level: int,
    global_filter: feature_filters.FeatureFilter = feature_filters.all_features,
):
    def _extract_level(feature: str, level: int):
        return feature.split("_")[level]

    def _unique_by_level(features: List[str], level: int):
        all_by_level = [_extract_level(feature, level) for feature in features]
        return set(all_by_level)

    def _drop_diagnostics(features: List[str]):
        return [
            feature for feature in features if not feature.startswith("diagnostics_")
        ]

    clfr_cs: List[classification_run_collection.ClassificationRunCollection] = []
    features = list(tcc.test_cases[0].feature_vector.keys())

    for group in _unique_by_level(_drop_diagnostics(features), level):
        pattern_str: str = f"^{r'.*?_' * level}{re.escape(group)}.*$"
        # pattern: re.Pattern = re.compile(pattern_str)
        identifier: str = f"{'_'*level}{group}{'_' * (features[0].count('_') - level)}"
        print(pattern_str)
        f: feature_filters.FeatureFilter = feature_filters.FeatureFilter(
            feature_filters.feature_regex, filter_kwargs={"pattern": pattern_str}
        )
        clfr: classification_run_collection.ClassificationRunCollection = (
            classification_run_collection.ClassificationRunCollection(
                tcc, feature_filter=f
            )
        )
        clfr.save_info(identifier)


if __name__ == "__main__":
    logging_util.setup_logging()
    performance_timer_start: int = time.perf_counter_ns()
    arguments = util.setup_arguments()
    tcc = test_case_collection.TestCaseCollection(
        arguments.file, arguments.radiomics_params
    )
