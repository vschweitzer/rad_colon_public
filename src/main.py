from typing import List, Tuple, Callable, Optional, Dict
import sklearn.ensemble as skensemble
import sklearn.metrics as smetrics
import csv
import pprint
import time
import logging

import test_case
import test_case_collection
import util
import classification_run_collection
import feature_filters
import logging_util
import main_functions

if __name__ == "__main__":
    logging_util.setup_logging()
    arguments = util.setup_arguments()
    tcc = test_case_collection.TestCaseCollection(
        arguments.file, arguments.radiomics_params
    )

    # main_functions.importance_filter_ramp(tcc)
    # main_functions.group_by_level(tcc, 0)
    # main_functions.group_by_level(tcc, 1)
    # main_functions.group_by_level(tcc, 2)

    main_functions.single_level_auc(tcc, arguments.filter_level)
