from typing import List, Optional, Any, Dict
import random
import radiomics
import logging
import csv
import os
import copy

import logging_util
import test_case
import feature_filters


class TestCaseCollection:
    test_cases_positive: List[test_case.TestCase]
    test_cases_negative: List[test_case.TestCase]
    extractor: radiomics.featureextractor.RadiomicsFeatureExtractor
    extractor_config: str
    random_seed: int

    def _str_dict(self):
        if self._saved_str_dict is not None:
            return self._saved_str_dict
        dict_representation: dict = {}
        dict_representation["test_cases"] = [
            test_case._str_dict() for test_case in self.test_cases
        ]
        dict_representation["extractor_config_path"] = [self.extractor_config]
        dict_representation["random_seed"] = self.random_seed
        self._saved_str_dict = dict_representation
        return dict_representation

    def __str__(self):
        return json.dumps(self._str_dict())

    @property
    def test_cases(self):
        return self.test_cases_positive + self.test_cases_negative

    def __init__(
        self,
        csv_path: str,
        extractor_config: str,
        random_seed: int = 0,
        filters: List[feature_filters.FeatureFilter] = [],
    ):
        """
        Loads collection of test cases via CSV file.
        """
        logging_util.setup_logging()

        self._saved_str_dict = None

        self.random_seed = random_seed

        self.test_cases_positive = []
        self.test_cases_negative = []

        self.extractor_config = extractor_config
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
            self.extractor_config
        )

        try:
            for test_case_data in self._read_in_csv(csv_path):
                # Disable multithreading until I get it working
                constructor_args: List[Any] = list(test_case_data) + [self.extractor]

                logging_util.log_wrapper(
                    f"Creating test case with args {constructor_args}", logging.DEBUG
                )

                self.add_test_case(*constructor_args)

            # for test_case_data in self._read_in_csv(csv_path):
            #     constructor_args: List[Any] = list(test_case_data) + [self.extractor]

            #     logging_util.log_wrapper(
            #         f"Creating test case with args {constructor_args}", logging.DEBUG
            #     )

            #     constructor_thread: threading.Thread = threading.Thread(
            #         target=self.add_test_case,
            #         args=constructor_args,
            #         name=os.path.basename(constructor_args[0]),
            #     )
            #     constructor_thread.start()
            #     constructor_threads.append(constructor_thread)
            # for constructor_thread in constructor_threads:
            #     if constructor_thread.is_alive():
            #         constructor_thread.join()
            # logging_util.log_wrapper(f"All constructor threads joined", logging.WARNING)
        except Exception as ex:
            logging_util.log_wrapper(
                f"An error occured while executing constructor threads: {ex}",
                logging.CRITICAL,
            )

        logging_util.log_wrapper(f"All constructor threads joined", logging.DEBUG)

    def _read_in_csv(
        self,
        csv_path: str,
        input_file_ending: str = ".nii.gz",
        annotation_suffix: str = "A",
        pcr_true: str = "pCR",
        pcr_false: str = "non-pCR",
        ignore_case: bool = True,
    ):
        """
        Parses each line of a test case collection csv. Returns
        a tuple containing the scan path, annotation path and pCr
        of each entry.
        Entry seperator is assumed to be \n.
        """
        base_dir: str
        base_dir = os.path.dirname(csv_path)
        with open(csv_path, newline="") as input_file:
            csv_reader = csv.reader(input_file)
            for line in csv_reader:
                scan_id: str
                pCr_string: str
                pCr: bool
                annotation_path: str
                scan_path: str
                scan_id, pCr_string = line
                scan_id = scan_id.strip()
                pCr_string = pCr_string.strip()

                scan_path = os.path.join(base_dir, scan_id + input_file_ending)
                annotation_path = os.path.join(
                    base_dir, scan_id + annotation_suffix + input_file_ending
                )

                if ignore_case:
                    pCr_string = pCr_string.lower()
                    pcr_true = pcr_true.lower()
                    pcr_false = pcr_false.lower()

                if pCr_string == pcr_true:
                    pCr = True
                elif pCr_string == pcr_false:
                    pCr = False
                else:
                    logging_util.log_wrapper(
                        f"{csv_reader.line_num}: Invalid value for pCr: {pCr_string}. Accepted values are {pcr_true}/{pcr_false}",
                        logging.ERROR,
                    )
                    continue
                yield (scan_path, annotation_path, pCr)

    def add_test_case(
        self,
        scan_path: str,
        annotation_path: str,
        pCr: bool,
        extractor: radiomics.featureextractor.RadiomicsFeatureExtractor,
        suffix_reconstructed: str = ".inverse_dist.nii.gz",
        feature_save_dir: str = "./savefiles",
    ):
        try:
            extractor = copy.deepcopy(extractor)

            # WARNING: This overwrites the argument. Only here for debugging purposes!
            # extractor = copy.deepcopy(self.extractor)

            tc: test_case.TestCase = test_case.TestCase(
                scan_path,
                annotation_path,
                pCr,
                extractor,
                suffix_reconstructed=suffix_reconstructed,
                feature_save_dir=feature_save_dir,
            )

            if pCr:
                # pass
                # self.lock_positive.acquire()
                self.test_cases_positive.append(tc)
                # self.lock_positive.release()
            else:
                # pass
                # self.lock_negative.acquire()
                self.test_cases_negative.append(tc)
                # self.lock_negative.release()
        except Exception as ex:
            logging_util.log_wrapper(f"Could not load test case: {ex}", logging.ERROR)
            return

    def all_but(self, but: List[test_case.TestCase], strict: bool = True):
        """
        Return all test cases but the ones included in the argument.
        If strict is true and a test case is not included in the
        test_cases member, an exception is raised.
        """
        to_return: List[test_case.TestCase] = list(self.test_cases)
        for test_case in but:
            if test_case not in to_return and strict:
                raise IndexError(f"Test case {test_case} is not part of test_cases")

            to_return.remove(test_case)
        return to_return

    def equal_sample(
        self,
        count: int,
        percent: bool = True,
        random_seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Return count of each test case type.
        If percent is true, return <count>% of
        the type with fewer elements.
        """
        if random_seed is None:
            random_seed = self.random_seed
        random_generator = random.Random(random_seed)

        min_by_type: int = min(
            len(self.test_cases_positive), len(self.test_cases_negative)
        )
        if percent:
            actual_count: int = (min_by_type * count) // 100
        else:
            actual_count = count

        if actual_count > min_by_type:
            raise ValueError(
                f"Cannot select more than {min_by_type} test cases from each type. {actual_count} requested"
            )

        return random_generator.sample(
            self.test_cases_positive, actual_count
        ) + random_generator.sample(self.test_cases_negative, actual_count)

    def ratio_preserving_sample(
        self, percent, random_seed: Optional[int] = None, *args, **kwargs
    ):
        """
        Returns percent% of each test case type.
        """
        if random_seed is None:
            random_seed = self.random_seed
        random_generator: random.Random = random.Random(random_seed)

        return random_generator.sample(
            self.test_cases_positive, (len(self.test_cases_positive) * percent) // 100
        ) + random_generator.sample(
            self.test_cases_negative, (len(self.test_cases_negative) * percent) // 100
        )

    def default_sample_function(
        self,
        count: int,
        percent: bool = True,
        random_seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        return self.equal_sample(count, percent=percent, random_seed=random_seed)

    def get_sklearn_data(
        self,
        test_cases: List[test_case.TestCase],
        feature_filters: List[feature_filters.FeatureFilter] = [],
    ) -> dict[str, list[Any]]:
        sklearn_dict: Dict[str, Any] = {}
        sklearn_dict["DESC"] = [
            f"Data extracted from {[str(tc) for tc in test_cases]}."
        ]
        sklearn_dict["target"] = [int(tc.pcr) for tc in test_cases]
        sklearn_dict["paths"] = [
            [tc.scan_path, tc.annotation_path] for tc in test_cases
        ]
        sklearn_dict["diagnostics"] = []
        sklearn_dict["diagnostics_columns"] = None
        sklearn_dict["data"] = []
        sklearn_dict["data_columns"] = None
        sklearn_dict["paths"] = []

        for tc in test_cases:
            # Separate features from diagnostic infos
            features = []
            feature_names = []
            diagnostics = []
            diagnostics_names = []

            # print(tc.image_path, tc.annotation_path, tc.pcr)
            sklearn_dict["paths"].append([tc.scan_path, tc.annotation_path])

            features_to_convert = tc.feature_vector
            for feature_filter in feature_filters:
                features_to_convert = feature_filter.run(features_to_convert)

            for feature_name in features_to_convert:
                # Is this diagnostics info?
                if feature_name.split("_")[0] == "diagnostics":
                    diagnostics_names.append(feature_name)
                    diagnostics.append(tc.feature_vector[feature_name])
                else:
                    feature_value = tc.feature_vector[feature_name].tolist()
                    feature_names.append(feature_name)
                    features.append(feature_value)

            if sklearn_dict["diagnostics_columns"] is None:
                sklearn_dict["diagnostics_columns"] = diagnostics_names
            if sklearn_dict["data_columns"] is None:
                sklearn_dict["data_columns"] = feature_names

            sklearn_dict["diagnostics"].append(diagnostics)
            sklearn_dict["data"].append(features)
        return sklearn_dict

    @property
    def feature_categories(self):
        return self.test_cases[0].feature_categories

    # TODO: Develop filter based on this


if __name__ == "__main__":

    def test_randomness():
        random_seed: int = random.randint(0, 1000)
        tccs: List[TestCaseCollection] = []
        selections: List[test_case.TestCase] = []
        print(f"Random Seed: {random_seed}")
        for index in range(10):
            tccs.append(
                TestCaseCollection(
                    os.path.join(home, "Documents/Dataset_V2/images_clean.csv"),
                    config_path,
                    random_seed=random_seed,
                )
            )

        for index, tcc in enumerate(tccs):
            selections.append(
                tcc.default_sample_function(
                    count=1, percent=False, random_seed=tcc.random_seed
                )
            )
            print(f"{index}: {[str(tc) for tc in selections[-1]]}")

    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    logging_util.setup_logging()
    home: str = os.path.expanduser("~")
    config_path: str = "./src/settings/allTest.yaml"

    tcc = TestCaseCollection(
        os.path.join(home, "Documents/Dataset_V2/images_clean.csv"), config_path
    )

    pp.pprint(tcc.test_cases)
    sample = tcc.equal_sample(70, percent=True)
    pp.pprint(len(sample))
    pp.pprint(len(tcc.all_but(sample)))
    pp.pprint(len(tcc.test_cases))
    pp.pprint(tcc.feature_categories)

    print(f"Randomness: ")
    test_randomness()
