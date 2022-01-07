import os
from typing import List, Optional
import nibabel
import radiomics
import collections
import hashlib
import json
import pickle
import logging

import image_manip
import logging_util


class TestCase:
    # Path to the original, unchanged annotation file
    annotation_original_path: str
    # Path to the MRI scan file
    scan_path: str
    # Suffix for reconstructed annotation
    suffix_reconstructed: str
    # pyRadiomics extractor
    extractor: radiomics.featureextractor.RadiomicsFeatureExtractor
    feature_save_dir: str
    feature_vector: dict
    pcr: bool

    def __init__(
        self,
        scan_path: str,
        annotation_path: str,
        pcr: bool,
        extractor: radiomics.featureextractor.RadiomicsFeatureExtractor,
        suffix_reconstructed: str = ".inverse_dist.nii.gz",
        feature_save_dir: str = "./savefiles",
    ) -> None:
        logging_util.setup_logging()
        self.scan_path = scan_path
        self.annotation_original_path = annotation_path
        self.suffix_reconstructed = suffix_reconstructed
        self.feature_save_dir = feature_save_dir
        self.extractor = extractor
        self.pcr = pcr

        try:
            self.create_annotation()
        except Exception as ex:
            logging_util.log_wrapper(
                f"Error while loading reconstructed annotation: {ex}", logging.ERROR
            )
            raise ex
        try:
            self.create_feature_vector()
        except Exception as ex:
            logging_util.log_wrapper(
                f"Error while loading feature vector", logging.ERROR
            )
            raise ex

    @property
    def annotation_path(self) -> str:
        """
        Actual annotation to use
        """
        return self.annotation_reconstructed_path

    @property
    def annotation_reconstructed_path(self) -> str:
        """
        Path to the 3D-reconstructed annotation file
        """
        return self.annotation_original_path + self.suffix_reconstructed

    def create_feature_vector(self) -> None:
        """
        Create radiomics feature vector. Loads features from file,
        if exists. Generates features if not.
        """
        if os.path.exists(self.feature_save_path):
            # Load features from save file
            logging_util.log_wrapper(
                f"Loading feature vector from {self.feature_save_path}", logging.INFO
            )
            self._load_feature_vector()
        else:
            # Generate features
            logging_util.log_wrapper(f"Generating feature vector", logging.INFO)
            self._generate_feature_vector()
            logging_util.log_wrapper(
                f"Saving feature vector to {self.feature_save_path}", logging.INFO
            )
            self._save_feature_vector()

    def create_annotation(self) -> None:
        if os.path.exists(self.annotation_reconstructed_path):
            logging_util.log_wrapper(
                f"Reconstructed annotation {self.annotation_reconstructed_path} exists already.",
                logging.INFO,
            )
        else:
            logging_util.log_wrapper(
                f"Creating reconstructed annotation {self.annotation_reconstructed_path}",
                logging.INFO,
            )
            self._reconstruct_3D()

    @property
    def feature_save_path(self) -> str:
        """
        Save path for radiomics features.
        """
        save_file_name: str = f"{self._get_scan_hash()}_{self._get_annotation_hash()}_{self._get_extractor_settings_hash()}"
        return os.path.join(self.feature_save_dir, save_file_name)

    def _load_feature_vector(self):
        with open(self.feature_save_path, "rb") as input_file:
            self.feature_vector = pickle.load(input_file)

    def _save_feature_vector(self):
        if not os.path.exists(self.feature_save_dir):
            os.makedirs(self.feature_save_dir)
        with open(self.feature_save_path, "wb") as output_file:
            pickle.dump(self.feature_vector, output_file)

    def _generate_feature_vector(self):
        self.feature_vector = self.extractor.execute(
            self.scan_path, self.annotation_path
        )

    def _get_filehash(self, path: str) -> str:
        """
        Get sha256 hash for any file.
        """
        block_size: int = 1024
        hasher = hashlib.sha256()
        with open(path, "rb") as input_file:
            while True:
                data: bytes = input_file.read(block_size)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()

    def _get_scan_hash(self) -> str:
        """
        Get hash of scan file.
        """
        return self._get_filehash(self.scan_path)

    def _get_annotation_hash(self) -> str:
        """
        Get hash of annotation file.
        """
        return self._get_filehash(self.annotation_path)

    def _get_extractor_settings_hash(self) -> str:
        """
        Get hash of extractor settings to identify extractors
        with equivalent settings.
        """
        savedict: collections.OrderedDict = collections.OrderedDict()
        savedict["enabledFeatures"] = self.extractor.enabledFeatures
        savedict["enabledImagetypes"] = self.extractor.enabledImagetypes
        savedict["settings"] = self.extractor.settings
        json_representation: str = json.dumps(
            savedict,
            ensure_ascii=True,
            indent=0,
            sort_keys=True,
            separators=(", ", ": "),
        )
        savedict_hash = hashlib.sha256(json_representation.encode(encoding="utf-8"))
        return savedict_hash.hexdigest()

    def _check_nifti(self) -> None:
        """
        Checks if input NIFTI1 files exist and are valid.
        """
        missing_files: List[str] = []
        scan: nibabel.Nifti1Image
        annotation: nibabel.Nifti1Image

        for file in [self.scan_path, self.annotation_original_path]:
            if not os.path.exists(file):
                missing_files.append(file)
        if missing_files:
            raise FileNotFoundError(f"Input file(s) not found: {missing_files}")

        scan = nibabel.load(self.scan_path)
        annotation = nibabel.load(self.annotation_original_path)

        if scan.header.get_zooms() != annotation.header.get_zooms():
            raise ValueError(
                f"Dimensions of scan/annotation do not match: {scan.header.get_zooms()} != {annotation.header.get_zooms()}"
            )

    def _reconstruct_3D(self) -> None:
        """
        Create a 3D reconstructed annotation and save
        it to the annotation reconstruction path.
        """
        annotation_original: nibabel.Nifti1Image = nibabel.load(
            self.annotation_original_path
        )
        annotation_reconstructed: nibabel.Nifti1Image = image_manip.twoD23D(
            annotation_original
        )
        nibabel.save(annotation_reconstructed, self.annotation_reconstructed_path)

    def _str_dict(self):
        dict_representation: dict = {}
        dict_representation["scan_path"] = self.scan_path
        dict_representation["annotation_path"] = self.annotation_path
        dict_representation["feature_save_path"] = self.feature_save_path
        dict_representation["pcr"] = self.pcr
        return dict_representation

    def __str__(self):
        return json.dumps(self._str_dict())
        # return f"{self.scan_path}/{self.annotation_path}/{self.pcr} - {self.feature_save_path}"

    @property
    def feature_categories(self):
        feature_categories_dict: dict[str, Any] = {}
        for key in self.feature_vector:
            levels: List[str] = key.split("_")
            inner_level: Optional[dict] = None
            # Build tree of feature categories
            for level in levels:
                if inner_level is None:
                    if level not in feature_categories_dict:
                        feature_categories_dict[level] = {}
                    inner_level = feature_categories_dict[level]
                else:
                    if level not in inner_level:
                        inner_level[level] = {}
                    inner_level = inner_level[level]
        return feature_categories_dict


if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    logging_util.setup_logging()
    home: str = os.path.expanduser("~")

    extractor: radiomics.featureextractor.RadiomicsFeatureExtractor = (
        radiomics.featureextractor.RadiomicsFeatureExtractor()
    )
    tc = TestCase(
        os.path.join(home, "Documents/Dataset_V2/MR1.nii.gz"),
        os.path.join(home, "Documents/Dataset_V2/MR1A.nii.gz"),
        False,
        extractor,
    )

    pp.pprint(tc.feature_vector)
