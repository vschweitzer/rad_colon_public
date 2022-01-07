import logging
import argparse
import datetime
import radiomics
import os


def setup_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # ap.add_argument(
    #     "--loading-mode",
    #     "-l",
    #     choices=["list", "collection"],
    #     help="Load files from list of single files or from file collection(s).",
    #     default="collection",
    # )
    ap.add_argument(
        "file",
        type=str,
        help="Collection of files to read in.",
        default="/media/watson/Dataset_V2/images_clean.csv",
    )

    ap.add_argument(
        "--radiomics-params",
        "-p",
        type=str,
        default=os.path.join(".", "src", "settings", "allTest.yaml"),
        help="Parameter file for pyRadiomics extractor.",
    )

    ap.add_argument(
        "--force-feature-regenerate",
        "-f",
        action="store_true",
        default=False,
        help="Generate radiomics features, regardless of saved features.",
    )

    ap.add_argument(
        "--seed",
        action="store",
        type=int,
        default=0,
        help="Random seed to use for sample selection.",
    )

    ap.add_argument(
        "--filter-level",
        action="store",
        type=float,
        default=0.0,
        help="Importance filter level.",
    )

    return ap.parse_args()
