import pickle
import argparse
import pprint


def setup_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "file", type=str, help="Pickled feature save file to load features from."
    )
    args = ap.parse_args()
    return args


def setup_pprint():
    pp = pprint.PrettyPrinter(indent=4)
    return pp


if __name__ == "__main__":
    args = setup_args()
    pp = setup_pprint()
    with open(args.file, "rb") as input_file:
        features = pickle.load(input_file)
    pp.pprint(features)
