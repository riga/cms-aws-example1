# coding: utf-8

"""
Helpers and constants.
"""

import os
import shutil

import numpy as np


# data base directory on EOS
eos_data_dir = "/eos/user/m/mrieger/swan_aws_data"

# data constants
n_constituents = 200
n_events_per_file = 50000
n_files = {
    "train": 20,
    "valid": 8,
    "test": 8,
}


def _check_dataset_file(kind, index):
    # validate the kind
    if kind not in n_files:
        raise ValueError("unknown dataset kind '{}', must be one of {}".format(
            kind, ",".join(n_files.keys())))

    # validate the index
    if not (0 <= index < n_files[kind]):
        raise ValueError("dataset '{}' has no file index {}".format(kind, index))

    return "{}_{}.npz".format(kind, index)


def load_data(kind, start_file=0, stop_file=-1):
    """
    Loads a certain *kind* of dataset ("train", "valid", or "test") and returns the four-vectors of
    the jet constituents and the truth label in a 2-tuple. Internally, each dataset consists of
    multiple files whose arrays are automatically concatenated. For faster prototyping and testing,
    *start_file* (included) and *stop_file* (first file that is *not* included) let you define the
    range of files to load.
    """
    # fix the file range if necessary
    if stop_file < 0:
        stop_file = n_files[kind] - 1

    # get all local file paths
    file_paths = []
    for i in range(start_file, stop_file):
        file_name = _check_dataset_file(kind, i)
        file_path = os.path.join(eos_data_dir, file_name)
        file_paths.append(file_path)

    # instead of loading all files, storing their contents and concatenating in the end, which can
    # have a peak memory consumption of twice the inputs, define output arrays with the correct
    # dimensions right away and fill it while iterating through files
    n_events = len(file_paths) * n_events_per_file
    vecs = np.zeros((n_events, n_constituents, 4), dtype=np.float32)
    labels = np.zeros((n_events,), dtype=np.float32)

    # open files and fill arrays
    for i, file_path in enumerate(file_paths):
        start = i * n_events_per_file
        stop = start + n_events_per_file

        data = np.load(file_path)
        vecs[start:stop, ...] = data["constituents"]
        labels[start:stop, ...] = data["truth_label"]
        data.close()

    return vecs, labels
