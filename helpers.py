# coding: utf-8

"""
Helpers and constants.
"""

import os
import shutil
import subprocess

import six
import numpy as np


# the project directory
this_dir = os.path.dirname(os.path.abspath(__file__))

# the local data directory for caching
data_dir = os.path.join(this_dir, "data")

# data base directory on CERN EOS
eos_data_dir = "/eos/user/m/mrieger/swan_aws_data"

# base url of files on CERN EOS
eos_data_url = "https://cernbox.cern.ch/index.php/s/GAkUpZjhEbzi2Uy/download?path={}&files={}"

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


def _wget(url, path):
    # create the parent directory, remove the file if existing
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    elif os.path.exists(path):
        os.remove(path)

    # build the wget command and run it
    cmd = ["wget", "-O", path, url]
    try:
        subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise Exception("download of url '{}' failed: {}".format(url, e))

    return path


def provide_file(kind, index):
    """
    Returns the absolute file path of a file of an input dataset given by *kind* ("train", "valid"
    or "test") and its *index*. When the file is not locally accessible, it is downloaded first from
    the public CERNBox directory.
    """
    # get the name of the file
    file_name = _check_dataset_file(kind, index)

    # make sure the user data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # when it's already in the user data directy, return the full path
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        return file_path

    # when the public eos data dir is accessible, copy the file into the user data dir
    has_eos = os.access(eos_data_dir, os.R_OK)
    if has_eos:
        public_path = os.path.join(eos_data_dir, file_name)
        print(f"copy file from {public_path}")
        shutil.copy2(public_path, file_path)
        return file_path

    # otherwise, download it using the public CERNBox url
    quote = six.moves.urllib.parse.quote
    url = eos_data_url.format(quote("/", safe=""), quote(file_name, safe=""))
    print(f"wget file from {url}")
    _wget(url, file_path)

    return file_path


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

    # get all local file paths via download or public access
    file_paths = [provide_file(kind, i) for i in range(start_file, stop_file)]

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
