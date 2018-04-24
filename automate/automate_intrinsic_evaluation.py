# python automate/automate_intrinsic_evaluation.py dawn16.stanford.edu,dawn9.stanford.edu maxlam 1
# -----------------------------------------------------------------------------------
# This program launches a set of intrinsic evaluation tasks across specified dawn machines.
#
# This program makes several assumptions
# - Hyperparameters are hardcoded in the file
# - Word2bits binary is called 'w2b'
# - Can ssh without password into the target machines
# - Called from Word2Bits base directory
# - Called on a dawn machine cluster...

import sys
import time
import itertools
import os
import commands
from automate_training import *

# Be wary of conflicts
preprocessed_vectors_directory = os.path.abspath("./preprocessed_vectors")
intrinsic_results_directory = os.path.abspath("./automate_intrinsic_results")
intrinsic_outputlog_directory = "%s/output_logs" % results_directory

# Parameters
intrinsic_evaluation_parameters = {
    "maximum_parallel_jobs_per_machine" : 3,
}

# Tasks
tasks = [
    {
        "path" : os.path.abspath("./automate_results/output_vectors/vectors_binary=0_bitlevel=2_iter=25_min-count=5_negative=12_reg=0_sample=0.0001_size=400_threads=35_train=dfsscratch0maxlamwiki.en.txt_window=10"),
        "preprocess" : None,
    },
]

# Get available machines for launching jobs. Assumes that all our jobs are basically python.
# Works by counting number of running python scripts under $username and checking if
# they're less than maximum_parallel_jobs_per_machine
def get_available_targets_intrinsic(targets):
    outputs = perform_commands(targets, 'pgrep -u "$(whoami)" python')
    available_targets = []
    for (output, exit_code), target in zip(outputs, targets):
        if len(output.strip().split()) < intrinsic_evaluation_parameters["maximum_parallel_jobs_per_machine"]:
            available_targets.append(target)
    return available_targets

if __name__=="__main__":
    print("Usage: python automate_intrinsic_evaluation.py host1,host2,...hostn username [verbosity]")
    print("Automating intrinsic evaluation...")

    hosts = sys.argv[1].split(",")
    username = sys.argv[2]
    if len(sys.argv) >= 4:
        global verbosity
        verbosity = sys.argv[3]
    targets = ["%s@%s" % (username, host) for host in hosts]

    mkdirp(preprocessed_vectors_directory)
    mkdirp(intrinsic_results_directory)
    mkdirp(intrinsic_outputlog_directory)

    
    print(get_available_targets_intrinsic(targets))
