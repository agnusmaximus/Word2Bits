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
intrinsic_outputlog_directory = "%s/output_logs" % intrinsic_results_directory

# Parameters
intrinsic_evaluation_parameters = {
    "maximum_parallel_jobs_per_machine" : 3,
}

# prepr functions should return tuple: (command_to_prep, new_path_of_vectors)
def top_100_vectors_prepr(in_path, out_dir):
    new_path = out_dir + "/" + in_path.split("/")[-1] + "head100"
    return ("head -n 100 %s > %s" % (in_path, new_path), new_path)
    
# Tasks
tasks = [
    {
        "path" : os.path.abspath("./automate_results/output_vectors/vectors_binary=0_bitlevel=2_iter=25_min-count=5_negative=12_reg=0_sample=0.0001_size=400_threads=35_train=dfsscratch0maxlamwiki.en.txt_window=10"),
        #"preprocess" : top_100_vectors_prepr,
        "preprocess" : None,
    },
]

# Get available machines for launching jobs. Assumes that all our jobs are basically python.
# Works by counting number of running python scripts under $username and checking if
# they're less than maximum_parallel_jobs_per_machine
def get_available_targets_intrinsic(targets, override_num_tasks=None):
    outputs = perform_commands(targets, 'pgrep -u "$(whoami)" python')
    available_targets = []
    for (output, exit_code), target in zip(outputs, targets):
        if override_num_tasks is None:
            if len(output.strip().split()) < intrinsic_evaluation_parameters["maximum_parallel_jobs_per_machine"]:
                available_targets.append(target)
        else:
            if len(output.strip().split()) < override_num_tasks:
                available_targets.append(target)
    return available_targets

def run_intrinsic_task_on_target(target, task, run_single_emb_script, preprocess_dir, output_dir):

    word_vectors_path = task["path"]
    preprocess_command, new_path = (None,None) if "preprocess" not in task else task["preprocess"](word_vectors_path, preprocess_dir)
    if new_path is not None:
        word_vectors_path = new_path

    emb_script_location = "/".join(run_single_emb_script.split("/")[:-1])

    # source python; cd to script location; run script on path
    cd_to_script_location_command = "cd %s" % emb_script_location
    source_python_command = "source /dfs/scratch0/maxlam/env2/bin/activate"
    run_script_command = "bash %s %s %s" % (run_single_emb_script, word_vectors_path, output_dir)
    full_command = "%s && %s && %s && %s" % (preprocess_command, cd_to_script_location_command, source_python_command, run_script_command)

    #print(full_command)
    perform_command_remote(target, run_async_krbtmux_command(full_command))

if __name__=="__main__":
    print("Usage: python automate_intrinsic_evaluation.py host1,host2,...hostn username [verbosity]")
    print("Automating intrinsic evaluation...")

    hosts = sys.argv[1].split(",")
    username = sys.argv[2]
    if len(sys.argv) >= 4:
        global verbosity
        verbosity = sys.argv[3]
    targets = ["%s@%s" % (username, host) for host in hosts]

    # Make directories
    mkdirp(preprocessed_vectors_directory)
    mkdirp(intrinsic_results_directory)
    mkdirp(intrinsic_outputlog_directory)

    # Get bash script which runs all intrinsic tasks
    run_single_emb_script, exit_status = perform_command_local('find `pwd`/intrinsic_evaluation -name "run_single_emb_test.sh"')
    assert_ok(run_single_emb_script, exit_status)
    
    for task in tasks:
        available_targets = []
        while len(available_targets) == 0:
            available_targets = get_available_targets_intrinsic(targets)
            print("Available:", available_targets)
            if len(available_targets) != 0:
                break;
            time.sleep(10) # replace with PAUSE_TIME

        target = available_targets[0]
        run_intrinsic_task_on_target(target, task, run_single_emb_script, preprocessed_vectors_directory, intrinsic_outputlog_directory)

    # Wait for all tasks to finish
    available_targets = get_available_targets_intrinsic(targets, override_num_tasks=1)
    while len(available_targets) != len(targets):
        available_targets = get_available_targets_intrinsic(targets, override_num_tasks=1)
        print("Waiting for finish...")
        print(available_targets)
        
        
