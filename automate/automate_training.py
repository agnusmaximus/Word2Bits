# python automate/automate_training.py dawn16.stanford.edu,dawn9.stanford.edu maxlam
# -----------------------------------------------------------------------------------
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

global verbosity
verbosity = 0

# Seconds in between checking for available machine
PAUSE_TIME = 300

# Directory structure
results_directory = os.path.abspath("./automate_results")
outputlog_directory = "%s/output_logs" % results_directory
vectors_directory = "%s/output_vectors" % results_directory

# Parameters
hyperparams = {
    #"epochs":[1,10,25],
    #"epochs":[25],
    "epochs":[10],
    #"bitlevel_or_regs":[0,1,2,"reg_.001"],
    "bitlevel_or_regs":["reg_.0005"],
    #"dimensions":[200,400,800,1000],
    "dimensions":[400],
    #"window_neg":[(2,5),(10,12),(5,2)],
    "window_neg":[(10,12)],
    "corpus":["/dfs/scratch0/maxlam/wiki.en.txt"],
    #"corpus":["/dfs/scratch0/maxlam/text8"],
    "min_count":[5],
    "subsample":[1e-4],
    "output_as_binary":[0],
    "threads":[35]
}
hyperparam_keys = ["epochs",
                   "bitlevel_or_regs",
                   "dimensions",
                   "window_neg",
                   "corpus",
                   "min_count",
                   "subsample",
                   "output_as_binary",
                   "threads"]

        
def extract_bitlevel_reg_from_param(bitlevel_reg):
    bitlevel, reg = None, None
    if type(bitlevel_reg) == type(""):
        if "reg_" in bitlevel_reg:
            reg = bitlevel_reg.split("reg_")[-1]
            bitlevel = 0
        else:
            assert(0)
    else:
        reg = 0
        bitlevel = bitlevel_reg
    assert bitlevel is not None
    assert reg is not None
    return bitlevel, reg

def argument_dict_from_hyperparameters(epochs,
                                       bitlevel_reg,
                                       dimension,
                                       window_neg,
                                       corpus,
                                       min_count,
                                       subsample,
                                       output_as_binary,
                                       threads,
                                       output_path=None):
    # WARNING: order of arguments must match hyperparam_keys
    # WARNING: Keys of dict must correspond to exact arguments for w2b binary
    r = {}
    r["iter"] = str(epochs)
    bitlevel, reg = extract_bitlevel_reg_from_param(bitlevel_reg)
    r["bitlevel"] = str(bitlevel)
    r["reg"] = str(reg)
    r["size"] = str(dimension)
    r["window"] = str(window_neg[0])
    r["negative"] = str(window_neg[1])
    r["train"] = str(corpus)
    r["min-count"] = str(min_count)
    r["sample"] = str(subsample)
    r["binary"] = str(output_as_binary)
    r["threads"] = str(threads)
    if output_path is not None:
        r["output"] = str(output_path)
    return r

def argument_dict_to_string(d, k_prefix="", kv_delim="=", kv_pair_delim="_", replace_slash=""):
    keys = sorted(d.keys())
    k_to_v = ["%s%s%s%s" % (k_prefix, k, kv_delim, d[k].replace("/", replace_slash)) for k in keys]
    return kv_pair_delim.join(k_to_v)

def argument_dict_to_parameter_string(d):
    return argument_dict_to_string(d, k_prefix="-", kv_delim=" ", kv_pair_delim=" ", replace_slash="/")

def important_log(message):
    if verbosity > 0:
        print("*"*len(message))
        print(message)
        print("*"*len(message))

def assert_ok(output, exit_status):
    if exit_status != 0:
        important_log("Error with previous command (output, exit_status): (%s, %s)" % (output, str(exit_status)))
        sys.exit(1)

def mkdirp(path):
    if not os.path.exists(path):
        important_log("Making path: %s" % path)
        os.makedirs(path)

def perform_command_remote(target, command, suppress_log=False):
    if not suppress_log:
        important_log("Performing command on target %s: '%s'" % (target, command))
    status, text = commands.getstatusoutput("ssh -t -q %s '%s'" % (target, command))
    exit_code = status >> 8
    return text, exit_code

def perform_commands(targets, command):
    results = []
    errors = []
    important_log("Performing command on targets %s: '%s'" % (",".join(targets), command))
    for target in targets:
        text, exit_code = perform_command_remote(target, command, suppress_log=True)
        results.append((text, exit_code))
    return results

def perform_command_local(command):
    important_log("Performing command locally: '%s'" % (command))
    status, text = commands.getstatusoutput("%s" % command)
    exit_code = status >> 8
    return text, exit_code

def wait_for_available_target(targets):
    while True:
        outputs = perform_commands(targets, 'pgrep -u "$(whoami)" w2b')
        available_targets = []
        unavailable_targets = []
        for (output, exit_code), target in zip(outputs, targets):
            if output.strip() == "":
                available_targets.append(target)
            else:
                unavailable_targets.append(target)
        important_log("Available: %s, Unavailable: %s" % (
            str(available_targets),
            str(unavailable_targets)))
        if len(available_targets) > 0:
            available_target = available_targets[0]

            # Shut tmux down on available target
            #perform_command_remote(available_target, 'tmux kill-server')
            
            return available_target, len(available_targets)
        time.sleep(PAUSE_TIME)

    assert 0
    
    # Should not reach here
    return available_target

def run_async_krbtmux_command(command):
    assert '"' not in command
    cur_path, exit_status = perform_command_local('pwd')    
    return 'bash %s/automate/maxstmux new "%s" ";" detach' % (cur_path, command)


def train_vector_on_target(target, arg_dict, raw_args_list, w2b_path):
    identifier = argument_dict_to_string(arg_dict)
    train_log_path = "%s/log_%s" % (outputlog_directory,
                                    identifier)
    vector_output_path = "%s/vectors_%s" % (vectors_directory,
                                            identifier)
    arg_dict_with_output_path = (
        argument_dict_from_hyperparameters(*(raw_args_list + (vector_output_path,))))    
    full_argument_string = argument_dict_to_parameter_string(arg_dict_with_output_path)
    full_command_string = "%s %s > %s" % (w2b_path, full_argument_string, train_log_path)
    perform_command_remote(target, run_async_krbtmux_command(full_command_string))

if __name__=="__main__":

    # python automate/automate_training.py dawn16.stanford.edu,dawn9.stanford.edu maxlam
    print("Usage: python automate_training.py host1,host2,..hostn username [verbosity]")
    print("Automating training...")
    
    hosts = sys.argv[1].split(",")
    username = sys.argv[2]
    if len(sys.argv) >= 4:
        global verbosity
        verbosity = sys.argv[3]
    targets = ["%s@%s" % (username, host) for host in hosts]

    # Make directories
    mkdirp(results_directory)
    mkdirp(outputlog_directory)
    mkdirp(vectors_directory)

    # Compile w2b
    assert_ok(*perform_command_local("make clean"))    
    assert_ok(*perform_command_local("make compile"))
    w2b_path, exit_status = perform_command_local('find `pwd` -name "w2b"')
    assert_ok(w2b_path, exit_status)

    # Generate all hyperparameters
    hyperparameter_lists = [hyperparams[k] for k in hyperparam_keys]
    for cfg_permutation in itertools.product(*hyperparameter_lists):
        cfg_argument_dict = argument_dict_from_hyperparameters(*cfg_permutation)
        available_target, n_available = wait_for_available_target(targets)
        train_vector_on_target(available_target, cfg_argument_dict, cfg_permutation, w2b_path)

    # Wait for all targets to be available
    n_available = 0
    while n_available != len(targets):
        _, n_available = wait_for_available_target(targets)
    
    # Example commands (commented out)
    #print(perform_commands(targets, run_async_krbtmux_command("sleep 60; echo `hostname` > /dfs/scratch0/maxlam/testing`hostname`")))
    #print(perform_commands(targets, 'pgrep -u "$(whoami)" w2b'))
    #print(perform_command_local("pwd"))
    
    
