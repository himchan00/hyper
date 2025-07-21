import os
import glob
import argparse
import itertools
import subprocess

from multiprocessing import Pool
from time import time



EXPERIMENT_SET = [] 


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def get_cmd(experiment_set):
    cmds = []
    log_dir = os.path.join(os.path.join(THIS_DIR,'data'), experiment_set["dir_name"])
    for experiment in experiment_set["experiments"]:
        # create Cartesian product over search parameters
        search_params = itertools.product(*experiment_set["search_arguments"].values())
        for search_param_setting in search_params:
            arg_dict = {}
            # add shared params other parameters
            arg_dict.update(experiment_set["shared_arguments"])
            # add params for this experiment
            arg_dict.update(experiment) # this will override any shared params
            # add this combination to the args dict. Will overwrite prior two.
            named_search_params = {list(experiment_set["search_arguments"].keys())[i] : param_value for i, param_value in enumerate(search_param_setting)}
            exp_label = "_".join([str(k) + "_" + str(v) for k, v in named_search_params.items()])
            arg_dict.update(named_search_params)
            arg_dict.update({"results_log_dir": log_dir})
            arg_dict.update({"exp_label": exp_label})
            cmd = []
            for key, value in sorted(list(arg_dict.items())):
                cmd.append("--"+key)
                cmd.append(str(value))
            cmds.append(cmd)
    mujoco_ver = str(experiment_set["mujoco_version"]) if "mujoco_version" in experiment_set else 150  # default mujoco version
    return cmds, mujoco_ver



def import_experiment(experiment_file):
    global EXPERIMENT_SET
    imported_file = __import__("experiment_sets."+experiment_file, fromlist=[''])
    EXPERIMENT_SET.extend(imported_file.GPU_EXPERIMENT_SETS)


def safe_exec_cmdstr_syncronous(cmd_str, name_of_caller, print_info=True):
    success = True
    try:
        if print_info:
            exit_code = subprocess.call(cmd_str, stderr=subprocess.STDOUT, shell=True)
        else:
            exit_code = subprocess.call(cmd_str, stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL, shell=True)
        assert exit_code == 0, "exit_code was not 0: {}".format(exit_code)
    except Exception as e:
        if print_info:
            print("Error in", name_of_caller+":", e)
            print("for cmd:", cmd_str)
            if hasattr(e, "output"):
                print(e.output)
        success = False
    return success

def build_dockerfile(docker_file, docker_dir):
    dockerfile_name = os.path.basename(docker_file)
    print("Updating Docker:", dockerfile_name, flush=True)
    success = safe_exec_cmdstr_syncronous("cd "+docker_dir+" && bash build.sh " + dockerfile_name, "build_dockers()", print_info=False)
    if not success:
        print("Warning: Docker file could not be built:", docker_file, flush=True)

def build_dockers(args):
    if args.docker_dir is None or args.docker_dir == ["None"]:
        return
    docker_files = glob.glob(os.path.join(os.path.expanduser(args.docker_dir), "Dockerfile_*"))
    print("\nUpdating Dockers...", flush=True)
    with Pool(processes=len(docker_files)) as p_pool:
        p_pool.starmap(build_dockerfile, [(f, args.docker_dir) for f in docker_files])
    print("", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('EXPERIMENT_FILE', help='the filename of experiment to run (in ./experiment_sets/) with no commas and no extension')
    parser.add_argument('--docker_dir', help='specify a directories with docker imgs to build. Names must start with Dockerfile_', default=THIS_DIR)
    parser.add_argument('--docker_cmd', help='a cmd for running with gpu argument, NOT detached', type=str, default=os.path.join(THIS_DIR,'run_gpu.sh')+' mujoco')
    parser.add_argument('--python_cmd', type=str, default='python -u ' + os.path.join(THIS_DIR,'main.py'))
    parser.add_argument('--device', type=int, default=0, help='id of the gpu to run on')

    args = parser.parse_args()

    # Import Experiments
    import_experiment(args.EXPERIMENT_FILE)

    #### Setup ####
    build_dockers(args)     # Build Dockers (make sure up to date)
    # get commands to run on each device
    cmds, mujoco_ver = get_cmd(EXPERIMENT_SET[0])

    print("Running experiments:", len(cmds), flush=True)
    for cmd in cmds:
        full_next_cmd = [args.docker_cmd+str(mujoco_ver), str(args.device), args.python_cmd] + cmd
        full_cmd_str = " ".join(full_next_cmd)
        print("Running cmd:", full_cmd_str, flush=True)
        # Run the command
        subprocess.run(full_cmd_str, shell=True)

    print("Done!")

if __name__ == "__main__":
    main()

