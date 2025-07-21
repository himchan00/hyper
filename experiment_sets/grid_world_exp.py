from experiment_sets.models import *


GPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for gpu
CPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for cpu only





GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "grid",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "grid_test",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "gridworld_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "Grid10-60-v0",        # env 
        "max_rollouts_per_task": 1, # zero-shot
        #
        "policy_layers": "64 64", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "512",
        "latent_dim": 25,  # latent dim for policy
        "policy_latent_embedding_dim": 50,
        "policy_task_embedding_dim": 50, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "eval_interval": 5,
        "log_interval": 5,
        "vis_interval": 50,
        "num_frames": int(4e6),
        "tbptt_stepsize": None,
        "lr_vae": 0.001,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_policy": [1e-3, 3e-4, 1e-4],
        "seed": [73],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        # VI_HN,
        RNN_HN,
        ]
    })

# some checks on experiments above
check_exps(CPU_EXPERIMENT_SETS+GPU_EXPERIMENT_SETS)


