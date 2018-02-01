import sys
import numpy as np

COMMON_ACTOR_NAME = "Actor"

# Settings
somites = 5
oscillators = somites - 1

# Algorith related params (common among actor and critic)
params = {
    # About PEPG
    "batch_size": 8,
    "episodes": 200,
    "steps": 2000,
    "time_dalte": 0.01,
    "default_sample_steps": 10000,
    "init_sigma": 1.,
    "converged_sigma": 0.01,
    "sigma_upper_bound": 2.,
    "learning_rate": 0.1,
    "tension_divisor": 8000,
}

caterpillar_params = {
    "somite_mass": .3,
    "somite_radius": .35,
    "normal_angular_velocity": np.pi,
    "rts_max_natural_length": 0.5,
    "rts_k": 100.,
    "rts_c": 1.,
    "rts_amp": .3,
    "friction_coeff_rate": 10.0,
}

# Execution params such number of processes
exec_params = {
    "worker_processes": 8,
    "save_params": True,
}


def dump_config(dir_path: str):
    with open("{}/config_dump".format(dir_path), 'w') as f:
        print_config(f)


def print_config(file=None):
    if file is None:
        file = sys.stdout
    print("==========configs==========", file=file)
    for k, v in params.items():
        print("{}: {}".format(k, v), file=file)
    print("==========caterpillar config=================", file=file)
    for k, v in caterpillar_params.items():
        print("{}: {}".format(k, v), file=file)
