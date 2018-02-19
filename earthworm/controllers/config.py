from datetime import datetime
import sys
import numpy as np

COMMON_ACTOR_NAME = "Actor"

# Settings
somites = 5
# somites = 12
# somites = 32
oscillators_list = (1,2,3)
# oscillators_list = tuple(range(1, somites-1))
oscillators = len(oscillators_list)

fixed_angles = {}

# Algorith related params (common among actor and critic)
params = {
    # About PEPG
    "batch_size": 8,
    "episodes": 600,
    "steps": 5000,
    "time_delta": 0.01,
    "default_sample_steps": 10000,
    "init_sigma": 2.,
    "parameter_upper_bound": 10,
    "parameter_lower_bound": -10.,
    "sigma_upper_bound": 10.,
    "mu_learning_rate": 0.1,
    "sigma_learning_rate": 0.07,
    "baseline_moving_average_gamma": 0.01,
}

earthworm_params = {
    "somite_mass": .3,
    "somite_radius": .35,
    "normal_angular_velocity": np.pi,
    "rts_max_natural_length": .7,
    "sp_natural_length": 0.7,
    "sp_k": 60.0,
    "dp_c": 10.0,
    "static_friction_coeff": 1.0,
    "dynamic_friction_coeff": 0.1,
    "viscosity_friction_coeff": 10.0,
}

# Execution params such number of processes
exec_params = {
    "worker_processes": 8,
    "save_params": True,
}


def dump_config(dir_path: str, additionals: {}):
    with open("{}/config_dump".format(dir_path), 'w') as f:
        print_config(f)

        print("additional info", file=f)
        for k, v in additionals.items():
            print("{}: {}".format(k, v), file=f)


def print_config(file=None):
    if file is None:
        file = sys.stdout
    print("==========configs==========", file=file)
    print("timestamp: {}".format(datetime.now()), file=file)
    print("{} smoites".format(somites), file=file)
    print("oscillators on somite {}".format(oscillators_list), file=file)
    for k, v in params.items():
        print("{}: {}".format(k, v), file=file)
    print("==========earthworm config=================", file=file)
    for k, v in earthworm_params.items():
        print("{}: {}".format(k, v), file=file)
