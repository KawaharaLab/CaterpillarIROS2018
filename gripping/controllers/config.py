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
# grippers_list = tuple(range(somites))
# grippers_list = (0,2,3,4,5,8,9)
# grippers_list = (0,3,4,5,6,8,9,10,11,14,15,16,19,20,22,23,25,26,29,30)
grippers_list = tuple(range(somites))

oscillators = len(oscillators_list)
grippers = len(grippers_list)

fixed_angles = {3: np.pi*1/2}

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

caterpillar_params = {
    "somite_mass": .3,
    "somite_radius": .35,
    "normal_angular_velocity": np.pi,
    "sp_natural_length": 0.7,
    "sp_k": 80.0,
    "dp_c": 10.0,
    "horizon_ts_k0": 0.,
    "horizon_ts_k1": 0.,
    "vertical_ts_k0": 4.,
    "vertical_ts_k1": 0.,
    "vertical_ts_c": 1.,
    "vertical_realtime_tunable_torsion_spirng_k": 12.,
    "realtime_tunable_ts_rom": np.pi * 1. / 3.,
    "static_friction_coeff": 1.0,
    "dynamic_friction_coeff": 0.1,
    "viscosity_friction_coeff": 10.0,
    "tip_sub_static_friction_coeff": 0.1,
    "tip_sub_dynamic_friction_coeff": 0.01,
    "tip_sub_viscosity_friction_coeff": 1.0,
    "friction_switch_tan": np.tan(np.pi * 1. / 3.),
    "gripping_phase_threshold": np.sin(np.pi * 5. / 4.),
    "gripping_shear_stress_k": 500.,
    "gripping_shear_stress_c": 10.,
}

# Execution params such number of processes
exec_params = {
    "worker_processes": 16,
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
    print("grippers {}".format(grippers_list), file=file)
    for k, v in params.items():
        print("{}: {}".format(k, v), file=file)
    print("==========caterpillar config=================", file=file)
    for k, v in caterpillar_params.items():
        print("{}: {}".format(k, v), file=file)
