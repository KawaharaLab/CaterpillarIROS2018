# Settings
somites = 5
oscillators = somites - 1

# Algorith related params (common among actor and critic)
params = {
    # About PEPG
    "batch_size": 8,
    "episodes": 40,
    "steps": 2000,
    "time_dalte": 0.01,
    "default_sample_steps": 10000,
    "init_sigma": 1.,
    "converged_sigma": 0.01,
    "sigma_upper_bound": 2.,
    "learning_rate": 0.1,
}

# Execution params such number of processes
exec_params = {
    "worker_processes": 8,
    "save_params": True,
}


def dump_config(dir_path: str):
    with open("{}/config_dump.txt".format(dir_path), 'w') as f:
        f.write("\n---configs--------------\n")
        f.write(str(params))
