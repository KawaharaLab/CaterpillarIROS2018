from multiprocessing import Pool
from importlib import import_module
import shutil
from collections.abc import Iterable
import sys
import os
import numpy as np

from controllers.pepg_actor_manager import PEPGActorManager
from controllers import base_actor, config, utils
from caterpillar import Caterpillar
import caterpillar_runner
from data_csv_saver import DataCSVSaver

np.seterr(all='raise')


class SimulationError(Exception):
    pass


def append_line(file_path: str, line: str):
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(line + '\n')
    else:
        with open(file_path, 'w') as f:
            f.write(line + '\n')


def exec_steps(steps: int, actor: base_actor.BaseActor, caterpillar: Caterpillar,
               disable_list: list, episode=None):
    for step in range(steps):
        caterpillar_runner.observe_and_act(actor, caterpillar, disable_list, episode=episode)
        caterpillar.step(step, int(1 / config.params["time_dalte"] / 10))


def run_simulation(sim_vals) -> float:
    """
    Run caterpillar with a policy given in argument.

    This function is for multiprocessing.
    sim_vals: (
        steps: int,
        actor_module_name: str,
        actor_params: [np.array(params_0), np.array(params_1), ...],
        disable_list: list
    )
    """
    steps, actor_module_name, actor_params, disable_list = sim_vals
    assert isinstance(steps, int)
    assert isinstance(actor_params, Iterable)
    assert isinstance(disable_list, Iterable)

    # Init actor
    actor_module = import_module(actor_module_name)
    actor = getattr(actor_module, config.COMMON_ACTOR_NAME)()
    actor.set_params(actor_params)

    # Init caterpillar
    caterpillar = Caterpillar(config.somites, mode="default")

    # Run steps
    exec_steps(steps, actor, caterpillar, disable_list=disable_list)

    reward = caterpillar.moved_distance()
    return reward


def mute():
    sys.stderr = open(os.devnull, 'w')


# Reinforcement Learning
def train_caterpillar(save_dir: utils.SaveDir, actor_module_name: str):
    # Dump train parameters
    config.print_config()

    actor_module = import_module(actor_module_name)
    actor_class = getattr(actor_module, config.COMMON_ACTOR_NAME)
    rlm = PEPGActorManager(actor_class)

    caterpillar = Caterpillar(config.somites, mode="default")
    config.dump_config(save_dir.log_dir())

    distance_log = DataCSVSaver(os.path.join(save_dir.log_dir(), "distance.txt"), ("episode", "reward"))
    sigma_log = DataCSVSaver(os.path.join(save_dir.log_dir(), "sigma.txt"), ("episode", "average sigma"))

    episode = 0
    while episode < config.params["episodes"]:
        params_sets = rlm.sample_params()

        print("\nEpisode: {}".format(episode))
        print("---------------------------------------------------------------------")
        rewards = []

        try:
            with Pool(processes=config.exec_params["worker_processes"], initializer=mute) as pool:
                rewards = pool.map(run_simulation, [(config.params["steps"], actor_module_name, p_set, []) for p_set in params_sets])

            rlm.update_params(np.array(rewards))
            episode += 1

            # Try parameters after this episode --------------------------------------
            rlm.set_params(rlm.parameters)
            caterpillar.reset()

            exec_steps(config.params["steps"], rlm.get_actor(), caterpillar, [], episode=episode - 1)

            # Save parameter performance
            distance_log.append_data(episode, caterpillar.moved_distance())
            sigma_log.append_data(episode, np.mean(rlm.sigmas))

            announce = "  --- Distance: {}".format(caterpillar.moved_distance())
            print(announce)

        except KeyboardInterrupt:
            command = input("\nSample? Finish? : ")
            if command in ["sample", "Sample"]:
                rlm.set_params(rlm.parameters)
                test_current_params(rlm.get_actor(), save_dir.log_dir(), episode)
                continue
            if command in ["finish", "Finish"]:
                print("Ending training ...")
                break

    rlm.set_params(rlm.parameters)
    rlm.save(save_file_path=os.path.join(save_dir.model_dir(), 'actor_model.pickle'))
    return rlm.get_actor()


def test_current_params(actor: base_actor, log_dir: str, episode: int):
    steps = input("How many steps for this sample?: ")
    if steps == "":
        utils.notice("default steps {}".format(config.params["default_sample_steps"]))
        steps = config.params["default_sample_steps"]

    # Record during sampling
    sim_distance_file = DataCSVSaver(
        "{}/train_result_ep{}_distance.txt".format(log_dir, episode),
        ("step", "distance")
    )
    sim_phase_diffs_file = DataCSVSaver(
        "{}/train_result_ep{}_phase_diffs.txt".format(log_dir, episode),
        ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)]
    )
    sim_actions_file = DataCSVSaver(
        "{}/train_result_ep{}_actions.txt".format(log_dir, episode),
        ["step"] + ["action_{}".format(i) for i in range(config.oscillators)]
    )
    sim_frictions_file = DataCSVSaver(
        "{}/train_result_ep{}_frictions.txt".format(log_dir, episode),
        ["step"] + ["friction_{}".format(i) for i in range(config.somites)]
    )

    caterpillar = Caterpillar(config.somites, mode="default")
    for step in range(int(steps)):
        try:
            _, action = caterpillar_runner.observe_and_act(actor, caterpillar, [])
            caterpillar.step(step, int(1 / config.params["time_dalte"] / 10))
        except Exception:
            continue
        else:
            # Save data
            sim_distance_file.append_data(step, caterpillar.moved_distance())
            sim_phase_diffs_file.append_data(step, *caterpillar.phases_from_base())
            sim_actions_file.append_data(step, *action)
            frictions = caterpillar.frictions
            sim_frictions_file.append_data(step, *frictions)

    print("Moved distance:", caterpillar.moved_distance())
    caterpillar.save_simulation("{}/train_result_ep{}.sim".format(log_dir, episode))
