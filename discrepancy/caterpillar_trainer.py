from fast_rl_algo.pepg_actor_manager import PEPGActorManager
from fast_rl_algo.pepg import SymmetricPEPG
from fast_rl_algo.base_actor import BaseActor
from fast_rl_algo.config import config
from caterpillar import Caterpillar
from caterpillar_runner import observe_and_act
from data_csv_saver import DataCSVSaver

import numpy as np
from multiprocessing import Pool
import multiprocessing
from collections import abc
from importlib import import_module
import shutil
from optparse import OptionParser
import sys
import os


np.seterr(all='raise')


TIME_DALTE = config.params["time_dalte"]


class SimulationError(Exception):
    pass


def append_line(file_path: str, line: str):
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(line + '\n')
    else:
        with open(file_path, 'w') as f:
            f.write(line + '\n')


def exec_steps(steps: int, actor: BaseActor, caterpillar: Caterpillar, episode=None, disable_list=[]):
    """
        To save and visualize variables in Tensorboard, you should provide episode.
    """
    for step in range(steps):
        observe_and_act(actor, caterpillar, episode=episode, disable_list=disable_list)
        caterpillar.step(step, int(1 / TIME_DALTE / 10))


def run_simulation(sim_vals) -> float:
    """
        Run caterpillar with a policy given in argument.
        This function is for multiprocessing.

        sim_vals: (
            steps: int,
            actor_module_name: str,
            actor_name: str,
            actor_params: [np.array(params_0), np.array(params_1), ...],
            disable_list: list
        )
    """
    steps, actor_module_name, actor_name, actor_params, disable_list = sim_vals
    assert isinstance(steps, int)
    assert isinstance(actor_params, abc.Iterable)

    # Init actor
    actor_module = import_module(actor_module_name)
    actor = getattr(actor_module, actor_name)()
    actor.set_params(actor_params)

    # Init caterpillar
    caterpillar = Caterpillar(config.params["somites"], mode="default")

    # Run steps
    exec_steps(steps, actor, caterpillar, disable_list=disable_list)

    reward = caterpillar.moved_distance()
    return reward


def mute():
    sys.stderr = open(os.devnull, 'w')


# Delete directory and recreate
def reset_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# Reinforcement Learning
def train_caterpillar(save_dir: str, actor_module_name: str, actor_name: str, disable_list=[]):
    actor_module = import_module(actor_module_name)
    actor_class = getattr(actor_module, actor_name)
    rlm = PEPGActorManager(actor_class=actor_class)

    caterpillar = Caterpillar(config.params["somites"], mode="default")
    config.dump_config()

    # Create save place
    reset_dir(save_dir)
    # Metrics
    data_dir = os.path.join(save_dir, 'logs', 'data')
    os.makedirs(data_dir)
    distance_log = DataCSVSaver(os.path.join(data_dir, "distance.txt"), ("episode", "reward"))
    sigma_log = DataCSVSaver(os.path.join(data_dir, "sigma.txt"), ("episode", "average sigma"))
    # Models
    model_dir = os.path.join(save_dir, 'models')
    os.makedirs(model_dir)

    episode = 0
    while episode < config.params["episodes"]:
        params_sets = rlm.sample_params()

        print("\nEpisode: {}".format(episode))
        print("---------------------------------------------------------------------")
        rewards = []

        try:
            with Pool(processes=config.exec_params["worker_processes"], initializer=mute) as pool:
                rewards = pool.map(run_simulation, [(config.params["steps"], actor_module_name, actor_name, p_set, disable_list) for p_set in params_sets])

            rlm.update_params(np.array(rewards))
            episode += 1

            # Try parameters after this episode --------------------------------------
            rlm.set_params(rlm.parameters)
            caterpillar.reset()

            exec_steps(config.params["steps"], rlm.get_actor(), caterpillar, episode=episode - 1, disable_list=disable_list)

            # Save parameter performance
            distance_log.append_data(episode, caterpillar.moved_distance())
            sigma_log.append_data(episode, np.mean(rlm.sigmas))

            announce = "  --- Distance: {}".format(caterpillar.moved_distance())
            print(announce)

        except KeyboardInterrupt:
            command = input("\nSample? Finish? : ")
            if command in ["sample", "Sample"]:
                steps = input("How many steps for this sample?: ")
                if steps == "":
                    config.notice("default steps {}".format(config.params["default_sample_steps"]))
                    steps = config.params["default_sample_steps"]

                # Record during sampling
                sim_distance_file = DataCSVSaver(
                    "{}/train_result_ep{}_distance.txt".format(data_dir, episode),
                    ("step", "distance"))
                sim_phase_diffs_file = DataCSVSaver(
                    "{}/train_result_ep{}_phase_diffs.txt".format(data_dir, episode),
                    ["step"] + ["phi_{}".format(i) for i in range(config.params["oscillators"])])
                sim_actions_file = DataCSVSaver(
                    "{}/train_result_ep{}_actions.txt".format(data_dir, episode),
                    ["step"] + ["action_{}".format(i) for i in range(config.params["oscillators"])]
                )
                sim_frictions_file = DataCSVSaver(
                    "{}/train_result_ep{}_frictions.txt".format(data_dir, episode),
                    ["step"] + ["friction_{}".format(i) for i in range(config.params["somites"])]
                )

                rlm.set_params(rlm.parameters)
                caterpillar.reset()

                for step in range(int(steps)):
                    try:
                        _, action = observe_and_act(rlm.get_actor(), caterpillar, disable_list=disable_list)
                        caterpillar.step(step, int(1 / TIME_DALTE / 10))
                    except:
                        continue
                    else:
                        # Save data
                        sim_distance_file.append_data(step, caterpillar.moved_distance())
                        sim_phase_diffs_file.append_data(step, *caterpillar.phases_from_base())
                        sim_actions_file.append_data(step, *action)
                        frictions = caterpillar.frictions
                        sim_frictions_file.append_data(step, *frictions)

                print("Moved distance:", caterpillar.moved_distance())
                caterpillar.save_simulation("{}/train_result_ep{}.sim".format(data_dir, episode))
                continue

            if command in ["finish", "Finish"]:
                print("Ending training ...")
                break

    rlm.set_params(rlm.parameters)
    rlm.save(save_file_path=os.path.join(model_dir, 'actor_model.pickle'))
    return rlm.get_actor()
