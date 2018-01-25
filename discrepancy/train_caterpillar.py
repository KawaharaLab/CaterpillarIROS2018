from rl_algo.reinforce_learning_manager import ReinforceLearningManager
from rl_algo.pepg_actor_manager import PEPGActorManager
from rl_algo.pepg import SymmetricPEPG
from rl_algo.base_actor import BaseActor
from rl_algo.nn_actor import NeuralNetworkActor
from rl_algo.config import config
from caterpillar import Caterpillar
# from earthworm import Earthworm as Caterpillar
from data_csv_saver import DataCSVSaver

import tensorflow as tf
import numpy as np
from multiprocessing import Pool
import multiprocessing
from collections import abc
from pickle import dump
from optparse import OptionParser
import sys
import os


np.seterr(all='raise')


TIME_DALTE = .01


class SimulationError(Exception):
    pass


def append_line(file_path: str, line: str):
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write(line + '\n')
    else:
        with open(file_path, 'w') as f:
            f.write(line + '\n')


def run_simulation(sim_vals) -> float:
    """
        Run caterpillar with a policy given in argument.
        This function is for multiprocessing.

        sim_vals: (
            steps: int,
            actor_params: [np.array(params_0), np.array(params_1), ...],
        )
    """
    steps, actor_params = sim_vals
    assert isinstance(steps, int)
    assert isinstance(actor_params, abc.Iterable)

    # Init actor
    tf.reset_default_graph()
    actor = NeuralNetworkActor()
    actor.set_params(actor_params)

    # Init caterpillar
    caterpillar = Caterpillar(config.params["somites"], mode="default")

    # Run steps
    exec_steps(steps, actor, caterpillar)

    reward = caterpillar.moved_distance()
    return reward


def exec_steps(steps: int, actor: BaseActor, caterpillar: Caterpillar, episode=None):
    """
        To save and visualize variables in Tensorboard, you should provide episode.
    """
    for step in range(steps):
        observe_and_act(actor, caterpillar, episode)
        caterpillar.step(step, int(1 / TIME_DALTE / 10))


def observe_and_act(actor: BaseActor, caterpillar: Caterpillar, episode=None):
    """
        To save and visualize variables in Tensorboard, you should provide episode.
    """
    frictions = np.matmul(caterpillar.frictions, np.array([1, 0, 0]))
    phis = caterpillar.phis
    state = np.concatenate([frictions, phis])
    action = actor.get_action(state, step=episode)
    caterpillar.feedback_phis(action)

    observation = (frictions, phis)
    return observation, action


def mute():
    sys.stderr = open(os.devnull, 'w')


# Reinforcement Learning
def train_caterpiller(save_name=None):
    multiprocessing.set_start_method('spawn')   # Tensorflow is not fork safe.
    caterpillar = Caterpillar(config.params["somites"], mode="default")
    rlm = PEPGActorManager(save_name=save_name)
    config.dump_config()

    distance_log = DataCSVSaver("{}/distance.txt".format(config.metrics_dir()), ("episode", "reward"))
    sigma_log = DataCSVSaver("{}/sigma.txt".format(config.metrics_dir()), ("episode", "average sigma"))

    episode = 0
    while episode < config.params["episodes"]:
        params_sets = rlm.sample_params()

        print("\nEpisode: {}".format(episode))
        print("---------------------------------------------------------------------")
        rewards = []

        try:
            with Pool(processes=config.exec_params["worker_processes"], initializer=mute) as pool:
                rewards = pool.map(run_simulation, [(config.params["steps"], p_set) for p_set in params_sets])

            rlm.update_params(np.array(rewards))
            episode += 1

            # Try parameters after this episode --------------------------------------
            rlm.set_params(rlm.parameters)
            caterpillar.reset()

            exec_steps(config.params["steps"], rlm.get_actor(), caterpillar, episode - 1)

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
                    "{}/train_result_ep{}_distance.txt".format(config.metrics_dir(), episode),
                    ("step", "distance"))
                sim_phase_diffs_file = DataCSVSaver(
                    "{}/train_result_ep{}_phase_diffs.txt".format(config.metrics_dir(), episode),
                    ["step"] + ["phi_{}".format(i) for i in range(config.params["oscillators"])])
                sim_actions_file = DataCSVSaver(
                    "{}/train_result_ep{}_actions.txt".format(config.metrics_dir(), episode),
                    ["step"] + ["action_{}".format(i) for i in range(config.params["oscillators"])]
                )
                sim_frictions_file = DataCSVSaver(
                    "{}/train_result_ep{}_frictions.txt".format(config.metrics_dir(), episode),
                    ["step"] + ["friction_{}".format(i) for i in range(config.params["somites"])]
                )

                rlm.set_params(rlm.parameters)
                caterpillar.reset()

                for step in range(int(steps)):
                    try:
                        _, action = observe_and_act(rlm.get_actor(), caterpillar)
                        caterpillar.step(step, int(1 / TIME_DALTE / 10))
                    except:
                        continue
                    else:
                        # Save data
                        sim_distance_file.append_data(step, caterpillar.moved_distance())
                        sim_phase_diffs_file.append_data(step, *caterpillar.phases_from_base())
                        sim_actions_file.append_data(step, *action)
                        frictions = np.matmul(caterpillar.frictions, np.array([1, 0, 0]))
                        sim_frictions_file.append_data(step, *frictions)

                print("Moved distance:", caterpillar.moved_distance())
                caterpillar.save_simulation("{}/train_result_ep{}.sim".format(config.metrics_dir(), episode))
                continue

            if command in ["finish", "Finish"]:
                print("Ending training ...")
                if config.exec_params["save_params"]:
                    print("Saving trained actor ...")
                    rlm.save()
                break

    rlm.set_params(rlm.parameters)
    rlm.save()
    return rlm.get_actor(), config.metrics_dir()


def main(args, opts):
    actor, log_dir = train_caterpiller(opts.trial_name)

    # Render trained caterpillar
    caterpillar = Caterpillar(config.params["somites"])

    exec_steps(config.params["default_sample_steps"], actor, caterpillar)

    print("Moved distance:", caterpillar.moved_distance())
    caterpillar.save_simulation("{}/train_result.sim".format(log_dir))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--name", default=None, dest="trial_name")
    opts, args = parser.parse_args()

    main(args, opts)
    sys.exit(0)
