from multiprocessing import Pool
from importlib import import_module
import shutil
from collections.abc import Iterable
import itertools
import sys
import os
import numpy as np

from controllers.pepg import PEPG
from controllers import base_actor, config, utils
from caterpillar_lib.caterpillar import Caterpillar
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
               disable_list: list, episode=None) -> float:
    """
    Run caterpillar for designated steps.

    return run information which is (accumulated tensions,)
    """
    accumulated_tension = 0
    for step in range(steps):
        (_, _, tensions), action = caterpillar_runner.observe_and_act(actor, caterpillar, disable_list, episode=episode)
        accumulated_tension += np.sum(np.power(tensions, 2))

        feedbacks, angle_ranges = action[0], action[1]
        caterpillar.set_oscillation_ranges(tuple(angle_ranges))
        caterpillar.step_with_feedbacks(config.params["time_delta"], tuple(feedbacks))
    return (accumulated_tension,)


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
    # assert isinstance(actor_params, Iterable)
    assert isinstance(disable_list, Iterable)

    # Init actor
    actor_module = import_module(actor_module_name)
    actor = getattr(actor_module, config.COMMON_ACTOR_NAME)()
    actor.set_params(actor_params)

    # Init caterpillar
    caterpillar = Caterpillar(config.somites, config.oscillators_list, config.caterpillar_params)
    locomotion_distance = utils.locomotion_distance_logger(caterpillar)

    # Run steps
    (accumulated_tension,) = exec_steps(steps, actor, caterpillar, disable_list=disable_list)

    # reward = locomotion_distance(caterpillar.center_of_mass()[0]) - accumulated_tension / config.params["tension_divisor"]
    reward = locomotion_distance(caterpillar)
    return reward


def new_caterpillar() -> Caterpillar:
    return Caterpillar(config.somites, config.oscillators_list, config.caterpillar_params)


def mute():
    sys.stderr = open(os.devnull, 'w')


# Reinforcement Learning
def train_caterpillar(save_dir: utils.SaveDir, actor_module_name: str):
    actor_module = import_module(actor_module_name)
    actor_class = getattr(actor_module, config.COMMON_ACTOR_NAME)
    actor = actor_class()

    # Dump train parameters
    config.print_config()
    config.dump_config(save_dir.log_dir(), actor.dump_config())

    pepg = PEPG(actor.params_num())
    pepg.set_parameters(
        np.fromiter(itertools.chain.from_iterable([p.flatten().tolist() for p in actor.current_params()]), np.float64)
    )

    distance_log = DataCSVSaver(os.path.join(save_dir.log_dir(), "distance.txt"), ("episode", "distance"))
    reward_log = DataCSVSaver(os.path.join(save_dir.log_dir(), "reward.txt"), ("episode", "reward"))
    sigma_log = DataCSVSaver(os.path.join(save_dir.log_dir(), "sigma.txt"), ("episode", "average sigma"))

    episode = 0
    while episode < config.params["episodes"]:
        current_mus = pepg.get_parameters()
        epsilons = pepg.sample_epsilons(config.params["batch_size"])
        params_sets = np.concatenate([current_mus[:, np.newaxis] + epsilons, current_mus[:, np.newaxis] - epsilons], axis=1).T

        print("\nEpisode: {}".format(episode))
        print("---------------------------------------------------------------------")
        rewards = []

        try:
            with Pool(processes=config.exec_params["worker_processes"], initializer=mute) as pool:
                rewards = pool.map(run_simulation, [(config.params["steps"], actor_module_name, p_set, []) for p_set in params_sets])

            sample_num = epsilons.shape[1]
            r_plus = np.array(rewards[:sample_num])
            r_minus = np.array(rewards[sample_num:])
            nan_samples = np.where(np.isnan(r_plus) + np.isnan(r_minus))[0]
            # delete nan samples
            epsilons = np.delete(epsilons, nan_samples, axis=1)
            r_plus = np.delete(r_plus, nan_samples, axis=0)
            r_minus = np.delete(r_minus, nan_samples, axis=0)
            if epsilons.shape[1] > 0:
                pepg.update_parameters(epsilons, r_plus, r_minus)
                episode += 1

                # Try parameters after this episode --------------------------------------
                actor.set_params(pepg.get_parameters())
                caterpillar = Caterpillar(config.somites, config.oscillators_list, config.caterpillar_params)
                locomotion_distance = utils.locomotion_distance_logger(caterpillar)

                (accumulated_tension,) = exec_steps(config.params["steps"], actor, caterpillar, [], episode=episode - 1)
                d = locomotion_distance(caterpillar)
                # reward = d - accumulated_tension / config.params["tension_divisor"]
                reward = d

                # Save parameter performance
                distance_log.append_data(episode, d)
                sigma_log.append_data(episode, np.mean(pepg.get_sigmas()))
                reward_log.append_data(episode, reward)

                announce = "  --- Distance: {}   Reward: {}".format(d, reward)
                print(announce)
            else:
                print("got nan position. update failed")

        except KeyboardInterrupt:
            command = input("\nSample? Finish? : ")
            if command in ["sample", "Sample"]:
                actor.set_params(pepg.get_parameters())
                test_current_params(actor, save_dir.log_dir(), episode)
                continue
            if command in ["finish", "Finish"]:
                print("Ending training ...")
                break

    actor.set_params(pepg.get_parameters())
    actor.save(os.path.join(save_dir.model_dir(), 'actor_model.pickle'))
    return actor


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

    caterpillar = new_caterpillar()
    locomotion_distance = utils.locomotion_distance_logger(caterpillar)
    for step in range(int(steps)):
        try:
            (phases, frictions, _), action = caterpillar_runner.observe_and_act(actor, caterpillar, [])

            feedbacks, angle_ranges = action[0], action[1]
            caterpillar.set_oscillation_ranges(tuple(angle_ranges))
            caterpillar.step_with_feedbacks(config.params["time_delta"], tuple(feedbacks))
        except Exception as e:
            print("exception occured during sample run,", e)
            continue
        else:
            # Save data
            sim_distance_file.append_data(step, locomotion_distance(caterpillar))
            sim_phase_diffs_file.append_data(step, *utils.phase_diffs(phases))
            sim_actions_file.append_data(step, *action[0])
            sim_frictions_file.append_data(step, *frictions)

    print("Moved distance:", locomotion_distance(caterpillar))
    caterpillar.save_simulation("{}/train_result_ep{}.json".format(log_dir, episode))
