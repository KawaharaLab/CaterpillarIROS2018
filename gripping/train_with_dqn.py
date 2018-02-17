import os
import shutil
from optparse import OptionParser
import numpy as np

from gym import spaces
import chainer
from chainer import optimizers
from chainerrl.agents import DoubleDQN
from chainerrl import explorers
from chainerrl import policy
from chainerrl import q_functions
from chainerrl import replay_buffer

from caterpillar_lib.caterpillar import Caterpillar
from controllers import config, utils
from data_csv_saver import DataCSVSaver

F_OUTPUT_BOUND = 1.
GAMMA = 0.95
OU_SIGMA = (F_OUTPUT_BOUND - (-F_OUTPUT_BOUND)) * 0.2
GPU = None
STEPS = 2000

def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def observe(caterpillar: Caterpillar) -> np.array:
    frictions = caterpillar.frictions_x()
    tensions = np.array(caterpillar.tensions()) / 10.
    somite_phases = caterpillar.somite_phases()
    gripper_phases = caterpillar.gripper_phases()
    return np.concatenate((frictions, tensions, np.cos(somite_phases), np.sin(somite_phases), np.cos(gripper_phases), np.sin(gripper_phases))), somite_phases, gripper_phases


def run(agent: DoubleDQN, steps: int, save_dir: str) -> float:
    sim_distance_file = DataCSVSaver(os.path.join(save_dir, "distance.txt"), ("step", "distance"))
    sim_somite_phase_diffs_file = DataCSVSaver(os.path.join(save_dir, "somite_phase_diffs.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_gripper_phase_diffs_file = DataCSVSaver(os.path.join(save_dir, "gripper_phase_diffs.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.grippers)])
    sim_somite_phases_file = DataCSVSaver(os.path.join(save_dir, "somite_phases.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_gripper_phases_file = DataCSVSaver(os.path.join(save_dir, "gripper_phases.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.grippers)])
    sim_somite_actions_file = DataCSVSaver(os.path.join(save_dir, "somite_actions.txt"), ["step"] + ["action_{}".format(i) for i in range(config.oscillators)])
    sim_gripper_actions_file = DataCSVSaver(os.path.join(save_dir, "gripper_actions.txt"), ["step"] + ["action_{}".format(i) for i in range(config.grippers)])
    sim_frictions_file = DataCSVSaver(os.path.join(save_dir, "frictions.txt"), ["step"] + ["friction_{}".format(i) for i in range(config.somites)])
    sim_tension_file = DataCSVSaver(os.path.join(save_dir, "tensions.txt"), ["step"] + ["tension_{}".format(i) for i in range(config.somites - 2)])

    caterpillar = Caterpillar(config.somites, config.oscillators_list, config.grippers_list, config.caterpillar_params)
    locomotion_distance = utils.locomotion_distance_logger(caterpillar)
    for step in range(steps):
        obs, somite_phases, gripper_phases = observe(caterpillar)
        actions = agent.act(obs)
        feedbacks_somite, feedbacks_gripper = actions[:config.oscillators], actions[config.oscillators:]
        caterpillar.step_with_feedbacks(config.params["time_delta"], tuple(feedbacks_somite), tuple(feedbacks_gripper))

        frictions, tensions, _, _ = np.split(
            obs, [config.somites, config.somites * 2 - 2, config.somites * 2 - 2 + config.oscillators])
        sim_distance_file.append_data(step, locomotion_distance(caterpillar))
        sim_somite_phase_diffs_file.append_data(step, *utils.phase_diffs(np.array(somite_phases)))
        sim_gripper_phase_diffs_file.append_data(step, *utils.phase_diffs(np.array(gripper_phases)))
        sim_somite_phases_file.append_data(step, *utils.mod2pi(np.array(somite_phases)))
        sim_gripper_phases_file.append_data(step, *utils.mod2pi(np.array(gripper_phases)))
        sim_frictions_file.append_data(step, *frictions)
        sim_tension_file.append_data(step, *(tensions * 10))
        sim_somite_actions_file.append_data(step, *feedbacks_somite)
        sim_gripper_actions_file.append_data(step, *feedbacks_gripper)

    caterpillar.save_simulation("{}/render.json".format(save_dir))
    return locomotion_distance(caterpillar)


def build_agent() -> DoubleDQN:
    # observation:
    # friction on each somite (#somite)
    # tension on each somite except for both ends (#somite - 2)
    # cos(somite phases), sin(somites phases) (#oscillator x 2)
    # cos(gripper phases), sin(gripper phases) (#gripper x 2)
    obs_size = config.somites + (config.somites - 2) + config.oscillators*2 + config.grippers*2
    # actions: feedbacks to somite oscillators, feedbacks to gripper oscillators
    action_size = config.oscillators + config.grippers
    action_space = spaces.Box(low=-F_OUTPUT_BOUND, high=F_OUTPUT_BOUND, shape=(action_size))

    q_func = q_functions.FCQuadraticStateQFunction(
        obs_size,
        action_size,
        n_hidden_channels=6,
        n_hidden_layers=2,
        action_space=action_space)
    opt = optimizers.Adam()
    opt.setup(q_func)

    explorer = explorers.AdditiveOU(sigma=OU_SIGMA)
    rbuf = replay_buffer.ReplayBuffer(capacity=5 * 10 ** 5)
    phi = lambda x: x.astype(np.float32)

    agent = DoubleDQN(q_func, opt, rbuf, gamma=GAMMA, explorer=explorer,
                 phi=lambda x: x.astype(np.float32), gpu=GPU, replay_start_size=10000)
    return agent


def  train(save_dir_path: str):
    agent = build_agent()

    reset_dir(save_dir_path)
    train_log_dir = os.path.join(save_dir_path, "train_log")
    os.makedirs(train_log_dir, exist_ok=True)

    config.dump_config(train_log_dir, {"RL method": "DoubleDQN"})

    distance_log = DataCSVSaver(os.path.join(train_log_dir, "distance.txt"), ("episode", "distance"))

    ep = 0
    while ep < config.params['episodes']:
        try:
            caterpillar = Caterpillar(config.somites, config.oscillators_list, config.grippers_list, config.caterpillar_params)
            locomotion_distance = utils.locomotion_distance_logger(caterpillar)
            obs, _, _ = observe(caterpillar)
            position = 0 # current position
            reward = 0
            R = 0  # accumulated reward
            t = 0
            while t < STEPS:
                actions = agent.act_and_train(obs, reward)
                feedbacks_somite, feedbacks_gripper = actions[:config.oscillators], actions[config.oscillators:]
                caterpillar.step_with_feedbacks(config.params["time_delta"], tuple(feedbacks_somite), tuple(feedbacks_gripper))

                reward = locomotion_distance(caterpillar) - position
                if np.isnan(reward):
                    print("got invalid reward, {}".format(reward))
                    continue
                obs, _, _ = observe(caterpillar)
                R += reward
                position = position + reward
                t += 1
            print("epoch: {}   R: {}".format(ep+1, R))
            distance_log.append_data(ep+1, R)

            agent.stop_episode_and_train(obs, reward)
        # except FloatingPointError as e:
        #     print("episode {} --- got floating point error, {}. Skip".format(ep, e))
        #     continue
        except KeyboardInterrupt:
            command = input("\nSample? Finish? : ")
            if command in ["sample", "Sample"]:
                steps = input("How many steps for this sample?: ")
                if steps == "":
                    utils.notice("default steps {}".format(config.params["default_sample_steps"]))
                    steps = config.params["default_sample_steps"]

                run_dir = os.path.join(train_log_dir, "train_result_ep{}".format(ep))
                os.makedirs(run_dir, exist_ok=True)
                distance = run(agent, int(steps), run_dir)
                print("test run for {} steps, moved distance {}".format(int(steps), distance))
                continue
            if command in ["finish", "Finish"]:
                print("Ending training ...")
                break
        else:
            ep += 1

    print('Finished. Saving to {}...'.format(save_dir_path))
    agent.save(save_dir_path)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", dest="is_train", action="store_true", default=False, help="To train, specify this flag.")
    parser.add_option("--load", dest="load_dir_path", default=None, help="Directory path to load learned policy.")
    parser.add_option("--steps", dest="steps", default=config.params['default_sample_steps'], type=int, help="Steps to run caterpillar.")
    parser.add_option("--save", dest="save_dir_path", default=None, help="Directory path to save learned policy.")
    opts, args = parser.parse_args()

    if opts.is_train:
        train(opts.save_dir_path)
    else:
        agent = build_agent()
        agent.load(opts.load_dir_path)
        reset_dir(opts.save_dir_path)
        distance = run(agent, opts.steps, opts.save_dir_path)
        print("moved {} per {} steps".format(distance, opts.steps))
