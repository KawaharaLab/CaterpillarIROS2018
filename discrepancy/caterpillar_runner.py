from importlib import import_module
import shutil
from optparse import OptionParser
import os
import numpy as np

from controllers import config, base_actor
from caterpillar import Caterpillar
from data_csv_saver import DataCSVSaver

np.seterr(all='raise')


def observe_and_act(actor: base_actor.BaseActor, caterpillar: Caterpillar, disable_list: list,
                    broken_value=0, episode=None):
    assert np.all(np.array(disable_list) < config.somites)

    if disable_list is None:
        disable_list = []
    mask = np.ones(config.somites)
    mask[disable_list] = 0
    bias = np.zeros(config.somites)
    bias[disable_list] = broken_value

    frictions = caterpillar.frictions
    phis = caterpillar.phis
    state = np.concatenate([frictions * mask + bias, phis])
    action = actor.get_action(state)
    caterpillar.feedback_phis(action)

    observation = (frictions, phis)
    return observation, action


# Reinforcement Learning
def run_caterpillar(actor, save_dir: str, steps: int, disable_list=None, broken_value=0):
    if disable_list is None:
        disable_list = []

    caterpillar = Caterpillar(config.somites, mode="default")

    # Record during sampling
    sim_distance_file = DataCSVSaver(os.path.join(save_dir, "distance.txt"), ("step", "distance"))
    sim_phase_diffs_file = DataCSVSaver(os.path.join(save_dir, "phase_diffs.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_phases_file = DataCSVSaver(os.path.join(save_dir, "phases.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_actions_file = DataCSVSaver(os.path.join(save_dir, "actions.txt"), ["step"] + ["action_{}".format(i) for i in range(config.oscillators)])
    sim_frictions_file = DataCSVSaver(os.path.join(save_dir, "frictions.txt"), ["step"] + ["friction_{}".format(i) for i in range(config.somites)])

    for step in range(steps):
        _, action = observe_and_act(actor, caterpillar, disable_list=disable_list, broken_value=broken_value)
        caterpillar.step(step, int(1 / config.params["time_dalte"] / 10))

        # Save data
        sim_distance_file.append_data(step, caterpillar.moved_distance())
        sim_phase_diffs_file.append_data(step, *caterpillar.phases_from_base())
        sim_phases_file.append_data(step, *caterpillar.phis)
        sim_actions_file.append_data(step, *action)
        frictions = caterpillar.frictions
        sim_frictions_file.append_data(step, *frictions)

    caterpillar.save_simulation("{}/render.sim".format(save_dir))

    return caterpillar.moved_distance()


# Delete directory and recreate
def reset_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--module", dest="module_name")
    parser.add_option("--model", dest="model_file_path")
    parser.add_option("--steps", default=10000, dest="steps", type="int")
    parser.add_option("--save", default="./run_result", dest="save_dir")
    parser.add_option("-d", action="append", dest="disable_list", default=[], type=int)
    parser.add_option("--broken_value", dest="broken_value", default=0, type="float")
    opts, args = parser.parse_args()

    reset_dir(opts.save_dir)

    # import actor
    module_ = import_module(opts.module_name)
    actor_class = getattr(module_, config.COMMON_ACTOR_NAME)
    actor = actor_class(actor_model_file=opts.model_file_path)

    if opts.disable_list != []:
        print("force sensor disable list", opts.disable_list)
    moved_distance = run_caterpillar(actor, opts.save_dir, opts.steps, opts.disable_list, opts.broken_value)
    announce = "moved {} (per {} steps)".format(moved_distance, opts.steps)
    print(announce)

    with open("{}/report".format(opts.save_dir), 'w') as f:
        f.write(announce)
