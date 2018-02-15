from importlib import import_module
import shutil
from optparse import OptionParser
import os
import numpy as np

from controllers import config, base_actor, utils
from caterpillar_lib.caterpillar import Caterpillar
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

    frictions = caterpillar.frictions_x()
    tensions = caterpillar.tensions()
    somite_phases = caterpillar.somite_phases()
    gripper_phases = caterpillar.gripper_phases()
    phases = np.concatenate((somite_phases, gripper_phases))
    state = np.concatenate([frictions * mask + bias, phases, tensions])
    action = actor.get_action(state)

    observation = (np.concatenate((somite_phases, gripper_phases)), frictions, tensions)
    return observation, action


# Reinforcement Learning
def run_caterpillar(actor, save_dir: str, steps: int, disable_list=None, broken_value=0):
    if disable_list is None:
        disable_list = []

    caterpillar = Caterpillar(config.somites, config.oscillators_list, config.grippers_list, config.caterpillar_params)

    # Record during sampling
    sim_distance_file = DataCSVSaver(os.path.join(save_dir, "distance.txt"), ("step", "distance"))
    sim_somite_phase_diffs_file = DataCSVSaver(os.path.join(save_dir, "somite_phase_diffs.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_gripper_phase_diffs_file = DataCSVSaver(os.path.join(save_dir, "gripper_phase_diffs.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.grippers)])
    sim_somite_phases_file = DataCSVSaver(os.path.join(save_dir, "somite_phases.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_gripper_phases_file = DataCSVSaver(os.path.join(save_dir, "gripper_phases.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.grippers)])
    sim_somite_actions_file = DataCSVSaver(os.path.join(save_dir, "somite_actions.txt"), ["step"] + ["action_{}".format(i) for i in range(config.oscillators)])
    sim_gripper_actions_file = DataCSVSaver(os.path.join(save_dir, "gripper_actions.txt"), ["step"] + ["action_{}".format(i) for i in range(config.grippers)])
    sim_frictions_file = DataCSVSaver(os.path.join(save_dir, "frictions.txt"), ["step"] + ["friction_{}".format(i) for i in range(config.somites)])
    sim_tension_file = DataCSVSaver(os.path.join(save_dir, "tensions.txt"), ["step"] + ["tension_{}".format(i) for i in range(config.somites - 2)])
    sim_grip_thresholds_file = DataCSVSaver(os.path.join(save_dir, "grip_thresholds.txt"), ["step"] + ["gripper_{}".format(i) for i in range(config.grippers)])

    locomotion_distance = utils.locomotion_distance_logger(caterpillar) # closure to keep locomotion distance
    for step in range(steps):
        obv, action = observe_and_act(actor, caterpillar, disable_list=disable_list, broken_value=broken_value)
        feedbacks, _, gripping_phase_thresholds = action[0], action[1], action[2]
        caterpillar.set_gripping_phase_thresholds(tuple(gripping_phase_thresholds))
        caterpillar.step_with_feedbacks(config.params["time_delta"], tuple(feedbacks[:config.oscillators]), tuple(feedbacks[config.oscillators:]))

        # Save data
        phases, frictions, tensions = obv
        sim_distance_file.append_data(step, locomotion_distance(caterpillar))
        sim_somite_phase_diffs_file.append_data(step, *utils.phase_diffs(phases[:config.oscillators]))
        sim_gripper_phase_diffs_file.append_data(step, *utils.phase_diffs(phases[config.oscillators:]))
        sim_somite_phases_file.append_data(step, *utils.mod2pi(phases[:config.oscillators]))
        sim_gripper_phases_file.append_data(step, *utils.mod2pi(phases[config.oscillators:]))
        sim_frictions_file.append_data(step, *frictions)
        sim_tension_file.append_data(step, *tensions)
        sim_somite_actions_file.append_data(step, *feedbacks[:config.oscillators])
        sim_gripper_actions_file.append_data(step, *feedbacks[config.oscillators:])
        sim_grip_thresholds_file.append_data(step, *gripping_phase_thresholds)

    caterpillar.save_simulation("{}/render.json".format(save_dir))

    return locomotion_distance(caterpillar)


# Delete directory and recreate
def reset_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--module", dest="module_name", help="Module name of actor model to use.")
    parser.add_option("--model", default=None, dest="model_file_path", help="Path to model file.")
    parser.add_option("--steps", default=10000, dest="steps", type="int", help="Steps of this run.")
    parser.add_option("--save", default="./run_result", dest="save_dir", help="Path to save directory.")
    parser.add_option("-d", action="append", dest="disable_list", default=[], type=int, help="List of sensors to disable.")
    parser.add_option("--broken_value", dest="broken_value", default=0, type="float", help="Fixed value if sensors are disabled.")
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
