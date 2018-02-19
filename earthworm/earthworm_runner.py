from importlib import import_module
import shutil
from optparse import OptionParser
import os
import numpy as np

from controllers import config, base_actor, utils
from earthworm_lib.earthworm import Earthworm
from data_csv_saver import DataCSVSaver

np.seterr(all='raise')


def observe_and_act(actor: base_actor.BaseActor, earthworm: Earthworm, disable_list: list,
                    broken_value=0, episode=None):
    assert np.all(np.array(disable_list) < config.somites)

    if disable_list is None:
        disable_list = []
    mask = np.ones(config.somites)
    mask[disable_list] = 0
    bias = np.zeros(config.somites)
    bias[disable_list] = broken_value

    frictions = earthworm.frictions_x()
    tensions = earthworm.tensions()
    phases = earthworm.somite_phases()
    state = np.concatenate([frictions * mask + bias, phases, tensions])
    action = actor.get_action(state)

    observation = (np.array(phases), np.array(frictions), np.array(tensions))
    return observation, action


# Reinforcement Learning
def run_earthworm(actor, save_dir: str, steps: int, disable_list=None, broken_value=0):
    if disable_list is None:
        disable_list = []

    earthworm = Earthworm(config.somites, config.oscillators_list, config.earthworm_params)

    # Record during sampling
    sim_distance_file = DataCSVSaver(os.path.join(save_dir, "distance.txt"), ("step", "distance"))
    sim_somite_phase_diffs_file = DataCSVSaver(os.path.join(save_dir, "somite_phase_diffs.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_somite_phases_file = DataCSVSaver(os.path.join(save_dir, "somite_phases.txt"), ["step"] + ["phi_{}".format(i) for i in range(config.oscillators)])
    sim_somite_actions_file = DataCSVSaver(os.path.join(save_dir, "somite_actions.txt"), ["step"] + ["action_{}".format(i) for i in range(config.oscillators)])
    sim_frictions_file = DataCSVSaver(os.path.join(save_dir, "frictions.txt"), ["step"] + ["friction_{}".format(i) for i in range(config.somites)])
    sim_tension_file = DataCSVSaver(os.path.join(save_dir, "tensions.txt"), ["step"] + ["tension_{}".format(i) for i in range(config.somites - 2)])

    locomotion_distance = utils.locomotion_distance_logger(earthworm) # closure to keep locomotion distance
    for step in range(steps):
        obv, action = observe_and_act(actor, earthworm, disable_list=disable_list, broken_value=broken_value)
        # for (oscillator_id, target_angle) in config.fixed_angles.items():
        #     earthworm.set_target_angle(oscillator_id, target_angle)
        earthworm.step_with_feedbacks(config.params["time_delta"], tuple(action))

        # Save data
        phases, frictions, tensions = obv
        sim_distance_file.append_data(step, locomotion_distance(earthworm))
        sim_somite_phase_diffs_file.append_data(step, *utils.phase_diffs(phases[:config.oscillators]))
        sim_somite_phases_file.append_data(step, *utils.mod2pi(phases[:config.oscillators]))
        sim_frictions_file.append_data(step, *frictions)
        sim_tension_file.append_data(step, *tensions)
        sim_somite_actions_file.append_data(step, *feedbacks[:config.oscillators])

    earthworm.save_simulation("{}/render.json".format(save_dir))

    return locomotion_distance(earthworm)


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
    moved_distance = run_earthworm(actor, opts.save_dir, opts.steps, opts.disable_list, opts.broken_value)
    announce = "moved {} (per {} steps)".format(moved_distance, opts.steps)
    print(announce)

    with open("{}/report".format(opts.save_dir), 'w') as f:
        f.write(announce)
