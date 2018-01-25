import numpy as np
from datetime import datetime
import os


class ReinforceConfig:
    def __init__(self):
        self.__somites = 5
        self.__phis = self.__somites - 1

        # Algorith related params (common among actor and critic)
        self.__params = {
            "somites": self.__somites,
            "oscillators": self.__phis,
            "state_dim": self.__somites + self.__phis,
            "action_dim": self.__phis,
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

        # Set actor
        self.__params["actor_module_name"] = "rl_algo.nn_actor"
        self.__params["actor_class_name"] = "NeuralNetworkActor"
        self.__params["actor_params"] = {"save_summary": True}

        # Execution params such number of processes
        self.__exec_params = {
            "worker_processes": 8,
            "save_params": True,
        }

        # File paths and etc.
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.__log_dir_root = "{}/logs/nn_model".format(root_path)
        self.__model_dir_root = "{}/models/nn_model".format(root_path)
        self.__path_samples_root = "{}/path_samples".format(root_path)
        self.__simulation_result_root = "{}/simulation_results".format(root_path)
        self.__timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def actor_params(self) -> dict:
        try:
            return self.__actor_params
        except AttributeError:
            params = {
                "in_dim": self.__params["somites"],
                "out_dim": self.__params["oscillators"],
            }

            params["init"] = {
                # NeuralNetworkActor
                "layer0_w": np.zeros((params["in_dim"], 10), dtype=np.float32),
                "layer0_b": np.zeros(10, dtype=np.float32),
                "layer1_w": np.zeros((10, params["out_dim"]), dtype=np.float32),
                "layer1_b": np.zeros(params["out_dim"], dtype=np.float32),
                # LinearActor
                "w_cos": np.zeros((params["in_dim"], params["out_dim"]), dtype=np.float32),
                "w_sin": np.zeros((params["in_dim"], params["out_dim"]), dtype=np.float32),
                # LocalLinearActor
                "wl_cos": np.zeros(params["out_dim"], dtype=np.float32),
                "wl_sin": np.zeros(params["out_dim"], dtype=np.float32),
                # UmedachiActor
                "alpha": .0,
                # FixedUmedachiActor
                "sigma": .2,
                "L_bar": .5,
                "A": 0.3,
            }

            self.__actor_params = params
            return self.__actor_params

    @property
    def random_path_train_params(self) -> dict:
        try:
            return self.__random_path_train_params
        except AttributeError:
            params = {
                "path_len": 20,
                "segment_len": 0.35 * 2 * 2,   # Twice the diameter of a somite
                "samples_per_one_params_set": 10,
            }

            self.__random_path_train_params = params
            return self.__random_path_train_params

    @property
    def path_params(self) -> dict:
        try:
            return self.__path_params
        except AttributeError:
            params = {
                "sample_paths_num": 100,
                "path_len": 20,
                "segment_len": 0.35 * 2 * 2,   # Twice the diameter of a somite
                "friction_coeff_min": .0,
                "friction_coeff_max": 10.,
                # "actor": {"module": "rl_algo.nn_actor", "class": "NeuralNetworkActor"},
                "actor": {"module": "rl_algo.fixed_umedachi_actor", "class": "FixedUmedachiActor"},
                "trials_for_one_actor": 5,
                "trials_for_one_path": 5,
            }

            params["samples_dir"] = "{}/{}".format(self.__path_samples_root, self.__timestamp)

            self.__path_params = params
            return self.__path_params

    def reset_path_sample_dir(self, dir_path: str):
        self.__path_params["samples_dir"] = dir_path

    @property
    def params(self) -> dict:
        return self.__params

    @property
    def exec_params(self):
        return self.__exec_params

    @staticmethod
    def notice(notice_text):
        notice_style = "\x1b[0;31;45m Notice: {} \x1b[0m"
        print(notice_style.format(notice_text))

    @property
    def trial_name(self) -> str:
        try:
            return self.__trial_name
        except AttributeError:
            self.notice("No trial name is specified.")
            self.__trial_name = ""
            return self.__trial_name

    def set_trial_name(self, name: str):
        self.__trial_name = name

    def model_file(self) -> str:
        return "{}/actor_model.ckpt".format(self.model_dir())

    def model_dir(self) -> str:
        try:
            return self.__model_dir
        except AttributeError:
            self.__model_dir = "{}/{}_{}".format(
                self.__model_dir_root, self.trial_name, self.__timestamp)
            if not os.path.exists(self.__model_dir):
                os.makedirs(self.__model_dir)
                print("Created {}".format(self.__model_dir))
            return self.__model_dir

    def log_dir(self) -> str:
        try:
            return self.__log_dir
        except AttributeError:
            self.__log_dir = "{}/{}_{}".format(
                self.__log_dir_root, self.trial_name, self.__timestamp)
            if not os.path.exists(self.__log_dir):
                os.makedirs(self.__log_dir)
                print("Created {}".format(self.__log_dir))
            return self.__log_dir

    def metrics_dir(self):
        try:
            return self.__metrics_dir
        except AttributeError:
            self.__metrics_dir = "{}/data".format(self.log_dir())
            if not os.path.exists(self.__metrics_dir):
                os.makedirs(self.__metrics_dir)
                print("Created {}".format(self.__metrics_dir))
            return self.__metrics_dir

    def simulation_result_dir(self) -> str:
        try:
            return self.__simulation_result_dir
        except AttributeError:
            self.__simulation_result_dir = "{}/{}".format(self.__simulation_result_root, self.__timestamp)
            os.makedirs(self.__simulation_result_dir)
            print("Created {}".format(self.__simulation_result_dir))
            return self.__simulation_result_dir

    def dump_config(self, dir_path=None):
        if dir_path is None:
            dir_path = self.log_dir()
        with open("{}/config_dump.txt".format(dir_path), 'w') as f:
            f.write("\n---General config--------------\n")
            f.write(str(self.__params))
            f.write("\n---Actor config----------------\n")
            f.write(str(self.actor_params))


config = ReinforceConfig()
