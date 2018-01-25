from itertools import chain
import numpy as np

from . import config
from . import pepg


class PEPGActorManager:
    def __init__(self, model_path, actor_class):
        self.__actor = actor_class()
        self.__pepg = pepg.SymmetricPEPG(self.__actor.params_num(),
                                         learning_rate=config.params["learning_rate"],
                                         init_sigma=config.params["init_sigma"],
                                         fin_sigma=config.params["converged_sigma"])

        # Set initialized values in actor network as pepg params
        flatten_params = np.array(
            list(chain.from_iterable([
                p.flatten().tolist() for p in self.__actor.current_params()
            ]))
        )
        self.__pepg.set_mus(flatten_params)

    def predict(self, states, step=None):
        return self.__actor.get_action(states, step=step)

    def sample_params(self, batch_size=None):
        """Sample params from Gaussian distribution."""
        if batch_size is None:
            batch_size = config.params["batch_size"]
        return self.__pepg.sample_batch(batch_size)

    def update_params(self, rewards: np.array):
        self.__pepg.update_params(rewards, config.params["sigma_upper_bound"])

    def set_params(self, params: list):
        # params: [params_00, params_01, ..., params_10, ...]
        self.__actor.set_params(params)

    @property
    def parameters(self):
        return self.__pepg.parameters

    @property
    def sigmas(self):
        return self.__pepg.sigmas()

    def get_actor(self):
        return self.__actor

    def is_converged(self) -> bool:
        self.__pepg.is_converged()

    def save(self, save_file_path=None, episode=None):
        self.__actor.save(save_file_path, episode)
