import numpy as np

from . import config
from . import base_actor

np.seterr(all="raise")


class Actor(base_actor.BaseActor):
    def __str__(self):
        return "SimpleNeuralNetworkActor"

    def _build_network(self):
        def action_infer(state: np.array) -> np.array:
            """
            Get state and return feedback.

            state: [f_0, f_1, ..., phi_0, phi_1, ..., t_0, t_1, ...]
            t_i is tension on i th RTS
            """
            forces = state[:config.somites]
            phis = state[config.somites:config.somites + config.oscillators]

            adj_forces = (forces[:-1] + forces[1:]) / 2.
            return - .2 * .5 * .3 * adj_forces * np.sin(phis)

        return action_infer
