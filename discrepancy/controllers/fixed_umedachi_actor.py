from .config import config
from .base_actor import BaseActor
import numpy as np

np.seterr(all="raise")


class FixedUmedachiActor(BaseActor):
    def __str__(self):
        return "SimpleNeuralNetworkActor"

    def _build_network(self):
        def action_infer(state: np.array) -> np.array:
            """
                state: [f_0, f_1, ..., phi_0, phi_1, ...]
            """
            forces = state[:config.params["somites"]]
            phis = state[config.params["somites"]:]

            adj_forces = (forces[:-1] + forces[1:]) / 2.
            return - .2 * .5 * .3 * adj_forces * np.sin(phis)

        return action_infer
