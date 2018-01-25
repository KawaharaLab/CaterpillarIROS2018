import numpy as np


class PEPG:
    def __init__(self, params_num: int, learning_rate=0.1, gamma=0.01, boudn_f=lambda x: x,
                 fin_sigma=0.01, init_sigma=1.):
        self.__params_num = params_num
        self.__mus = np.zeros(params_num, dtype=np.float32)
        self.__sigmas = init_sigma * np.ones(params_num, dtype=np.float32)
        self.__baseline = .0
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__bound_f = boudn_f
        self.__fin_sigma = fin_sigma

    def mus(self) -> np.array:
        return self.__mus

    def sigmas(self) -> np.array:
        return self.__sigmas

    def baseline(self) -> float:
        return self.__baseline

    def is_converged(self):
        return np.all(
            self.__sigmas < self.__fin_sigma * np.ones(self.__params_num, dtype=np.float32))

    def set_mus(self, vals: np.array):
        assert vals.shape == (self.__params_num,)
        self.__mus = self.__bound_f(vals)

    def _update_mus(self, val):
        assert val.shape == (self.__params_num,)
        self.__mus = self.__bound_f(self.__mus + self.__learning_rate * val)

    def _update_sigmas(self, val, bound):
        assert val.shape == (self.__params_num,)
        self.__sigmas = self.__sigmas + self.__learning_rate * val
        if np.any(self.__sigmas > bound):
            print("\x1b[6;30;41mSigma overflow. Bounded to {}.\x1b[0m".format(bound))
        self.__sigmas = np.minimum(self.__sigmas, np.absolute(bound))
        self.__sigmas = np.maximum(self.__sigmas, -np.absolute(bound))

    @property
    def parameters(self):
        return self.__mus

    @property
    def params_num(self):
        return self.__params_num

    def sample_batch(self, batch_size: int):
        # return: [batch x params_num]
        self.__current_params = np.random.normal(
            loc=self.__mus, scale=np.square(self.__sigmas), size=(batch_size, self.__params_num))
        self.__current_params = self.__bound_f(self.__current_params)
        return self.__current_params

    def update_params(self, rewards: np.array, sigma_bound=np.inf):
        try:
            assert len(rewards) == self.__current_params.shape[0]
        except AttributeError as e:
            print("\x1b[6;30;41mCall sample_batch first.\x1b[0m")
            raise e

        T = (self.__current_params - self.__mus).T
        S = (np.square(T) - np.square(self.__sigmas)[:, np.newaxis]) / self.__sigmas[:, np.newaxis]
        r = rewards - self.__baseline
        self._update_mus(T @ r)
        self._update_sigmas(S @ r, sigma_bound)
        self._update_baseline(rewards)

    def _update_baseline(self, rewards: np.array):
        mean_reward = np.mean(rewards)
        self.__baseline = self.moving_average(self.__baseline, mean_reward, {'gamma': self.__gamma})

    @staticmethod
    def moving_average(average, new_value, params):
        """Exponential moving average."""
        return params['gamma'] * new_value + (1 - params['gamma']) * average


class SymmetricPEPG(PEPG):
    def __init__(self, params_num: int, learning_rate=0.1, gamma=0.01, boudn_f=lambda x: x,
                 fin_sigma=0.01, init_sigma=2.):
        super().__init__(params_num, learning_rate, gamma, boudn_f, fin_sigma, init_sigma)
        self.__max_reward = .0

    def sample_batch(self, batch_size: int):
        """
        Sample several parameters sets.

        return: [
            (mus + peturbations_(0)),...,(mus + peturbations_(n-1)),
            (mus - peturbations_(0)),...,(mus - peturbations)_(n-1)
        ]
        """
        self.__perturbation = np.random.normal(
            loc=0, scale=np.square(self._sigmas), size=(batch_size, self.params_num))
        return np.concatenate(
            (self._mus + self.__perturbation, self._mus - self.__perturbation), axis=0)

    def _update_max_reward(self, rewards: np.array):
        self.__max_reward = max(self.__max_reward, np.max(rewards))

    def update_params(self, rewards: np.array, sigma_bound=np.inf):
        """
        Update parameters based on rewards so that expected reward rise.

        rewards: [r(mus + peturbations_0),..., r(mus - perturbation_0)]
        """
        try:
            assert len(rewards) == self.__perturbation.shape[0] * 2
        except AttributeError as e:
            print("\x1b[6;30;41mCall update_params first.\x1b[0m")
            raise e

        self._update_max_reward(rewards)

        T = self.__perturbation.T
        S = ((np.square(self.__perturbation) - np.square(self._sigmas)) / np.maximum(self._sigmas, 0.0001)).T
        rewards_p, rewards_m = rewards[:len(self.__perturbation)], rewards[len(self.__perturbation):]
        r_m = (rewards_p + rewards_m)
        r_t = (rewards_p - rewards_m)
        r_s = r_m / 2. - self._baseline
        if self.__max_reward > .0:
            self._update_mus(T @ (r_t / (2 * self.__max_reward - r_m)))
            self._update_sigmas(S @ r_s / (self.__max_reward - self.baseline), sigma_bound)
        else:
            self._update_mus(T @ r_t)
            self._update_sigmas(S @ r_s, sigma_bound)
        self._update_baseline(rewards)
