import numpy as np
from . import config
np.seterr(divide='raise')   # error on division by zero
EPSILON = 1.0e-5

class Baseline:
    def __init__(self, init_val: float):
        self.__val = init_val
        self.__alpha = config.params["baseline_moving_average_gamma"]

    def add_new_value(self, val: float) -> float:
        self.__val = self.__alpha * self.__val + (1 - self.__alpha) * val
        return self.__val

    def get_value(self) -> float:
        return self.__val


class PEPG:
    """
    PEPG instance manages parameters and thier standard deviations during optimization.

    With sample_epsilons method, epsilons are sampled from normal distribution whose mean is 0.
    Parameters theta+ = mu + epsilon, theta- = mu - epsilon should be evaluated through such as a simulation,
    and rewards r+, r- respectively should be feeded in order to conduct parameter update.
    Baseline which is moving average of mean reward should be updated then.
    Maximum reward encounted so far, m, should be updated as well.
    Parameters are bound by upper and lower bound specified in config.
    """
    def __init__(self, parameter_number: int):
        self.__mu = np.zeros([parameter_number], dtype=np.float64)
        self.__sigma = np.ones([parameter_number], dtype=np.float64) * config.params["init_sigma"]   # should be non-negative value
        self.__baseline = Baseline(.0)
        self.__maximum_reward = .0
        self.__learning_rate = {
            "mu": config.params["mu_learning_rate"],
            "sigma": config.params["sigma_learning_rate"],
        }
        self.__parameter_bound = {
            "upper": config.params["parameter_upper_bound"],
            "lower": config.params["parameter_lower_bound"],
        }
        self.__sigma_upper_bound = config.params["sigma_upper_bound"]

    def set_parameters(self, new_params: np.array):
        assert self.__mu.shape == new_params.shape,\
            "number of new parameters is inconsistant with the original parameter number, got {} actually expected {}".format(new_params.shape, self.__mu.shape)
        self.__mu = new_params

    def get_parameters(self) -> np.array:
        return self.__mu

    def get_sigmas(self) -> np.array:
        return self.__sigma

    def sample_epsilons(self, sample_number: int) -> np.array:
        """
        This method sample perturbation factor epsilons from normal distribution N(0, sigma^2) and return it.

        epsilons is parameter_number x sample_number
        theta+ = mu + epsilon and theta- = mu - epsilon should be evaluated
        """
        return np.random.normal(loc=.0, scale=self.__sigma[:, np.newaxis], size=[self.__mu.shape[0], sample_number])

    def update_parameters(self, epsilons: np.array, r_plus: np.array, r_minus: np.array):
        """
        This method update internal mu and sigma according to evaluation results of perturbated parameters

        mus, sigmas, baseline, maximum_reward are updated.
        epsilons: parameter_number x sample_number
        r_plus: sample_number
        r_minus: sample_number
        """
        self.__mu += self.mu_delta(epsilons, r_plus, r_minus)
        self.__mu = np.minimum(self.__mu, self.__parameter_bound["upper"])  # prevent too large
        self.__mu = np.maximum(self.__mu, self.__parameter_bound["lower"])  # prevent too small

        self.__sigma += self.sigma_delta(epsilons, r_plus, r_minus)
        self.__sigma = np.minimum(self.__sigma, self.__sigma_upper_bound)  # prevent too large sigma
        self.__sigma = np.maximum(self.__sigma, EPSILON)  # prevent too large sigma
        # assert np.all(self.__sigma > 0), "got negative sigma\n{}".format(self.__sigma)
        print("updated sigma\n", self.__sigma)

        self.__maximum_reward = max(self.__maximum_reward, r_plus.max(), r_minus.max())
        b = self.__baseline.add_new_value((r_plus.mean() + r_minus.mean())/2.)

    def mu_delta(self, epsilons: np.array, r_plus: np.array, r_minus: np.array) -> np.array:
        """
        Calculate mu delta.

        alpha_mu * epsilons x {(r_plus - r_minus) / (2m - r_plus - r_minus)}
        (size: [parameter_number])
        Learning rate scaling is included.
        """
        r = (r_plus - r_minus) / (2 * self.__maximum_reward - r_plus - r_minus)
        return  self.__learning_rate["mu"] * epsilons @ r

    def sigma_delta(self, epsilons: np.array, r_plus: np.array, r_minus: np.array) -> np.array:
        """
        Calculate sigma delta.

        alpha_sigma/(m-b) * {(epsilons^2 - sigmas^2) / sigmas} x {(r_plus + r_minus)/2 - b}
        (size: [parameter_number])
        Learning rate scaling is included.
        """
        S = (np.square(epsilons) - np.square(self.__sigma[:, np.newaxis])) / np.maximum(self.__sigma[:, np.newaxis], EPSILON)
        r = (r_plus + r_minus) / 2. - self.__baseline.get_value()
        if self.__maximum_reward - self.__baseline.get_value() > 0:
            scale = self.__maximum_reward - self.__baseline.get_value()
        else:
            scale = 1.
        delta = self.__learning_rate["sigma"] / scale * S @ r
        if np.any(np.isnan(delta)):
            with open("./nan_in_sigma_delta", 'w') as f:
                print("S: {}\nr: {}\nr_+: {}\nr_-:{}\nbase: {}\nS@r: {}\nscale: {}\ndelta: {}".format(
                S, r, r_plus, r_minus, self.__baseline.get_value(), S@r, scale, delta), file=f)
        assert not np.any(np.isnan(delta)), "got nan in sigma\n{}".format(delta)
        return delta
