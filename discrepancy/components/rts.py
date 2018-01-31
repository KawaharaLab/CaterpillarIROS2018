from .particle import Particle
import numpy as np
import math
import ctypes
import platform
import os


def import_lib():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(root_path, "caterpillar", "target", "release", "libcaterpillar")
    if platform.system() == "Darwin":
        return ctypes.CDLL(lib_path + ".dylib")
    elif platform.system() == "Linux":
        return ctypes.CDLL(lib_path + ".so")
    return None


class RTS:
    def __init__(self, p0: Particle, p1: Particle, natural_length_max: float, w: float, amp: float,
                 k: float, c: float):
        self.__dim = p0.position.shape[0]
        self._w = w
        self.__natural_length_max = natural_length_max
        assert amp <= 1.
        self.__amp = amp
        self._k = k
        self._c = c

        self.__p0 = p0
        self.__p1 = p1
        self.__phase = .0
        self._natural_length = natural_length_max
        self.__phase_d = self._w
        self._force = np.zeros(self.__dim)

        self.__cdll = import_lib()
        self.__cdll.calculate_force.argtypes = (
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
        self.__cdll.calculate_force.restype = ctypes.c_double

    @property
    def start_particle(self) -> Particle:
        return self.__p0

    @property
    def end_particle(self) -> Particle:
        return self.__p1

    @property
    def length(self):
        return np.linalg.norm(self.__p0.position - self.__p1.position)

    @property
    def natural_length(self) -> float:
        """
            Only returns the natural length. Doesn't update.
        """
        return self.__natural_length_max * (1. + self.__amp * (np.cos(self.__phase) - 1.))

    @property
    def force(self):
        return self._force

    @property
    def phase(self):
        return self.__phase % (2 * np.pi)

    @property
    def raw_phase(self):
        return self.__phase

    @property
    def phase_d(self):
        return self.__phase_d

    @property
    def direction(self):
        return (self.__p1.position - self.__p0.position) / self.length

    def reset_force(self):
        self._force = np.zeros(self.__dim)

    def set_phase(self, phase: float):
        self.__phase = phase

    def set_phase_d(self, feedback: float):
        self.__phase_d = self._w + feedback

    def set_k(self, k: float):
        self._k = k

    def set_c(self, c: float):
        self._c = c

    def set_natural_length_max(self, max_length: float):
        self.__natural_length_max = max_length

    def set_natural_length(self, length: float):
        self._natural_length = length

    def set_amp(self, amp: float):
        assert amp <= 1.
        self.__amp = amp

    def update_theta(self, dt):
        self.__phase += self.__phase_d * dt

    def update_natural_length(self):
        self._natural_length = self.__natural_length_max * (1. + self.__amp * (np.cos(self.__phase) - 1.))

    def calc_f(self):
        x_force = self.__cdll.calculate_force(
            self.__p0.position[0], self.__p1.position[0], self.__p0.verocity[0], self.__p1.verocity[0],
            self.natural_length, self._k, self._c,
        )
        self._force = np.array([x_force, 0, 0])

    def calc_tension(self) -> np.array:
        x_force = self.__cdll.calculate_force(
            self.__p0.position[0], self.__p1.position[0], self.__p0.verocity[0], self.__p1.verocity[0],
            self.natural_length, self._k, self._c,
        )
        return x_force

        # self.reset_force()
        # l = self.length
        # if l > 0.0001:
        #     diff_direction = (self.__p1.position - self.__p0.position) / l
        #     r_target = self._natural_length * diff_direction + self.__p0.position
        #     self._force = - self._k * (self.__p1.position - r_target)
        # else:
        #     self._force = np.zeros(self.__dim)
        #
        # self._force -= self._c * (self.__p1.verocity - self.__p0.verocity)
