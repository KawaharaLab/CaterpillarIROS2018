import numpy as np


class Particle:
    def __init__(self, radius: float, pos: np.array, mass: float):
        self._radius = radius
        self._dim = pos.shape[0]
        self.__mass = mass

        self._pos = pos
        self.__verocity = np.zeros(self._dim)
        self._force = np.zeros(self._dim)
        self.__prev_force = np.zeros(self._dim)

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def mass(self) -> float:
        return self.__mass

    @property
    def position(self) -> np.array:
        return self._pos

    @property
    def verocity(self) -> np.array:
        return self.__verocity

    @property
    def prev_force(self) -> np.array:
        return self.__prev_force

    def set_mass(self, mass: float):
        self.__mass = mass

    def set_radius(self, radius: float):
        self._radius = radius

    def set_position(self, pos: np.array):
        assert self._dim == pos.shape[0]
        self._pos = pos

    def reset_force(self):
        self._force = np.zeros(self._dim)

    def set_prev_force(self, force: np.array):
        self.__prev_force = force

    def add_force(self, force: np.array):
        assert self._dim == force.shape[0]
        self._force += force

    def update_position(self, dt: float):
        self._pos += dt * self.__verocity + 0.5 * (dt**2) * self.__prev_force / self.__mass

    def update_verocity(self, dt: float):
        self.__verocity = self.__verocity + 0.5 * dt * (self._force + self.__prev_force) / self.__mass

    def set_verocity(self, v: np.array):
        self.__verocity = v

    def save_prev(self):
        self.__prev_force = self._force
