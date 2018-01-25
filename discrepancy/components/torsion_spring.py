import numpy as np
import math


# ToDo(matthewlujp): Fix torsion spring


class TorsionSpring:
    def __init__(self, p0: np.array, p1: np.array, p2: np.array, natural_angle=math.pi, k=1000.):
        """
        p0 is the middle of a torsion spring.
        For simplicity, only deal with 2d.
        """
        self._dim = 2
        self._natural_angle = natural_angle
        self._k = k

        self._angle = math.pi
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2

        self._f0 = np.zeros(self._dim)
        self._f1 = np.zeros(self._dim)
        self._f2 = np.zeros(self._dim)

    @staticmethod
    def _rotate(theta: float, x: np.array):
        """
        theta: radian
        x: 2d
        """
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.matmul(R, x)

    @staticmethod
    def _get_angle(v0, v1):
        cross = float(np.cross(v0, v1))
        dot = float(np.dot(v0, v1))
        return np.arctan2(cross, dot)

    @staticmethod
    def normal_vector(r):
        d = np.linalg.norm(r)
        n = TorsionSpring._rotate(math.pi / 2, r / d)
        return n

    @property
    def force(self):
        return self._f0, self._f1, self._f2

    @property
    def angle(self):
        """
        Angle of p1 to p2.
        Measured clock-wise.
        """
        v_01 = self._p1.position[:2] - self._p0.position[:2]
        v_02 = self._p2.position[:2] - self._p0.position[:2]
        ang = self._get_angle(v_01, -v_02) + math.pi
        return ang % 2 * math.pi

    def calc_f(self):
        ang_diff = self.angle - self._natural_angle
        d_01 = np.linalg.norm(self._p1.position[:2] - self._p0.position[:2])
        d_02 = np.linalg.norm(self._p2.position[:2] - self._p0.position[:2])
        n_01 = self.normal_vector(self._p1.position[:2] - self._p0.position[:2])
        n_02 = self.normal_vector(self._p2.position[:2] - self._p0.position[:2])

        self._f1 = self._k * (ang_diff) * n_01 / d_01
        self._f2 = -self._k * (ang_diff) * n_02 / d_02
        self._f0 = -(self._f1 + self._f2)
