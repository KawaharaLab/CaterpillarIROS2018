from pickle import dump
import ctypes
import platform
import os
import numpy as np

from components.particle import Particle
from components.rts import RTS

np.seterr(all='raise')


# Caterpillar model settings
MASS = .3
RADIUS = .35
RTS_NATURAL_LENGTH_MAX = 0.5
OMGA = np.pi
RTS_K = 100.
RTS_C = 1.
RTS_AMP = .3
RTS_W = np.pi
TS_K = 1000.
FRICTION_APPEAR_LENGTH_RATIO = 0.8
MU = 1
PROTEST_P = 0
EPSILON = .2
SIGMA = .2
ALPHAS = [.0, 10.0]


def import_lib():
    root_path = os.path.dirname((os.path.abspath(__file__)))
    lib_path = os.path.join(root_path, "caterpillar", "target", "release", "libcaterpillar")
    if platform.system() == "Darwin":
        return ctypes.CDLL(lib_path + ".dylib")
    elif platform.system() == "Linux":
        return ctypes.CDLL(lib_path + ".so")
    return None


class Caterpillar:
    class CSOMITE(ctypes.Structure):
        _fields_ = [
            ("position", ctypes.c_double),
            ("verocity", ctypes.c_double),
            ("force", ctypes.c_double),
            ("mass", ctypes.c_double),
            ("friction_coeff", ctypes.c_double),
        ]

        def __str__(self) -> str:
            return "p {},  v {},  f {},  m {}".format(self.position, self.verocity, self.force, self.mass)

    class CRTS(ctypes.Structure):
        _fields_ = [
            ("position_0", ctypes.c_double),
            ("position_1", ctypes.c_double),
            ("verocity_0", ctypes.c_double),
            ("verocity_1", ctypes.c_double),
            ("phase", ctypes.c_double),
            ("natural_length", ctypes.c_double),
            ("angular_verocity", ctypes.c_double),
            ("max_length", ctypes.c_double),
            ("amplitude", ctypes.c_double),
            ("spring_const", ctypes.c_double),
            ("dump_coeff", ctypes.c_double),
        ]

        def __str__(self) -> str:
            return "p0 {},  p1 {}".format(self.position_0, self.position_1)

    def __init__(self, somites_amount: int, dim=3, mode="default", original_head_pos=None,
                 caterpillar_name=""):
        self.__counter = 0
        self.__somites_amount = somites_amount
        self.__caterpillar_name = caterpillar_name
        self.__dim = dim
        self.__somite_mass = MASS
        self.__somite_radius = RADIUS
        self.__rts_max_natural_length = RTS_NATURAL_LENGTH_MAX
        self.__friction_appear_length_ratio = FRICTION_APPEAR_LENGTH_RATIO
        self.__mu = MU
        self.__simulation_protocol = {'objects': [], 'frames': {}}
        self.__dt = 0.01
        if original_head_pos is not None:
            self.__original_head_position = np.array(original_head_pos)
        else:
            self.__original_head_position = np.zeros(self.__dim)
        self.__gravity = np.array([0, 0, 0])

        self.__create_caterpillar(mode=mode)
        self.__original_center_mass = self.center_mass_position

        # Set initial frictions
        self.__frictions = np.array([somite.friction()[0] for somite in self.__somites])
        for somite in self.__somites:
            friction = somite.friction()
            # somite.add_force(friction)  # For _update_caterpillar method
            somite.set_prev_force(friction)

        class CCaterpillarState(ctypes.Structure):
            _fields_ = [
                ("somites", Caterpillar.CSOMITE * somites_amount),
                ("rtses", Caterpillar.CRTS * (somites_amount - 1)),
                ("frictions", ctypes.c_double * somites_amount),
            ]

            def __str__(self):
                desc = "[caterpillar state]\n"
                for i, sm in enumerate(self.somites):
                    desc += "somite {}: {}\n".format(i, str(sm))
                for i, rts in enumerate(self.rtses):
                    desc += "rts {}: {}\n".format(i, str(rts))
                for i, friction in enumerate(self.frictions):
                    desc += "friction {}: {}\n".format(i, friction)
                return desc

        self.__ccaterpillar_state_class = CCaterpillarState

        # Prototype declaration of C codes
        self.__cdll = import_lib()
        self.__cdll.update_caterpillar.argtypes = (ctypes.POINTER(Caterpillar.CSOMITE), ctypes.POINTER(Caterpillar.CRTS), ctypes.c_double)
        self.__cdll.update_caterpillar.restype = CCaterpillarState

    @property
    def center_mass_position(self):
        pos = np.zeros(self.__dim)
        total_mass = 0
        for somite in self.__somites:
            pos += somite.mass * somite.position
            total_mass += somite.mass
        return pos / total_mass

    def update_original_position(self):
        self.__original_center_mass = self.center_mass_position

    def set_phases(self, phis: np.array):
        for phi, rts in zip(phis, self.__rts):
            rts.set_phase(phi)

    def set_to_natural_lengths(self):
        # Update to natural length of current phase
        for rts in self.__rts:
            rts.update_natural_length()
        # Set actual length to natural length of rts'
        self.__somites[0].set_position(np.array([0, 0, self.__somite_radius]))
        prev_somite_pos = self.__somites[0].position
        for somite, rts in zip(self.__somites[1:], self.__rts):
            somite.set_position(prev_somite_pos + np.array([rts.natural_length, 0, 0]))
            prev_somite_pos = somite.position

    @property
    def phis_dim(self) -> int:
        return len(self.__rts)

    @property
    def friction_dim(self) -> int:
        return self.__somites_amount

    @property
    def phis(self) -> np.array:
        """
            Returns: [phi_0, phi_1, phi_2, ...]
        """
        return np.array([rts.phase for rts in self.__rts])

    @property
    def raw_phis(self) -> np.array:
        """
            Returns: [phi_0, phi_1, phi_2, ...]
        """
        return np.array([rts.raw_phase for rts in self.__rts])

    def phases_sin(self) -> np.array:
        """
            Returns: [sin(phi_0), sin(phi_1), sin(phi_2), ...]
        """
        return np.sin(np.array([rts.phase for rts in self.__rts]))

    def phase_diffs(self) -> np.array:
        """
            Returns: [phi_1 - phi_0, phi_2 - phi_1, ...]
        """
        adjacent_phis = zip(self.phis[:-1], self.phis[1:])
        # return np.array(list(map(lambda e: (e[0] - e[1]) % (2 * np.pi), adjacent_phis)))
        return np.array(list(map(lambda e: e[0] - e[1], adjacent_phis)))

    def phases_from_base(self) -> np.array:
        """
            Returns: [0, phi_1 - phi_0, phi_2 - phi_0, ...]
        """
        base_phase = self.raw_phis[0]
        return self.raw_phis - base_phase

    @property
    def frictions(self) -> np.array:
        return self.__frictions

    def moved_distance(self) -> float:
        """
            x diff
        """
        return self.center_mass_position[0] - self.__original_center_mass[0]

    def feedback_phis(self, phase_feedbacks):
        """
            phase_feedbacks: [rts, leg_rts_vertical]
        """
        assert phase_feedbacks.shape[0] == len(self.__rts)
        for rts, phase_d in zip(self.__rts, phase_feedbacks):
            rts.set_phase_d(phase_d)

    def i2somite(self, idx: int) -> str:
        return '{}_somite_{}'.format(self.__caterpillar_name, idx)

    def __create_caterpillar(self, mode="default"):
        # Add somites
        self.__somites = [
            Somite(self.__somite_radius,
                   self.__original_head_position + np.array([0, 0, self.__somite_radius]) + np.array([i * self.__rts_max_natural_length, 0, 0]),
                   self.__somite_mass, self.__somite_radius)
            for i in range(self.__somites_amount)
        ]

        if mode == "crawling":
            self.__somites[1].set_alpha(False)
            self.__somites[5].set_alpha(False)
            self.__somites[6].set_alpha(False)
            self.__somites[7].set_alpha(False)
            self.__somites[11].set_alpha(False)
        elif mode == "inching":
            self.__somites[1].set_alpha(False)
            self.__somites[3].set_alpha(False)
            self.__somites[4].set_alpha(False)
            self.__somites[5].set_alpha(False)
            self.__somites[6].set_alpha(False)
            self.__somites[7].set_alpha(False)
            self.__somites[11].set_alpha(False)

        for i, somite in enumerate(self.__somites):
            if somite is not None:
                self.__simulation_protocol['objects'].append((somite.radius, tuple(somite.position), self.i2somite(i)))

        # Add connections
        self.__rts = []
        for i in range(self.__somites_amount - 1):
            self.__rts.append(RTS(self.__somites[i], self.__somites[i + 1], RTS_NATURAL_LENGTH_MAX, RTS_W, RTS_AMP, RTS_K, RTS_C))

    @property
    def somites_amount(self) -> int:
        return self.__somites_amount

    def update_caterpillar(self):
        # Set somite info for C code
        csomites_ = (Caterpillar.CSOMITE * len(self.__somites))(
            *[Caterpillar.CSOMITE(sm.position[0], sm.verocity[0], sm.prev_force[0], sm.mass, 1.) for sm in self.__somites]
        )
        # Set rts info for C code
        crtses_ = (Caterpillar.CRTS * len(self.__rts))(
            *[Caterpillar.CRTS(
                rts.start_particle.position[0], rts.end_particle.position[0],
                rts.start_particle.verocity[0], rts.end_particle.verocity[0],
                rts.raw_phase, rts.natural_length, rts.phase_d, RTS_NATURAL_LENGTH_MAX, RTS_AMP, RTS_K, RTS_C
            ) for rts in self.__rts]
        )

        caterpillar_state = self.__cdll.update_caterpillar(csomites_, crtses_, self.__dt)
        new_csomites = caterpillar_state.somites
        new_crtses = caterpillar_state.rtses
        self.__frictions = np.array(caterpillar_state.frictions)

        # Set results
        for (sm, csm) in zip(self.__somites, new_csomites):
            sm.set_position(np.array([csm.position, 0, sm.position[2]]))
            sm.set_verocity(np.array([csm.verocity, 0, 0]))
            sm.set_prev_force(np.array([csm.force, 0, 0]))

        for (rts, crts) in zip(self.__rts, new_crtses):
            rts.set_phase(crts.phase)
            rts.set_natural_length(crts.natural_length)

    def _update_caterpillar(self):
        # Update somites positions
        for somite in self.__somites:
            somite.save_prev()
            somite.update_position(self.__dt)
            somite.reset_force()

        # Add friction which depends on rts length
        self.__somites[0].set_mu(self.__rts[0].length)
        for i in range(1, self.__somites_amount - 1):
            self.__somites[i].set_mu(0.5 * (self.__rts[i - 1].length + self.__rts[i].length))
        self.__somites[-1].set_mu(self.__rts[-1].length)

        # Apply frictions to somites
        self.__frictions = [somite.friction()[0] for somite in self.__somites]
        for somite in self.__somites:
            friction = somite.friction()
            somite.add_force(friction)

        # Update theta and intra tension
        for i in range(self.__somites_amount - 1):
            self.__rts[i].update_theta(self.__dt)
            self.__rts[i].update_natural_length()
            self.__rts[i].calc_f()
            self.__somites[i].add_force(-self.__rts[i].force)
            self.__somites[i + 1].add_force(self.__rts[i].force)

        for somite in self.__somites:
            somite.add_force(self.__gravity)
            somite.update_verocity(self.__dt)

    def step(self, frame: int, steps=1):
        cur_frame = []
        if frame > 0:
            self.update_caterpillar()
        if frame % steps == 0:
            for i, somite in enumerate(self.__somites):
                cur_frame.append((self.i2somite(i), tuple(somite.position)))
            self.__simulation_protocol['frames'][frame // steps] = cur_frame

    def reset(self):
        # ToDo(matthewlujp): Only reset positions of spheres
        self.__simulation_protocol = {'objects': [], 'frames': {}}
        self.__create_caterpillar()

    def save_simulation(self, file_path, simulation_proc=None):
        """
            If None for simulation_protocol -> own simulation_protocol
        """
        if simulation_proc is None:
            simulation_proc = self.__simulation_protocol

        with open(file_path, 'wb') as f:
            dump(simulation_proc, f)

    @property
    def simulation_protocol(self) -> dict:
        return self.__simulation_protocol

    def merge_simulation(self, caterpillar) -> dict:
        """
            Return merged tocol
            Non-destructive

            simulation_protocol:
                "objects": [(r0, (x,y,z), 'somite_0'), (r1, (x,y,z), 'somite_1'), ...],
                "frames": {
                    0: [('somite_0', (x,y,z)), ('somite_1', (x,y,z)), ...],
                    1: ...,
                }
        """
        assert self.simulation_protocol["frames"].keys() == caterpillar.simulation_protocol["frames"].keys()   # Frames should be same
        new_simulation_proc = {}
        # Merge objects
        objects = self.simulation_protocol["objects"] + caterpillar.simulation_protocol["objects"]
        new_simulation_proc["objects"] = objects

        # Merge frames
        frames = {}
        for frame_num in self.simulation_protocol["frames"].keys():
            frames[frame_num] = self.simulation_protocol["frames"][frame_num]\
                + caterpillar.simulation_protocol["frames"][frame_num]
        new_simulation_proc["frames"] = frames

        return new_simulation_proc

    @property
    def somites_x_positions(self) -> np.array:
        # return: [s0_x, s1_x, ...]
        return np.array([somite.position[0] for somite in self.__somites])

    def set_friction_coeffs(self, coeffs: np.array):
        assert coeffs.shape == (self.__somites_amount,)
        for coeff, somite in zip(coeffs, self.__somites):
            somite.set_friction_coeff(coeff)

    @property
    def head_x_position(self) -> float:
        return self.__somites[-1].position[0]


class Somite(Particle):
    def __init__(self, radius: float, pos: np.array, mass: float, ground_touch_height: float):
        super().__init__(radius, pos, mass)
        self.__groud_touch_height = ground_touch_height
        self.__alpha = ALPHAS[1]   # Whether has leg
        self.__friction_coeff = 1.   # Friction coefficient
        self.__mu = .0

    @property
    def on_ground(self):
        if self._pos[2] > self.__groud_touch_height * 1.1:
            return False
        return True

    def update_position(self, dt: float):
        mask = np.ones(3, dtype=np.float32)
        if self.on_ground:
            mask[2] = 0.
        self._pos += mask * dt * self.verocity + 0.5 * (dt**2) * self.prev_force / self.mass

    def update_verocity(self, dt: float):
        v = self.verocity + 0.5 * dt * (self._force + self.prev_force) / self.mass
        if self.on_ground and v[2] < 0:
            v[2] = 0
        self.set_verocity(v)

    def friction(self):
        return -self.verocity * self.__mu * self.__friction_coeff

    def set_mu(self, rts_len: float):
        self.__mu = self.__alpha * rts_len

    def set_alpha(self, has_leg: bool):
        assert isinstance(has_leg, bool)
        self.__alpha = ALPHAS[int(has_leg)]

    def set_friction_coeff(self, coeff: float):
        assert isinstance(coeff, float)
        self.__friction_coeff = coeff
