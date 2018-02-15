import numpy as np
from caterpillar_lib import caterpillar


def locomotion_distance_logger(caterpillar: caterpillar.Caterpillar):
    start_pos = extract_caterpillar_position(caterpillar)
    return lambda c: extract_caterpillar_position(c) - start_pos


def extract_caterpillar_position(caterpillar: caterpillar.Caterpillar):
    positions = caterpillar.somite_positions()
    index = int(np.floor(len(positions) / 2))
    # index = len(positions) - 1
    return positions[index][0]


if __name__ == '__main__':
    caterpillar_params = {
        "somite_mass": .3,
        "somite_radius": .35,
        "normal_angular_velocity": np.pi,
        "sp_natural_length": 0.7,
        "sp_k": 80.0,
        "dp_c": 10.0,
        "horizon_ts_k0": 0.,
        "horizon_ts_k1": 0.,
        "vertical_ts_k0": 4.,
        "vertical_ts_k1": 0.,
        "vertical_ts_c": 1.,
        "vertical_realtime_tunable_torsion_spirng_k": 10.,
        "realtime_tunable_ts_rom": np.pi * 1. / 3.,
        "static_friction_coeff": 1.0,
        "dynamic_friction_coeff": 0.1,
        "viscosity_friction_coeff": 3.0,
        "tip_sub_static_friction_coeff": 0.1,
        "tip_sub_dynamic_friction_coeff": 0.01,
        "tip_sub_viscosity_friction_coeff": 1.0,
        "friction_switch_tan": np.tan(np.pi * 1. / 3.),
        # "gripping_phase_threshold": np.sin(np.pi * 5. / 4.),
        "gripping_phase_threshold": np.sin(np.pi * 4. / 4.),
        "gripping_shear_stress_k": 500.,
        "gripping_shear_stress_c": 10.,
    }

    c = caterpillar.Caterpillar(5, (1,2,3), (0,1,2,3,4), caterpillar_params)
    locomotion_distance = locomotion_distance_logger(c)

    # inching
    # for _ in range(75):
    #     c.step_with_feedbacks(0.01, (0, 0, 0), (-np.pi, -np.pi, -np.pi, -np.pi, -np.pi))
    # for _ in range(1000):
    #     c.step_with_feedbacks(0.01, (0, 0, 0), (0, 0, 0, 0, 0))

    # crawling
    for _ in range(1):
        for _ in range(100):
            c.step_with_feedbacks(0.01, (0.4, 3.1, -0.3), (-0.1, -0.5, 1.4, -3.0, 1.35))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (0.3, -2.2, 0.4), (0.7, -0.5, 1.73, 1.13, 1.35))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (0.5, -0.3, -0.1), (-2.1, -0.5, 1.4, -3.0, 0.9))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (0.73, 0.2, 0.8), (-1.9, -0.5, -1.9, .0, 1.35))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (0.21, 1.1, -0.3), (2.1, -0.5, 1.4, -2.0, 1.35))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (-0.43, 1.9, -0.3), (0.1, -0.5, -2.1, -1.23, 1.35))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (-0.8, -1.3, 0.97), (-0.7, -0.5, 1.4, 3.0, 0.))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (0.23, 1.7, -0.3), (-0.6, -0.5, -0.1, 3.0, 1.35))
        # print(c.frictions_x())
        print(c.somite_phases())
        for _ in range(100):
            c.step_with_feedbacks(0.01, (0.4, 2.2, -0.3), (-0.1, -0.5, 1.4, -3.0, 1.35))
        # print(c.frictions_x())
        print(c.somite_phases())
    print("total distance", locomotion_distance(c))

    # c.save_simulation("test_render.json")
