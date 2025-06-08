import numpy as np
from numpy import sqrt

METADATA = {
    'seed': 240202, 
    'mode': 'train',
    # 'mode': 'test',
    'is_discrete': False,
    'scenario_id': 1,
    'end_time': 300,

    'time_step_control': 0.1,
    'time_step_detection': 5,

    'torque_range': [-0.5, 0.5],
    'eloss_range': [0, 1],
    'bias_range': [-0.15, 0.15],

    'max_omega': np.pi / 12,  # radians / s

    'with_noise': True,
    'sigma_ratio': 1e-3,
    'N_actuator': 3,

    'inertia_matrix': np.array([
        [10, 0.02, 0.01],
        [0.02, 15, -0.01],
        [0.01, -0.01, 20]
    ]),

    'config_matrix': np.array([
        [-1, -1, 1, 1],
        [1, -1, -1, 1],
        [1, 1, 1, 1]
    ]),

}

Initial_state = {
    'Test1': [np.array([0.4611, 0.2515, 0.2774, 0.8044]), np.array([-0.0290, -0.0268, 0.0231])],
    'Test2': [np.array([0.3003, -0.3849, 0.2489, 0.8364]), np.array([-0.014, 0.0248, -0.0147])],
    'Test3': [np.array([0.0400, 0.0394, 0.5582, 0.8277]), np.array([0.0286, -0.0445, 0.0380])],
    'Test4': [np.array([-0.1044, -0.1486, 0.4554, 0.8715]), np.array([0.0222, -0.0134, 0.0216])],

}

Fault_time = {
    'Test1': 78,
    'Test2': 92,
    'Test3': 85,
    'Test4': 88,
}

Fault_value = {
    'Test1': np.array([0.11, 0.32, 0.43, 0.03, -0.09, 0.04]),
    'Test2': np.array([0.31, 0.42, 0.23, -0.03, 0.06, 0.04]),
    'Test3': np.array([0.10, 0.22, 0.18, -0.04, -0.02, 0.06]),
    'Test4': np.array([0.05, 0.22, 0.18, 0.05, -0.03, 0.06]),
}

Double_Fault_time = {
    'Test2': 158,
    'Test3': 165,
    'Test4': 162,
}

Double_Fault_value = {
    'Test2': np.array([0.34, 0.46, 0.30, -0.03, 0.06, 0.04]),
    'Test3': np.array([0.10, 0.22, 0.18, -0.06, -0.04, 0.08]),
    'Test4': np.array([0.17, 0.38, 0.26, 0.07, -0.05, 0.09]),
}
