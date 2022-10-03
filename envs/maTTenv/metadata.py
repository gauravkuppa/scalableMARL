import numpy as np

# easy
METADATA_v1 = {   
        'version' : 1,
        'sensor_r': 10.0,
        'comms_r': 15.0,
        'fov' : 360,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 1.0, # velocity limit of targets.
        'init_distance_min': 5.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 10.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2, -np.pi], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
        'coverage_reward': 25.0 # adding coverage reward
    }

# medium
METADATA_v2 = {
        'version' : 1,
        'sensor_r': 10.0,
        'comms_r': 15.0,
        'fov' : 360,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 1.0, # velocity limit of targets.
        'init_distance_min': 10.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 20.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2, -np.pi], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
        'coverage_reward': 25.0 # adding coverage reward
    }

# hard
METADATA_v3 = {   
        'version' : 1,
        'sensor_r': 10.0,
        'comms_r': 15.0,
        'fov' : 360,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'init_distance_min': 5.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 10.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2, -np.pi], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
        'coverage_reward': 25.0 # adding coverage reward
    }

# really hard
METADATA_v4 = {
        'version' : 1,
        'sensor_r': 10.0,
        'comms_r': 15.0,
        'fov' : 360,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.5, # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'init_distance_min': 10.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 20.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2, -np.pi], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
        'coverage_reward': 0.0 # adding coverage reward
    }


# really hard
METADATA_v4_no_comms = {
        'version' : 1,
        'sensor_r': 10.0,
        'comms_r': 10.0,
        'fov' : 360,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'init_distance_min': 10.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 20.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2, -np.pi], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
        'coverage_reward': 25.0 # adding coverage reward
    }


# really hard
METADATA_v4_inf_comms = {
        'version' : 1,
        'sensor_r': 10.0,
        'comms_r': 50.0,
        'fov' : 360,
        'sensor_r_sd': 0.2, # sensor range noise.
        'sensor_b_sd': 0.01, # sensor bearing noise.
        'target_init_cov': 30.0, # initial target diagonal Covariance.
        'target_init_vel': 0.0, # target's initial velocity.
        'target_vel_limit': 2.0, # velocity limit of targets.
        'init_distance_min': 10.0, # the minimum distance btw targets and the agent.
        'init_distance_max': 20.0, # the maximum distance btw targets and the agent.
        'init_belief_distance_min': 0.0, # the minimum distance btw belief and the target.
        'init_belief_distance_max': 5.0, # the maximum distance btw belief and the target.
        'margin': 1.0, # a marginal distance btw targets and the agent.
        'margin2wall': 0.5, # a marginal distance from a wall.
        'action_v': [2, 1.33, 0.67, 0], # action primitives - linear velocities.
        'action_w': [np.pi/2, 0, -np.pi/2, -np.pi], # action primitives - angular velocities.
        'const_q': 0.001, # target noise constant in beliefs.
        'const_q_true': 0.01, # target noise constant of actual targets.
        'coverage_reward': 25.0 # adding coverage reward
    }
METADATAS = {
    "1":METADATA_v1, 
    "2":METADATA_v2, 
    "3":METADATA_v3, 
    "4":METADATA_v4,
    "no_comms":METADATA_v4_no_comms,
    "inf_comms":METADATA_v4_inf_comms
}
##Beliefs are initialized near target
METADATA=METADATAS["4"]