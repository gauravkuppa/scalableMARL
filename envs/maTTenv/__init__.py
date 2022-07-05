from envs.utilities.ma_time_limit import maTimeLimit

def make(env_name, render=False, figID=0, record=False, directory='',
                    T_steps=None, num_agents=2, num_targets=1, **kwargs):
    """
    env_name : str
        name of an environment. (e.g. 'Cartpole-v0')
    type : str
        type of an environment. One of ['atari', 'classic_control',
        'classic_mdp','target_tracking']
    """
    if T_steps is None:
        T_steps = 200

    if env_name == 'setTracking-v0':
        from envs.maTTenv.env.setTracking_v0 import setTrackingEnv0
        env0 = setTrackingEnv0(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-v1':
        from envs.maTTenv.env.setTracking_v1 import setTrackingEnv1
        env0 = setTrackingEnv1(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-v2':
        from envs.maTTenv.env.setTracking_v2 import setTrackingEnv2
        env0 = setTrackingEnv2(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vGreedy':
        from envs.maTTenv.env.setTracking_vGreedy import setTrackingEnvGreedy
        env0 = setTrackingEnvGreedy(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vkGreedy':
        from envs.maTTenv.env.setTracking_vkGreedy import setTrackingEnvkGreedy
        env0 = setTrackingEnvkGreedy(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vGru':
        from envs.maTTenv.env.setTracking_vGru import setTrackingEnvGru
        env0 = setTrackingEnvGru(num_agents=num_agents, num_targets=num_targets, **kwargs)
    else:
        raise ValueError('No such environment exists.')

    env = maTimeLimit(env0, max_episode_steps=T_steps)
    #env = env0
    if render:
        from envs.maTTenv.display_wrapper import Display2D
        env = Display2D(env, figID=figID)
    if record:
        from envs.maTTenv.display_wrapper import Video2D
        env = Video2D(env, dirname = directory)
    
    return env