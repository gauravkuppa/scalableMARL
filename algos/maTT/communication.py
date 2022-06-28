class CommunicationModule:
    def __init__(self, env, range, obs_dict):
        self.env = env
        self.range = range
        self.obs_dict = obs_dict
    
    def communicate(self, agent_id):
        nearby_agents = []
        observations = []
        for agent in nearby_agents:
            observations.append(self.obs_dict[agent])
        return observations
