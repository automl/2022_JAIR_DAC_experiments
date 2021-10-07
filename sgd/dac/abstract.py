class DACEnv:

    def step(self, action):
        pass

    def render(self, mode='human', close=False):
        pass

    @property
    def conditions(self):
        pass

    def conditioned_reset(self, selected_condition):
        pass

    def policy_space(self):
        pass


class DACPolicy:

    def act(self, obs):
        raise NotImplementedError

    def reset(self):
        pass

    def number_of_parameters(self):
        return 0

    def current_configuration(self):
        return []

    def reconfigure(self, configuration):
        pass