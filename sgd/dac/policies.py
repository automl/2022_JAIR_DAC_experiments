import numpy as np

from sgd.dac.abstract import DACPolicy


class SGDStaticPolicy(DACPolicy):

    def __init__(self, lr=0.01):
        self.lr = lr

    def act(self, obs):
        return self.lr

    def reset(self):
        pass

    def number_of_parameters(self):
        return 1

    def current_configuration(self):
        return np.array([self.lr])

    def reconfigure(self, configuration):
        self.lr = configuration[0]

    def __str__(self):
        return "SGDStaticPolicy(lr: {})".format(self.lr)


class SGDLogLinearPolicy(DACPolicy):

    def __init__(self, features, action_range=[1e-14, 1e14]):
        self.action_range = action_range
        self.features = features
        self.ll_model_params = np.array([0.0] * len(self.features) + [np.log(0.01)])

    def act(self, obs):
        feature_values = []
        for k in self.features:
            feature_values.append(obs[k])
        input = np.array(feature_values + [1])
        output = np.exp(np.dot(input, self.ll_model_params))
        output.clip(self.action_range[0], self.action_range[1])
        return output

    def reset(self):
        pass

    def number_of_parameters(self):
        return len(self.ll_model_params)

    def current_configuration(self):
        return self.ll_model_params

    def reconfigure(self, configuration):
        self.ll_model_params = np.asarray(configuration)

    def __str__(self):
        return "SGDLogLinearPolicy({})".format(self.current_configuration())


class RMSpropPolicy(SGDLogLinearPolicy):
    """
    Special case of log-linear policy (Daniel et al, 2016) used for training Momentum
    """

    def __init__(self):
        features = ["predictiveChangeVarDiscountedAverage",
                    "lossVarDiscountedAverage",
                    "predictiveChangeVarUncertainty",
                    "lossVarUncertainty"]
        super().__init__(features, action_range=[1e-14, 1e14])

    def __str__(self):
        return "RMSpropPolicy({})".format(self.current_configuration())


class MomentumPolicy(SGDLogLinearPolicy):
    """
    Special case of log-linear policy (Daniel et al, 2016) used for training Momentum
    """

    def __init__(self):
        features = ["predictiveChangeVarDiscountedAverage",
                    "lossVarDiscountedAverage",
                    "predictiveChangeVarUncertainty",
                    "lossVarUncertainty"]
        super().__init__(features, action_range=[1e-14, 1e14])

    def number_of_parameters(self):
        return 3

    def current_configuration(self):
        return self.ll_model_params[[0, 2, 4]]

    def reconfigure(self, c):
        self.ll_model_params = np.asarray([c[0], -c[0], c[1], c[1], c[2]])

    def __str__(self):
        return "MomentumPolicy({})".format(self.current_configuration())