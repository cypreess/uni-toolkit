from abc import ABCMeta, abstractmethod


class UniAlgorithm(metaclass=ABCMeta):

    def __init__(self, observation_space, action_space):
        pass

    @abstractmethod
    def action(self, observation):
        pass