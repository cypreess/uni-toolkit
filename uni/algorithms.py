from abc import ABCMeta, abstractmethod


class UniAlgorithm(metaclass=ABCMeta):
    NAME = None
    PARAMETERS = {}
    PARAMETERS_CLEANERS = {}

    def __init__(self, runner, observation_space, action_space):
        self.runner = runner
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    def name(self):
        return self.__class__.__name__ if self.NAME is None else self.NAME

    @abstractmethod
    def action(self, episode, step, observation):
        pass

    def post_step(self, episode, step, action, observation, new_observation, reward, is_done, debug):
        """Run after environment step"""
        pass

    def prepare(self):
        pass

    def pre_episode(self, episode):
        """Method runs on the beginning of each episode just after environment had been reset"""
        pass

    def post_episode(self, episode):
        """Method runs on the end of each episode after all steps had been performed"""
        pass

    @abstractmethod
    def save(self, directory):
        """Should dump the model to any number of files to provided directory"""
        pass
