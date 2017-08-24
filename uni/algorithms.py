import os
from abc import ABCMeta, abstractmethod

from uni.helpers import ParameterReaderMixin


class UniAlgorithm(ParameterReaderMixin, metaclass=ABCMeta):
    NAME = None
    PARAMETERS = {}
    PARAMETERS_CLEANERS = {}

    def __init__(self, runner, observation_space, action_space):
        self.PARAMETERS = self.read_parameters()
        self.runner = runner
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    def name(self):
        return self.__class__.__name__ if self.NAME is None else self.NAME

    def prepare(self):
        """Set up some additional run-time properties for model"""

        # Ensure that model output directory exists
        if not os.path.exists(self.runner.parameter('UNI_OUTPUT_DIR')):
            self.runner.logger.warning('Creating directory {dir}'.format(dir=self.runner.parameter('UNI_OUTPUT_DIR')))
            os.makedirs(self.runner.parameter('UNI_OUTPUT_DIR'))

    def pre_episode(self, episode):
        """Method runs on the beginning of each episode just after environment had been reset"""
        pass

    @abstractmethod
    def action(self, episode, step, observation):
        """Perform action in environment based on observation in given episode step"""
        pass

    def action_train(self, episode, step, observation):
        """Specific version of action method that is used for training, by default is same as main action method"""
        return self.action(episode, step, observation)

    def post_step(self, episode, step, action, observation, new_observation, reward, is_done, debug):
        """Run after environment step"""
        pass

    def post_episode(self, episode):
        """Method runs on the end of each episode after all steps had been performed"""
        pass

    @abstractmethod
    def save(self, directory):
        """Should dump the model to any number of files to provided directory"""
        pass

    @abstractmethod
    def load(self, directory):
        """Should load the model from any number of files to provided directory"""
        pass
