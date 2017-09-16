import os
from abc import ABCMeta, abstractmethod

from uni.helpers import ParameterReaderMixin


class UniAlgorithm(ParameterReaderMixin, metaclass=ABCMeta):
    NAME = None
    PARAMETERS = {}
    PARAMETERS_CLEANERS = {}

    def __init__(self, runner, observation_space, action_space):
        self.PARAMETERS.update(self.read_parameters())
        self.runner = runner
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    def logger(self):
        return self.runner.logger

    @property
    def name(self):
        return self.__class__.__name__ if self.NAME is None else self.NAME

    def train(self):
        """
        train should be python generator. This generator steers the process of learning algorithm,
        yielding control to the runner whenever model should be evaluated for saving.

        Runner is making decision about conditions under which model will be saved.
        """
        episodes = self.runner['EPISODES']

        # self.environment #.prepare()
        self.prepare()

        episodes_rewards = []

        for episode in range(1, int(episodes) + 1):
            self.logger.info('Running episode #%d' % episode)
            episodes_rewards.append(0.0)
            observation = self.runner.environment.reset()

            self.pre_episode(episode)
            for step in range(1, self.runner['MAX_STEPS'] + 1):
                action = self.action_train(episode, step, observation)
                new_observation, reward, is_done, debug = self.runner.environment.step(action)
                episodes_rewards[-1] += reward
                self.post_step(episode, step, action, observation, new_observation, reward, is_done,
                                         debug)
                observation = new_observation

                if is_done:
                    self.logger.info('Episode #%d is done' % episode)
                    break

            self.post_episode(episode)

            yield episodes_rewards

        self.logger.info("Training has finished successfully")

    def prepare(self):
        """Set up some additional run-time properties for model"""

        # Ensure that model output directory exists
        if not os.path.exists(self.runner.parameter('UNI_MODEL_DIR')):
            self.runner.logger.warning('Creating directory {dir}'.format(dir=self.runner.parameter('UNI_MODEL_DIR')))
            os.makedirs(self.runner.parameter('UNI_MODEL_DIR'))

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

