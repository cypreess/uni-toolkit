import logging
import os
from abc import ABCMeta, abstractmethod

from uni.helpers import import_path


class UniRunner(metaclass=ABCMeta):
    ENVIRONMENT_VAR_NAME = 'UNI_ENVIRONMENT_PYTHON_PATH'
    ALGORITHM_VAR_NAME = 'UNI_ALGORITHM_PYTHON_PATH'

    def __init__(self, environment=None, algorithm=None):
        self._logger = logging.getLogger(self.__class__.__name__)

        # Getting environment path from user or system variable
        if environment is None:
            environment = os.environ.get(self.ENVIRONMENT_VAR_NAME)
        assert environment is not None, 'Please set UNI_ENVIRONMENT system variable or pass environment argument'
        self.environment_path = environment

        # Getting algorithm path from user or system variable
        if algorithm is None:
            algorithm = os.environ.get(self.ALGORITHM_VAR_NAME)
        assert algorithm is not None, 'Please set UNI_ALGORITHM system variable or pass algorithm parameter'
        self.algorithm_path = algorithm

        self.environment = self.get_environment_class(self.environment_path)()
        self.algorithm = self.get_algorithm_class(self.algorithm_path)(
            observation_space=self.environment.observation_space,
            action_space=self.environment.action_space
        )

    @property
    def logger(self):
        return self._logger

    def run_training(self, episodes=os.environ.get('EPISODES')):
        """
        Running simulation in training mode which learn better policy
        """

        assert episodes is not None, 'Please set EPISODES system variable or pass episodes argument'

        for episode in range(int(episodes)):
            self.logger.info('Running episode #%d' % episode)

    def run_model(self):
        while 1:
            step = 0
            score = 0
            observation = self.environment.reset()

            while 1:
                action = self.algorithm.action(observation)
                observation, reward, done, info = self.environment.step(action)
                score += reward
                step += 1
                self.environment._env.render("human")
                if done:
                    self.logger.info("score=%0.2f in %i frames" % (score, step))
                    break

    def get_environment_class(self, environment_path):
        self.logger.info('Loading environment %s' % environment_path)
        return import_path(environment_path)

    def get_algorithm_class(self, algorithm_path):
        self.logger.info('Loading algorithm %s' % algorithm_path)
        return import_path(algorithm_path)
