import argparse
import logging
import os
import sys

from uni.exceptions import UniConfigurationError, UniFatalError
from uni.helpers import import_path


class UniRunner:
    # To adjust verbosity of logging please override those class attributes
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(filename)s:%(lineno)d - %(levelname)s - %(message)s'

    # you can set any explicit runner name, by default class name will be used
    RUNNER_NAME = None

    ENVIRONMENT_VAR_NAME = 'UNI_ENVIRONMENT_PYTHON_PATH'
    ALGORITHM_VAR_NAME = 'UNI_ALGORITHM_PYTHON_PATH'

    # this is used only for parameters set using `--set` argument on command line
    # it will override any given environment variable
    PARAMETERS = {
        'UNI_OUTPUT_DIR': '/tmp/uni-models/',
    }
    PARAMETERS_OVERRIDDEN = {}
    PARAMETERS_CLEANERS = {
        'EPISODES': int,
        'CPU_NUMBER': int,
    }

    # Prepare runner object from shell arguments
    @classmethod
    def get_parser(cls):
        """Creates standard python argparse parser, use `--help` when calling runner to see options"""

        parser = argparse.ArgumentParser(description='%s' % cls.__name__)

        parser.add_argument('-m', '--mode', default='train', choices=['train', 'run'],
                            help='running mode; default=train')

        parser.add_argument('-e', '--environment', metavar=cls.ENVIRONMENT_VAR_NAME,
                            default=os.environ.get(cls.ENVIRONMENT_VAR_NAME),
                            help='environment python path; default=%s' % cls.ENVIRONMENT_VAR_NAME)

        parser.add_argument('-a', '--algorithm', metavar=cls.ALGORITHM_VAR_NAME,
                            default=os.environ.get(cls.ALGORITHM_VAR_NAME),
                            help='algorithm python path; default=%s' % cls.ALGORITHM_VAR_NAME)

        parser.add_argument('-s', '--set', metavar=('PARAMETER', 'VALUE'), nargs=2, default=[],
                            action='append', required=False,
                            help='set specific parameter name')

        return parser

    @classmethod
    def create_runner_from_args(cls):
        """
        This is a helper for handling use case of running UniRunner from console rather than just as python class.

        It handles creating argument parser and passing some variables also from command arguments (instead of only
        shell variables).

        Command arguments have precedence over shell variables.
        """
        parser = cls.get_parser()
        args = parser.parse_args()

        return cls(environment=args.environment, algorithm=args.algorithm, run_mode=args.mode,
                   parameters=dict(args.set))

    # Set up propper logging rules

    def create_logger(self):
        """
        Our logger outputs WARNINGS and ERRORS into stderr stream and everything below into stdout stream.
        This helps keep our logs being nicely formatted in web live logs view.
        """

        class LessThanFilter(logging.Filter):
            def __init__(self, exclusive_maximum, name=""):
                super(LessThanFilter, self).__init__(name)
                self.max_level = exclusive_maximum

            def filter(self, record):
                # non-zero return means we log this message
                return 1 if record.levelno < self.max_level else 0

        formatter = logging.Formatter(self.LOG_FORMAT)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.NOTSET)  # Make root logger pass everything

        # Everything below warning goes to stdout
        logging_handler_out = logging.StreamHandler(sys.stdout)
        logging_handler_out.setLevel(self.LOG_LEVEL)
        logging_handler_out.setFormatter(formatter)
        logging_handler_out.addFilter(LessThanFilter(logging.WARNING))
        root_logger.addHandler(logging_handler_out)

        # Warning and above goes to stderr
        logging_handler_err = logging.StreamHandler(sys.stderr)
        logging_handler_err.setLevel(logging.WARNING)
        logging_handler_err.setFormatter(formatter)
        root_logger.addHandler(logging_handler_err)

        # Return specific logger for this runner class
        logger = logging.getLogger(self.__class__.__name__)
        return logger

    @property
    def logger(self):
        """Get the log handler"""
        if self._logger is None:
            self._logger = self.create_logger()
        return self._logger

    # Runner object

    def __init__(self, environment=None, algorithm=None, run_mode='train', parameters=None):
        assert parameters is None or type(parameters) is dict, "parameters must be dict or None"

        if parameters is not None:
            self.PARAMETERS_OVERRIDDEN.update(parameters)

        self._logger = None
        self.run_mode = run_mode

        self._environment = None
        self.environment_path = environment or os.environ.get(self.ENVIRONMENT_VAR_NAME)

        self._algorithm = None
        self.algorithm_path = algorithm or os.environ.get(self.ALGORITHM_VAR_NAME)

        self._model_was_saved = False
        self._best_saved_episode_reward = None  # used for determining the last best saved model

    def parameter(self, name):
        """
        Returns run parameter. Run parameter is searched within different sources with following order:
        - parameters overridden by run argument (--set)
        - parameters from environment
        - default parameters from run
        - default parameters from algorithm
        - default parameters from environment

        After founding parameter name we appy a cleaner (convert str format to any desired type). Cleaner is just
        callable(value) that will return cleaned value or exception. Order of looking for cleaners:
        - from run
        - from algorithm
        - from environment
        """
        value = None

        # Scan sources of parameters in specific order, break at first one found
        for parameter_source in (self.PARAMETERS_OVERRIDDEN, os.environ, self.PARAMETERS, self.algorithm.PARAMETERS,
                                 self.environment.PARAMETERS):
            if name in parameter_source:
                value = parameter_source[name]
                break
        else:
            # Getting to inside of this for-else means that we finished for loop and did not break it before,
            # therefore `name` is not found in any of parameter sources
            # Please note that we are NOT checking if value is None; None is acceptable parameter value
            raise UniConfigurationError('Parameter {name} is missing. Please define it.'.format(name=name))

        for cleaner_source in (self.PARAMETERS_CLEANERS, self.algorithm.PARAMETERS_CLEANERS,
                               self.environment.PARAMETERS_CLEANERS):
            if name in cleaner_source:
                try:
                    value = cleaner_source[name](value)
                except Exception as e:
                    raise UniConfigurationError(
                        'Parameter {name} value `{value}` is in wrong format ({reason}).'.format(
                            name=name, value=value, reason=e))
                break

        return value

    @property
    def environment(self):
        """
        Gets environment object from cache or create new one.

        Environment is dynamically imported from a python path found in specific parameter name
        defined in self.ENVIRONMENT_VAR_NAME
        """

        if self._environment is None:
            if self.environment_path is None:
                raise UniConfigurationError('Please set {var_name} system variable or pass environment argument'.format(
                    var_name=self.ENVIRONMENT_VAR_NAME))
            else:
                self.logger.info('Loading environment {path}'.format(path=self.environment_path))
                klass = import_path(self.environment_path)
                self._environment = klass(runner=self)

        return self._environment

    @property
    def algorithm(self):
        """
        Gets algorithm object from cache or create new one

        Algorithm is dynamically imported from a python path found in specific parameter name
        defined in self.ALGORITHM_VAR_NAME
        """

        if self._algorithm is None:
            if self.algorithm_path is None:
                raise UniConfigurationError('Please set {var_name} system variable or pass algorithm argument'.format(
                    var_name=self.ALGORITHM_VAR_NAME))
            else:
                self.logger.info('Loading algorithm {path}'.format(path=self.algorithm_path))
                klass = import_path(self.algorithm_path)
                self._algorithm = klass(runner=self, observation_space=self.environment.observation_space,
                                        action_space=self.environment.action_space)
        return self._algorithm

    @property
    def name(self):
        """
        Returns name of runner that is mostly used for logging purposes
        """
        return self.RUNNER_NAME or self.__class__.__name__

    def run(self):
        """
        Runs whole machinery with regards to the mode that runner was created in.
        """
        try:
            self.logger.info('Starting {name} in {mode} mode...'.format(name=self.name, mode=self.run_mode))
            if self.run_mode == 'run':
                self.run_model()
            elif self.run_mode == 'train':
                self.run_training()
            else:
                raise UniConfigurationError('Unknown run mode "%s"' % self.run_mode)

        except UniFatalError as e:
            self.logger.error(e.message)
            sys.exit(1)

        except UniConfigurationError as e:
            self.logger.error("Bad configuration! {problem}".format(problem=e.message))
            sys.exit(2)

    def run_training(self, episodes=None, max_steps=None):
        """
        Runs simulation in training mode which learn better policy
        """
        if episodes is None:
            episodes = self.parameter('EPISODES')

        if max_steps is None:
            max_steps = self.parameter('MAX_STEPS')

        # self.environment #.prepare()
        self.algorithm.prepare()

        episodes_rewards = []

        for episode in range(1, int(episodes) + 1):
            self.logger.info('Running episode #%d' % episode)
            episodes_rewards.append(0.0)
            observation = self.environment.reset()
            self.algorithm.pre_episode(episode)

            for step in range(1, max_steps + 1):
                action = self.algorithm.action_train(episode, step, observation)
                new_observation, reward, is_done, debug = self.environment.step(action)
                episodes_rewards[-1] += reward
                self.algorithm.post_step(episode, step, action, observation, new_observation, reward, is_done, debug)
                observation = new_observation

                if is_done:
                    self.logger.info('Episode #%d is done' % episode)
                    break

            self.algorithm.post_episode(episode)

            if self.should_save_model(episodes_rewards):
                self.model_save(episodes_rewards[-1])

    def run_model(self):
        """
        Runs simulation in demo mode which just reads built before model and use it
        """

        self.algorithm.load(directory=self.parameter('UNI_OUTPUT_DIR'))

        episode = 0

        while True:
            episode += 1
            step = 0
            is_done = False
            observation = self.environment.reset()
            episode_reward = 0

            while not is_done:
                step += 1
                self.environment.render()

                action = self.algorithm.action(episode, step, observation)

                observation, reward, is_done, debug = self.environment.step(action)
                episode_reward += reward

            self.logger.info("Episode #{episode} reward {reward}".format(episode=episode, reward=episode_reward))

    def should_save_model(self, episodes_rewards):
        """
        Make decision if model is good enough to be saved as intermediate outcome policy

        UniRunner implements dummy policy that every model with higher total reward is saved
        """

        return self._best_saved_episode_reward is None or episodes_rewards[-1] > self._best_saved_episode_reward

    def model_save(self, episode_reward):
        """
        Perform model save
        """

        self.logger.info('Saving model with episode total reward {reward} to {directory}'.format(
            reward=episode_reward, directory=self.parameter('UNI_OUTPUT_DIR')))
        self._best_saved_episode_reward = episode_reward
        self._model_was_saved = True
        self.algorithm.save(directory=self.parameter('UNI_OUTPUT_DIR'))

        # def __getitem__(self, item):
        #     return self.parameter(item)
