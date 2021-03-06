import argparse
import logging
import os
import sys
import tarfile
import tempfile
from urllib.parse import urljoin

import datetime
import numpy as np
import requests

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
        'UNI_MODEL_DIR': '/tmp/uni-models/',
        'CPU_NUMBER': 1,
        'EPISODES': 100,
        'MODEL_SAVE_BEST_LAST_MEAN': 100,
        'MODEL_SAVE_FREQUENCY': 20,
    }
    PARAMETERS_OVERRIDDEN = {}
    PARAMETERS_CLEANERS = {
        'EPISODES': int,
        'CPU_NUMBER': int,
        'MODEL_SAVE_BEST_LAST_MEAN': int,
        'MODEL_SAVE_FREQUENCY': int,
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

        parser.add_argument('-l', '--local', default=False, action='store_true',
                            help='enable local run mode')

        parser.add_argument('-r', '--render', default=False, action='store_true',
                            help='enable environment rendering')

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
                   parameters=dict(args.set), render=args.render, local=args.local)

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

    def __init__(self, environment=None, algorithm=None, run_mode='train', parameters=None, render=False, local=False):
        assert parameters is None or type(parameters) is dict, "parameters must be dict or None"

        if parameters is not None:
            self.PARAMETERS_OVERRIDDEN.update(parameters)

        self._logger = None
        self.run_mode = run_mode

        self._environment = None
        self.environment_path = environment or os.environ.get(self.ENVIRONMENT_VAR_NAME)

        self._algorithm = None
        self.algorithm_path = algorithm or os.environ.get(self.ALGORITHM_VAR_NAME)

        self._best_last_saved_model_score = None  # used for determining the last best saved model
        self._last_episode_number_saved = 0

        self._last_update_episode_time = None
        self._last_update_episode = None

        self.render = render
        self.local = local

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
            if self.run_mode == 'run':
                self.run_model()
            elif self.run_mode == 'train':

                try:
                    self.run_training()
                except Exception as e:
                    if not self.local:
                        self._call_uni_api(
                            path='/organizations/{organization_pk}/projects/{project_pk}/runs/{run_pk}/instances/{id}/finish/'.format(
                                organization_pk=self['UNI_ORGANIZATION_ID'],
                                project_pk=self['UNI_PROJECT_ID'],
                                run_pk=self['UNI_RUN_ID'],
                                id=self['UNI_RIN_ID']
                            ),
                            verb='post',
                            data={'failed': True}
                        )
                    raise e
                else:
                    if not self.local:
                        self._call_uni_api(
                            path='/organizations/{organization_pk}/projects/{project_pk}/runs/{run_pk}/instances/{id}/finish/'.format(
                                organization_pk=self['UNI_ORGANIZATION_ID'],
                                project_pk=self['UNI_PROJECT_ID'],
                                run_pk=self['UNI_RUN_ID'],
                                id=self['UNI_RIN_ID']
                            ),
                            verb='post',
                            data={'failed': False}
                        )

            elif self.run_mode == 'info':
                self.run_info()
            else:
                raise UniConfigurationError('Unknown run mode "%s"' % self.run_mode)

        except UniFatalError as e:
            self.logger.error(e.message)
            sys.exit(1)

        except UniConfigurationError as e:
            self.logger.error("Bad configuration! {problem}".format(problem=e.message))
            sys.exit(2)

    def run_info(self):
        """Prints custom string to be shown in visualisation window"""
        print(self.name)

    def run_training(self):
        """
        Runs algorithm training intercepting yield control whenever model evaluation should happen
        to decide if we want to save the model.
        """

        self.logger.info("Running training...")

        for episodes_rewards in self.algorithm.train():
            model_score = self.get_model_score(episodes_rewards)
            episode = len(episodes_rewards)

            if len(episodes_rewards) % 1000 == 0:
                print('@metric score %d %f' % (len(episodes_rewards), model_score))

            if self.should_save_model(model_score, episode):
                self.model_save(model_score, episode)

            self.update_episode_count(len(episodes_rewards))

        self.update_episode_count(len(episodes_rewards), force=True)
        self.model_save(model_score, episode)

        self.logger.info("Training has finished successfully")

    def run_model(self):
        """
        Runs simulation in demo mode which just reads built before model and use it
        """
        self.logger.info("Running model...")

        self.algorithm.load(directory=self.parameter('UNI_MODEL_DIR'))

        episode = 0

        while True:  # Episode loop
            episode += 1
            step = 0
            is_done = False
            observation = self.environment.reset()
            episode_reward = 0

            while not is_done:  # Step loop
                step += 1

                if self.render:
                    self.environment.render()

                action = self.algorithm.action(episode, step, observation)

                observation, reward, is_done, debug = self.environment.step(action)
                episode_reward += reward

            self.logger.info("Episode #{episode} reward {reward}".format(episode=episode, reward=episode_reward))

    def get_model_score(self, episodes_rewards):
        """Dummy scoring; last N-th mean reward"""
        return round(np.mean(episodes_rewards[-self['MODEL_SAVE_BEST_LAST_MEAN']:]), 1)

    def should_save_model(self, model_score, episode_number):
        """
        Make decision if model is good enough to be saved as intermediate outcome policy
        """
        is_not_too_frequent = episode_number - self._last_episode_number_saved > self['MODEL_SAVE_FREQUENCY']
        has_greater_score = self._best_last_saved_model_score is None or model_score > self._best_last_saved_model_score
        return is_not_too_frequent and has_greater_score

    def _call_uni_api(self, path, verb, data):
        endpoint = urljoin(self['UNI_API_URL'], path)
        headers = {'Authorization': 'token %s' % self['UNI_API_TOKEN']}
        return getattr(requests, verb)(endpoint, data=data, headers=headers)

    def model_save(self, model_score, episode_number):
        """
        Perform model save
        """

        self.logger.info('Saving model with score={reward} to {directory}'.format(
            reward=model_score, directory=self.parameter('UNI_MODEL_DIR')))

        self.algorithm.save(directory=self.parameter('UNI_MODEL_DIR'))

        self._best_last_saved_model_score = model_score
        self._last_episode_number_saved = episode_number

        if not self.local:
            response = self._call_uni_api(path='/organizations/%s/projects/%s/runs/%s/models/' % (
            self['UNI_ORGANIZATION_ID'], self['UNI_PROJECT_ID'], self['UNI_RUN_ID']), verb='post',
                                          data={'score': float(model_score)})
            if response.status_code == 201:
                model_data = response.json()
                self.logger.info("Uploading model...")
                with tempfile.TemporaryFile() as temp_archive:
                    with tarfile.open(fileobj=temp_archive, mode="w:gz") as temp_tar_archive:
                        temp_tar_archive.add(self['UNI_MODEL_DIR'], arcname='')
                    temp_archive.seek(0)
                    response = requests.put(model_data['upload_url'], data=temp_archive.read())
                if response.status_code == 200:

                    self._call_uni_api(path='/organizations/%s/projects/%s/runs/%s/models/%s/' % (
                    self['UNI_ORGANIZATION_ID'], self['UNI_PROJECT_ID'], self['UNI_RUN_ID'], model_data['id']),
                                       verb='patch', data={'uploaded': True})
                    self.logger.info("Model successfully uploaded...")
                else:
                    self.logger.warning("Error while uploading model: %s %s" % (response, response.text))
            elif response.status_code == 204:
                self.logger.info(
                    'Skipping model upload because there is already better model score than %d' % model_score)
            else:
                self.logger.error('Problem with connecting to Uni API: %s %s' % (response, response.text))

    def __getitem__(self, item):
        return self.parameter(item)

    def update_episode_count(self, episodes, force=False):
        """Updates API with the fact that episodes number has changed; Be graceful for API server"""
        assert type(episodes) is int

        # Do not send anything to API if in 'local' run mode
        if not self.local:
            # Send anything if it is the first time or wait 10s between two api calls
            if force or self._last_update_episode_time is None \
                    or datetime.datetime.now() - self._last_update_episode_time > datetime.timedelta(
                seconds=5):
                # Update only if the value changed indeed
                if self._last_update_episode != episodes:
                    self._last_update_episode = episodes
                    self._last_update_episode_time = datetime.datetime.now()
                    self._call_uni_api(path='/organizations/%s/projects/%s/runs/%s/instances/%s/' % (
                        self['UNI_ORGANIZATION_ID'], self['UNI_PROJECT_ID'], self['UNI_RUN_ID'], self['UNI_RIN_ID']),
                                       verb='patch', data={'episodes': int(episodes)})
