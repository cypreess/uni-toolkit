from abc import ABCMeta, abstractmethod

from uni import monitor
from uni.helpers import ParameterReaderMixin


class UniEnvironment(ParameterReaderMixin, metaclass=ABCMeta):
    PARAMETERS = {}
    PARAMETERS_CLEANERS = {}

    def __init__(self, runner):
        self.PARAMETERS = self.read_parameters()
        self.runner = runner

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        """reset method must return observation"""
        pass

    @property
    @abstractmethod
    def action_space(self):
        """Return integer with action space size"""
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @abstractmethod
    def render(self, *args, **kwargs):
        pass


class OpenAiGymUniEnvironment(UniEnvironment):
    OPEN_AI_GYM_ENV_NAME = None

    def pre_init_hook(self):
        pass

    def __init__(self, runner):
        self._env = None
        super().__init__(runner)

    @property
    def env(self):
        if self._env is None:
            import gym

            gym.undo_logger_setup()  # Get rid of gym logging

            # logger = logging.getLogger()
            # logger.addHandler(logging.StreamHandler(sys.stdout))

            self.pre_init_hook()
            self._env = gym.make(self.OPEN_AI_GYM_ENV_NAME)
            if self.runner.run_mode == 'run':
                # We only run rendering to video in "run" mode (not training mode)
                self._env = monitor.UniMonitor(self._env)

        return self._env

    def step(self, action):
        return self.env.step(action)

    @property
    def action_space(self):
        return self.env.action_space.n

    @property
    def observation_space(self):
        return self.env.observation_space.shape

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
