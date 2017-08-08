from abc import ABCMeta, abstractmethod


class UniEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

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
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass



class OpenAiGymUniEnvironment(UniEnvironment):
    OPEN_AI_GYM_ENV_NAME = None

    def pre_init_hook(self):
        pass

    def __init__(self):
        import gym

        self.pre_init_hook()
        self._env = gym.make(self.OPEN_AI_GYM_ENV_NAME)

    def step(self, action):
        return self._env.step(action)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self._env.reset()


