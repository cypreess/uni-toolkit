import distutils.spawn
import distutils.version
import logging
import subprocess

import numpy as np
from gym import Wrapper
from gym import error

logger = logging.getLogger(__name__)


class UniMonitor(Wrapper):
    def __init__(self, env):
        super(UniMonitor, self).__init__(env)

        self.video_recorder = None
        self.enabled = True
        self.episode_id = 0
        self._monitor_id = None

    def _reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            return

        # Start recording the next video.
        self.video_recorder = UniStreamRecorder(
            env=self.env,
            enabled=self._video_enabled(),
        )
        self.video_recorder.capture_frame()

    def _step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)

        return observation, reward, done, info

    def _reset(self):
        self._before_reset()
        observation = self.env.reset()
        self._after_reset(observation)

        return observation

    def _close(self):
        super(UniMonitor, self)._close()

        if getattr(self, '_monitor', None):
            self.close()

    def _flush(self, force=False):
        return

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        if not self.enabled:
            return
        if self.video_recorder is not None:
            self._close_video_recorder()

        self.enabled = False

    def _before_step(self, action):
        return

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        # Record video
        self.video_recorder.capture_frame()

        return done

    def _before_reset(self):
        return

    def _after_reset(self, observation):
        if not self.enabled: return

        self._reset_video_recorder()

    def _close_video_recorder(self):
        self.video_recorder.close()

    def _video_enabled(self):
        return True

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()

class UniStreamRecorder(object):
    def __init__(self, env, enabled=True):
        self.enabled = enabled

        self.ansi_mode = False

        self.last_frame = None
        self.env = env

        self.frames_per_sec = env.metadata.get('video.frames_per_second', 30)
        self.encoder = None  # lazily start the process
        self.broken = False

        logger.info('Starting video stream.')
        self.empty = True

    @property
    def functional(self):
        return self.enabled and not self.broken

    def close(self):
        """Make sure to manually close, or else you'll leak the encoder process"""
        if not self.enabled:
            return

        if self.encoder:
            logger.debug('Closing stream encoder.')
            self.encoder.close()
            self.encoder = None

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = UniImageEncoder(frame.shape, self.frames_per_sec)
        try:
            self.encoder.capture_frame(frame)
        except error.InvalidFrame as e:
            logger.warning('Tried to pass invalid video frame, marking as broken: %s', e)
            self.broken = True
        else:
            self.empty = False

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        if not self.functional: return

        frame = self.env.render(mode='rgb_array')

        if frame is None:
            # Indicates a bug in the environment: don't want to raise
            # an error here.
            logger.warning(
                'Env returned None on render(). Disabling further rendering for stream recorder by marking as disabled')
            self.broken = True
        else:
            self.last_frame = frame
            self._encode_image_frame(frame)


class UniImageEncoder(object):
    def __init__(self, frame_shape, frames_per_sec):
        self.proc = None

        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e. RGB values for a w-by-h image, with an optional alpha channl.".format(
                    frame_shape))
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        elif distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        else:
            raise error.DependencyNotInstalled(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

        self.start()

    def start(self):
        self.cmdline = (self.backend,
                        # '-f', 'video4linux2',
                        # '-i', '/dev/video0',
                        # '-c:v', 'libx264',
                        # '-an',
                        # '-s', '640x360',
                        # '-b:v', '300K',
                        # '-probesize', '32',
                        # '-g', '30',
                        # '-tune', 'zerolatency',
                        # '-analyzeduration', '0',
                        '-nostats',
                        '-loglevel', 'error',  # suppress warnings
                        # '-y',
                        # '-r', '%d' % self.frames_per_sec,

                        # input
                        '-f', 'rawvideo',
                        '-s:v', '{}x{}'.format(*self.wh),
                        '-pix_fmt', ('rgb32' if self.includes_alpha else 'rgb24'),
                        '-re',
                        '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly

                        # output

                        # '-g', '30',
                        # '-tune', 'zerolatency',
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        # '-c', 'copy',
                        '-bsf:v', 'h264_mp4toannexb',
                        '-f', 'mpegts', 'http://127.0.0.1/publish/uni',
                        )

        self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                'Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                "Your frame has shape {}, but the UniStreamRecorder is configured for shape {}.".format(frame.shape,
                                                                                                    self.frame_shape))
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))

        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error("UniStreamRecorder encoder exited with status {}".format(ret))
