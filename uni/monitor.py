from gym import wrappers
from gym.monitoring import video_recorder
import subprocess
from gym import error
import logging

logger = logging.getLogger(__name__)


class UniMonitor(wrappers.Monitor):
    def __init__(self, env, directory, video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        super(UniMonitor, self).__init__(
            env,
            directory,
            video_callable=video_callable,
            force=force, resume=resume,
            write_upon_reset=write_upon_reset,
            uid=uid,
            mode=mode
        )

    def _reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        self.video_recorder = UniStreamRecorder(
            env=self.env,
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )
        self.video_recorder.capture_frame()


class UniStreamRecorder(video_recorder.VideoRecorder):
    def __init__(self, env, path=None, metadata=None, enabled=True, base_path=None):
        super(UniStreamRecorder, self).__init__(env, path=path, metadata=metadata, enabled=enabled, base_path=base_path)

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = UniImageEncoder(self.path, frame.shape, self.frames_per_sec)
            self.metadata['encoder_version'] = self.encoder.version_info
        try:
            self.encoder.capture_frame(frame)
        except error.InvalidFrame as e:
            logger.warning('Tried to pass invalid video frame, marking as broken: %s', e)
            self.broken = True
        else:
            self.empty = False


class UniImageEncoder(video_recorder.ImageEncoder):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        super(UniImageEncoder, self).__init__(output_path, frame_shape, frames_per_sec)

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
                        '-f', 'mpegts', 'http://127.0.0.1:8000/publish/uni',
                        )

        self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)
