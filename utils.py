
import subprocess
import os
import cv2
from os import path as p
from datetime import datetime
import tempfile

def def_colored(text, color):
    return text

try:
    from termcolor1 import colored
    safe_colored = colored
except ImportError:
    safe_colored = def_colored


class TimeCounter:
    def __init__(self):
        self.start_time = datetime.now()
    def start_count(self):
        self.start_time = datetime.now()
    def end_count(self, message=''):
        self.end_time = datetime.now()
        print ('{0} {1}'.format(message, str(self.end_time - self.start_time)))
    def end_time_str(self, message=''):
        self.end_time = datetime.now()
        return '{0} {1}'.format(message, str(self.end_time - self.start_time))

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

def exec_subproc(cmd, show_info=1):
    timer = TimeCounter()
    cmd_str = ' '.join(cmd)
    if show_info:
        print safe_colored('[command]', 'blue'), '[{0}]'.format(cmd_str),
    temp_out = tempfile.TemporaryFile()
    temp_err = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, stdout=temp_out, stderr=temp_err)
    p.wait()
    temp_out.seek(0)
    temp_err.seek(0)
    out = temp_out.read()
    err = temp_err.read()
    if p.returncode == 0:
        if show_info:
            print safe_colored(timer.end_time_str(' [OK]'), 'green')
    else:
        print safe_colored(timer.end_time_str(' [ERROR]'), 'red')
        print out
        print err

    return p.returncode


def batch_execute_proc(batch_name):
    f = open(batch_name, 'r')

    for line in f.read().split():
        cmd = line.split(' ')
        exec_subproc(cmd, 1)
    f.close()

def get_working_path(path):
    return '/'.join(path.split('/')[:-1] + [get_short_name(path)])


def get_jpg_path(path):
    video_name = get_video_name(path).split('.')[0]
    tmp_path = os.path.join(os.path.expanduser("~"), 'image_poroc')
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    tmp_path = os.path.join(tmp_path,video_name)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    return os.path.join(tmp_path, 'jpg')


def get_af_path_path(path):
    return '/'.join([get_working_path(path)] + ['af_jpg'])


def get_video_path(path):
    return '/'.join([get_working_path(path)] + ['video'])


def get_short_name(path):
    return path.split('/')[-1].split('.')[0]


def get_video_name(path):
    return path.split('/')[-1]

def get_frame_rate(video_path):
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def get_video_path_from_episode(episode_path):
    return p.dirname(episode_path) + '.mp4'
