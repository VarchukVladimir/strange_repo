
from os import listdir
import subprocess

from utils import *


paths = {
    "tl": {
        "path": "/media/sf_share_linux/video/2016-05-24",
        "stnum": '18',
        "maxnum":'4956'
    },
    "tr": {
        "path": "/media/sf_share_linux/video/2016-05-25",
        "stnum": '124',
        "maxnum": '5371'
    },
    "bl": {
        "path": "/media/sf_share_linux/video/2016-05-26",
        "stnum": '6',
        "maxnum": '5783'
    },
    "br": {
        "path": "/media/sf_share_linux/video/2016-05-27",
        "stnum": '66',
        "maxnum": '4644'
    },
}



max_image_num = 4644

save_path = "/media/sf_share_linux/video/joined"

frames_number = 4578

for image_index in range(0, frames_number + 1):
    images = []
    for directory in paths:
        images.append('{0}/jpg/SJCM{1:04d}.jpg'.format(paths[directory]['path'], int(paths[directory]['stnum']) + int(image_index)))
    out_image_name = save_path + '/out{0:04d}.jpg'.format(image_index)
    cmd = ['montage'] + images + ['-tile', '2x2', '-geometry', '+0+0', out_image_name]
    exec_subproc(cmd, 1)
