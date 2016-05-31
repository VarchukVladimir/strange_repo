__author__ = 'volodymyr'


import numpy
import os
import copy

from os import listdir

def write_list():
    path = '/home/volodymyr/git/strange_repo/images'
    files = sorted(listdir(path))
    videos_list_file = path + '/videos2.txt'
    concat_video = path+'/videos.mp4'
    f = open(videos_list_file, 'w')
    print(files)
    for file in files:
        str_line = "file '{0}'\n".format( file)
        print(str_line)
        f.write(str_line)
    f.close()
    cmd = ['ffmpeg', '-f', 'concat', '-i', videos_list_file, '-c', 'copy', concat_video]
    return cmd

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
# img_open = Image.open("white.jpg")
# font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerif.ttf", 100)
# width, height = img_open.size
#
# for i in range(720):
#     img = copy.copy(img_open)
#     draw = ImageDraw.Draw(img)
#     draw.text((int((width-100)/2), int(height/2)),str(i),(0,0,0),font=font)
#     img.save('/home/volodymyr/git/strange_repo/images/img{0:04d}.jpg'.format(i))


print(' '.join(write_list()))