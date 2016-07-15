from os import listdir, close
from utils import exec_subproc
import argparse
from os import path as p
from os import remove
import time
import utils
import sys

fps = 60

def get_conctac_cmd(path, ep_index, frame_index, video_type):
    fps = 60
    jpg_pattern = p.join(path, 'af_jpg','img_{0:03d}_%05d.jpg'.format(int(ep_index)))

    if video_type == 0:
        episode_name = p.join(path, 'ep_{0}_{1}_{2:03d}_n.mp4'.format(path.split(p.sep)[-2], p.basename(path),
                                                                 int(ep_index)))
        cmd = ['ffmpeg', '-r', str(fps), '-y', '-start_number', '{0:05d}'.format(int(frame_index)), '-i', jpg_pattern,
               '-c:v', 'libx264', '-r', str(fps), episode_name]
    else:
        episode_name = p.join(path, 'ep_{0}_{1}_{2:03d}_s.mp4'.format(path.split(p.sep)[-2], p.basename(path),
                                                                 int(ep_index)))
        cmd = ['ffmpeg', '-r', str(int(fps/args['slow'])), '-y', '-start_number', '{0:05d}'.format(int(frame_index)), '-i', jpg_pattern,
               '-c:v', 'libx264', '-r', str(fps), episode_name]

    return cmd


def duplicate_images(image_path, episode, min_frame, max_frame, type_):
    path = p.dirname(image_path)
    #path = '/'.join(image_path.split('/')[:-1])
    if type_ == 0:
        for i in range(max(0, min_frame - fps * args['time']), min_frame):
            save_path = p.join(path, 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            cmd = ['cp', image_path, save_path]
            exec_subproc(cmd, 0)
    else:
        for i in range(max_frame + 1, max_frame + int(fps / 10 * args['time'])):
            save_path = p.join(path, 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            cmd = ['cp', image_path, save_path]
            exec_subproc(cmd, 0)


def remove_duplicates(image_path, episode, min_frame, max_frame, type_):
    path = p.dirname(image_path)
    if type_ == 0:
        for i in range(max(0, min_frame - fps * args['time']), min_frame):
            save_path = p.join(path, 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            cmd = ['rm', save_path]
            exec_subproc(cmd, 0)
    else:
        for i in range(max_frame + 1, max_frame + int(fps / 10 * args['time'])):
            save_path = p.join(path, 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            # print(save_path)
            cmd = ['rm', save_path]
            exec_subproc(cmd, 0)

def get_min_max_frames(file_list, ep_index, s_f):

    max_ep = int(s_f[-1].split('.')[0])
    min_ep = int(s_f[-1].split('.')[0])

    for f_ in file_list:
        sf2 = f_.split('_')[2]
        sf3 = sf2.split('.')[0]
        # print(sf3)
        fr_index = int(sf3)
        ep_index_ = int(f_.split('_')[1])
        # print fr_index
        # print f_
        if ep_index_ == ep_index:

            if fr_index >= max_ep:
                max_ep = fr_index
            if fr_index <= min_ep:
                min_ep = fr_index
    return {'min':min_ep, 'max':max_ep}


def get_directory(path, list_episodes):
    video_dict = {}
    file_list = sorted(listdir(p.join(path, 'af_jpg')))
    for f in file_list:
        s_f = f.split('_')
        ep_index = int(s_f[1])
        if ep_index in list_episodes and not ep_index in video_dict.keys():

            videos = []
            min_max_ep = get_min_max_frames(file_list, ep_index, s_f)
            max_ep = min_max_ep['max']
            min_ep = min_max_ep['min']
            frame_index = s_f[-1].split('.')[0]
            video_dict[ep_index] = frame_index
            duplicated_image_path = p.join(path, 'af_jpg', 'img_{0:03d}_{1:05d}.jpg'.format(ep_index, min_ep))
            duplicate_images(duplicated_image_path, ep_index, min_ep, max_ep, 0)
            cmd = get_conctac_cmd(path, ep_index, max(0, min_ep - fps * 2), 0)
            videos.append(cmd[-1])
            exec_subproc(cmd, 1)
            remove_duplicates(duplicated_image_path, ep_index, min_ep, max_ep, 0)

            duplicated_image_path = p.join(path, 'af_jpg', 'img_{0:03d}_{1:05d}.jpg'.format(ep_index, max_ep))
            duplicate_images(duplicated_image_path, ep_index, min_ep, max_ep, 1)
            cmd = get_conctac_cmd(path, ep_index, min_ep, 1)

            exec_subproc(cmd, 1)
            videos.append(cmd[-1])
            remove_duplicates(duplicated_image_path, ep_index, min_ep, max_ep, 1)
            if args['join']:
                join_couple_videos(videos)


def join_couple_videos(videos):
    video_path = p.join(p.dirname(videos[0]),
                              '_'.join(p.basename(videos[0]).split('_')[:-1] + ['j.mp4']))
    list_file = p.join(p.dirname(videos[0]), 'list.txt')
    f = open(list_file, 'w')
    for file in videos:
        str_line = "file '{0}'\n".format( p.basename(file))
        f.write(str_line)
    f.close()
    cmd = ['ffmpeg', '-f', 'concat', '-y', '-i', list_file, '-c', 'copy', video_path]

    exec_subproc(cmd, 1)
    remove(list_file)

def concatenate_videos(path):
    # print(path)
    files = sorted(listdir(path))
    videos_list_file = p.join(path, 'videos.txt')
    concat_video = p.join(path, 'videos.mp4')
    f = open(videos_list_file, 'w')
    # print(files)
    for file in files:
        if file.endswith('.mp4'):
            str_line = "file '{0}'\n".format(file)
        else:
            continue
        # print(str_line)
        f.write(str_line)
    f.close()
    cmd = ['ffmpeg', '-f', 'concat', '-y', '-i', videos_list_file, '-c', 'copy', concat_video]
    return cmd


def read_episodes_list(path):
    file_episodes_list = p.join(path, 'episodes.txt')
    episodes = []
    for line in open(file_episodes_list):
        episodes.append(int(line))
    return episodes


def get_directory_blank(path, blank_path):
    video_dict = {}
    for f in sorted(listdir(path)):
        s_f = f.split('_')
        ep_index = '_'.join(s_f[:-1])
        if not ep_index in video_dict.keys():
            # print (f)
            blamk_name = ep_index + '_t.jpg'
            video_dict[ep_index] = blamk_name
            cmd = ['cp', blank_path, path + '/' + blamk_name]
            exec_subproc(cmd, 1)


def extend_episode(episode_name, frames_before, frames_after):
    path = p.dirname(episode_name)
    file_name = p.basename(episode_name)
    ep_index = file_name.split('_')[1]
    s_f = file_name.split('_')
    #'ep_{0}_{1}_{2:03d}_n.mp4'
    file_list = sorted(listdir(p.join(path, 'af_jpg')))
    min_max = get_min_max_frames(file_list, ep_index, s_f)
    for frame in range( max(0, min_max['min'] - frames_before), min_max['min']):
        img_from = p.join(path, 'jpg', 'image{0:06d}.jpg'.format(frame))
        img_to= p.join(path, 'af_jpg', 'img_{0:03d}_{1:05d}.jpg'.format(ep_index, frame))
        cmd = ['cp', img_from, img_to]
    for frame in range(min_max['max'], min_max['man'] - frames_after):
        img_from = p.join(path, 'jpg', 'image{0:06d}.jpg'.format(frame))
        img_to = p.join(path, 'af_jpg', 'img_{0:03d}_{1:05d}.jpg'.format(ep_index, frame))
        cmd = ['cp', img_from, img_to]

sys.stdout = utils.Unbuffered(sys.stdout)

# print 'Hello'
#
# print "print"
# print ("test"),
# # sys.stdout.flush()
# time.sleep(2)
# print 'OK'
# exit(0)
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="Path to the directory with preprocessed imgaes, videos, video lists")
ap.add_argument("-r", "--rebuild", action='store_true',
                help="Rebulid all video episodes regarding episodes list")
ap.add_argument("-j", "--join", action='store_true',
                help="Join video couples to single video")
ap.add_argument("-t", "--time", type=int, default=2,
                help="Freeze time before and after episode (Default 2 sec)")
ap.add_argument("-c", "--concatenate", action='store_true',
                help="Concatenate all videos in directory into single file")
ap.add_argument("-s", "--slow", type=float, default=10,
                help="Slow coefficient")
ap.add_argument("-f", "--fps", type=int, default=60, help="fps of input video")
args = vars(ap.parse_args())
# timer = utils.TimeCounter()
fps = args['fps']


if args['rebuild']:
    get_directory(args['path'], read_episodes_list(args['path']))
elif args['concatenate']:
    exec_subproc(concatenate_videos(args['path']))
else:
    ap.print_help()
