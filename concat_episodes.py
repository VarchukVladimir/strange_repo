from os import listdir
from utils import exec_subproc
import argparse
from os import path as p

fps = 60

def get_conctac_cmd(path, ep_index, frame_index, video_type):
    fps = 60
    jpg_pattern = p.join(path, 'af_jpg','img_{0:03d}_%05d.jpg'.format(int(ep_index)))

    if video_type == 0:
        episode_name = p.join(path, 'ep_{0}_{1}_{2:03d}_n.mp4'.format(path.split(p.sep)[-2], p.basename(),
                                                                 int(ep_index)))
        cmd = ['ffmpeg', '-r', '60', '-y', '-start_number', '{0:05d}'.format(int(frame_index)), '-i', jpg_pattern,
               '-c:v', 'libx264', '-r', '60', episode_name]
    else:
        episode_name = p.join(path, 'ep_{0}_{1}_{2:03d}_s.mp4'.format(path.split(p.sep)[-2], p.basename(),
                                                                 int(ep_index)))
        cmd = ['ffmpeg', '-r', '6', '-y', '-start_number', '{0:05d}'.format(int(frame_index)), '-i', jpg_pattern,
               '-c:v', 'libx264', '-r', '60', episode_name]
    return cmd


def duplicate_images(image_path, episode, min_frame, max_frame, type_):
    path = p.dirname(image_path)
    #path = '/'.join(image_path.split('/')[:-1])
    if type_ == 0:
        for i in range(max(0, min_frame - fps * args['time']), min_frame):
            save_path = p.join(path, 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            cmd = ['cp', image_path, save_path]
            exec_subproc(cmd, 1)
    else:
        for i in range(max_frame + 1, max_frame + int(fps / 10 * args['time'])):
            save_path = p.join(path, 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            cmd = ['cp', image_path, save_path]
            exec_subproc(cmd, 1)


def remove_duplicates(image_path, episode, min_frame, max_frame, type_):
    path = p.dirname(image_path)
    if type_ == 0:
        for i in range(max(0, min_frame - fps * args['time']), min_frame):
            save_path = p.join(path, 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            cmd = ['rm', save_path]
            exec_subproc(cmd, 1)
    else:
        for i in range(max_frame + 1, max_frame + int(fps / 10 * args['time'])):
            save_path = p.join(path + 'img_{0:03d}_{1:05d}.jpg'.format(episode, int(i)))
            cmd = ['rm', save_path]
            exec_subproc(cmd, 1)


def get_directory(path, list_episodes):
    video_dict = {}
    file_list = sorted(listdir(p.join(path, 'af_jpg')))
    for f in file_list:
        s_f = f.split('_')
        ep_index = int(s_f[1])
        if ep_index in list_episodes and not ep_index in video_dict.keys():
            max_ep = int(s_f[-1].split('.')[0])
            min_ep = int(s_f[-1].split('.')[0])
            videos = []
            for f_ in file_list:
                sf2 = f_.split('_')[2]
                sf3 = sf2.split('.')[0]
                fr_index = int(sf3)
                ep_index_ = int(f_.split('_')[1])
                if ep_index_ == ep_index:

                    if fr_index >= max_ep:
                        max_ep = fr_index
                    if fr_index <= min_ep:
                        min_ep = fr_index
            frame_index = s_f[-1].split('.')[0]
            video_dict[ep_index] = frame_index
            duplicated_image_path = p.join(path, 'af_jpg', 'img_{0:03d}_{1:05d}.jpg'.format(ep_index, min_ep))
            duplicate_images(duplicated_image_path, ep_index, min_ep, max_ep, 0)
            cmd = get_conctac_cmd(path, ep_index, max(0, min_ep - fps * 2), 0)
            videos.append(cmd[-1])
            print (cmd)
            exec_subproc(cmd, 1)
            remove_duplicates(duplicated_image_path, ep_index, min_ep, max_ep, 0)

            duplicated_image_path = p.join(path, 'af_jpg', 'img_{0:03d}_{1:05d}.jpg'.format(ep_index, max_ep))
            duplicate_images(duplicated_image_path, ep_index, min_ep, max_ep, 1)
            cmd = get_conctac_cmd(path, ep_index, min_ep, 1)
            print (cmd)
            exec_subproc(cmd, 1)
            videos.append(cmd[-1])
            remove_duplicates(duplicated_image_path, ep_index, min_ep, max_ep, 1)
            if args['join']:
                join_couple_videos(videos)


def join_couple_videos(videos):
    video_path = p.join(p.dirname(videos[0]),
                              '_'.join(p.basename(videos[0]).split('_')[:-1] + ['j.mp4']))
    cmd = ['ffmpeg', '-f', 'concat', '-i'] + videos + ['-c', 'copy', video_path]
    exec_subproc(cmd, 1)


def concatenate_videos(path):
    print(path)
    files = sorted(listdir(path))
    videos_list_file = p.join(path, 'videos.txt')
    concat_video = p.join(path, 'videos.mp4')
    f = open(videos_list_file, 'w')
    print(files)
    for file in files:
        str_line = "file '{0}'\n".format(file)
        print(str_line)
        f.write(str_line)
    f.close()
    cmd = ['ffmpeg', '-f', 'concat', '-i', videos_list_file, '-c', 'copy', concat_video]
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
            print (f)
            blamk_name = ep_index + '_t.jpg'
            video_dict[ep_index] = blamk_name
            cmd = ['cp', blank_path, path + '/' + blamk_name]
            exec_subproc(cmd, 1)


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

args = vars(ap.parse_args())


if args['rebuild']:
    get_directory(args['path'], read_episodes_list(args['path']))
elif args['concatenate']:
    concatenate_videos(args['path'])
else:
    ap.print_help()
