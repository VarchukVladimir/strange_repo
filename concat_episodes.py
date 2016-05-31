from os import listdir
import sys
from utils import exec_subproc


fps = 60


def get_conctac_cmd(path, ep_index, frame_index, video_type):
    fps = 60
    intro_time = 2
    affter_time = 12
    jpg_pattern = path + '/af_jpg/img_{0:03d}_%05d.jpg'.format(int(ep_index))
    # episode_name = '/'.join([path, 'ep_'+str(ep_index)+'.mp4'])

    if video_type == 0:
        #        episode_name = '/home/volodymyr/v/2016-05-11/ep_s_{0:03d}.mp4'.format(int(ep_index))
        episode_name = path + '/ep_{0}_{1}_{2:03d}_n.mp4'.format(path.split('/')[-2], path.split('/')[-1],
                                                                 int(ep_index))
        cmd = ['ffmpeg', '-r', '60', '-y', '-start_number', '{0:05d}'.format(int(frame_index)), '-i', jpg_pattern,
               '-c:v', 'libx264', '-r', '60', episode_name]
    else:
        episode_name = path + '/ep_{0}_{1}_{2:03d}_s.mp4'.format(path.split('/')[-2], path.split('/')[-1],
                                                                 int(ep_index))
        cmd = ['ffmpeg', '-r', '6', '-y', '-start_number', '{0:05d}'.format(int(frame_index)), '-i', jpg_pattern,
               '-c:v', 'libx264', '-r', '60', episode_name]
    return cmd


def duplicate_images(image_path, episode, min_frame, max_frame, type_):
    path = '/'.join(image_path.split('/')[:-1])
    if type_ == 0:
        for i in range(max(0, min_frame - fps * 2), min_frame):
            save_path = path + '/img_{0:03d}_{1:05d}.jpg'.format(episode, int(i))
            cmd = ['cp', image_path, save_path]
            exec_subproc(cmd, 1)
    else:
        for i in range(max_frame + 1, max_frame + int(fps/10 * 2)):
            save_path = path + '/img_{0:03d}_{1:05d}.jpg'.format(episode, int(i))
            cmd = ['cp', image_path, save_path]
            exec_subproc(cmd, 1)


def remove_duplicates(image_path, episode, min_frame, max_frame, type_):
    path = '/'.join(image_path.split('/')[:-1])
    if type_ == 0:
        for i in range(max(0, min_frame - fps * 2), min_frame):
            save_path = path + '/img_{0:03d}_{1:05d}.jpg'.format(episode, int(i))
            cmd = ['rm', save_path]
            exec_subproc(cmd, 1)
    else:
        for i in range(max_frame + 1, max_frame + int(fps/10 * 2)):
            save_path = path + '/img_{0:03d}_{1:05d}.jpg'.format(episode, int(i))
            cmd = ['rm', save_path]
            exec_subproc(cmd, 1)



def get_directory(path, list_episodes):
    video_dict = {}
    file_list = sorted(listdir(path + '/af_jpg'))

    for f in file_list:
        s_f = f.split('_')
        ep_index = int(s_f[1])
        if ep_index in list_episodes and not ep_index in video_dict.keys():
            #print (f)
            max_ep = int (s_f[-1].split('.')[0] )
            min_ep = int (s_f[-1].split('.')[0] )

            for f_ in file_list:
                sf2 = f_.split('_')[2]
                sf3 = sf2.split('.')[0]
                #print(sf3)
                fr_index = int(sf3)
                ep_index_ = int(f_.split('_')[1])
                # print fr_index
                # print f_
                if ep_index_ == ep_index:

                    if fr_index >= max_ep:
                        max_ep = fr_index
                    if fr_index <= min_ep:
                        min_ep = fr_index
            # print('min', min_ep)
            # print('max', max_ep)
            frame_index = s_f[-1].split('.')[0]
            video_dict[ep_index] = frame_index
            duplicated_image_path = path + '/af_jpg/img_{0:03d}_{1:05d}.jpg'.format(ep_index, min_ep)
            duplicate_images(duplicated_image_path, ep_index, min_ep, max_ep, 0)
            cmd = get_conctac_cmd(path, ep_index, max(0, min_ep - fps * 2), 0)
            print (cmd)
            exec_subproc(cmd, 1)
            remove_duplicates(duplicated_image_path, ep_index, min_ep, max_ep, 0)


            duplicated_image_path = path + '/af_jpg/img_{0:03d}_{1:05d}.jpg'.format(ep_index, max_ep)
            duplicate_images(duplicated_image_path, ep_index, min_ep, max_ep, 1)
            cmd = get_conctac_cmd(path, ep_index, min_ep, 1)
            print (cmd)
            exec_subproc(cmd, 1)
            remove_duplicates(duplicated_image_path, ep_index, min_ep, max_ep, 1)


# def pre_images_copy(image_name, fps, path):
#     jpeg_pattern = path + '/af_jpg/img_{0:03d}_%05d.jpg'.format(int(ep_index))

def read_episodes_list(path):
    file_episodes_list = path + '/episodes.txt'
    episodes = []
    #	with open(file_episodes_list) as f:
    #    	content = f.readlines()
    for line in open(file_episodes_list):
        episodes.append(int(line))
    return episodes


def get_directory_blank(path, blank_path):
    video_dict = {}
    # ep_2016-05-11_SJCM0003_123_n
    for f in sorted(listdir(path)):
        s_f = f.split('_')
        ep_index = '_'.join(s_f[:-1])
        if not ep_index in video_dict.keys():
            print (f)
            blamk_name = ep_index + '_t.jpg'
            video_dict[ep_index] = blamk_name
            cmd = ['cp', blank_path, path + '/' + blamk_name]
            exec_subproc(cmd, 1)


path = '/home/volodymyr/v/2016-05-11'
path = sys.argv[1]
make_blank = 0

for i, param in enumerate(sys.argv):
    if param == '-b':
        make_blank = 1
        blank_path = sys.argv[i + 1]

if make_blank:
    get_directory_blank(path, blank_path)
    exit(0)

get_directory(path, read_episodes_list(path))
