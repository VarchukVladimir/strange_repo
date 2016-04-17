__author__ = 'Volodymyr Varchuk'



import cv2
import itertools
import operator
import sys
import os
import subprocess
import shutil
#import numpy as np
#from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

image_set = []


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

def get_threshold(val, p):
    return {'min':trunc_val(val - val*p), 'max':int(round(val+val*p))}

def get_image_histogram(image):
    hist = []
    color = ('b','g','r')
    for j,col in enumerate(color):
        hist.append(cv2.calcHist([image],[j],None,[256],[0,256]))
    summ = 0
    for it in [ (hist[0][k] + hist[1][k] + hist[2][k])/3 for k in range(len(hist[0])) ]:
        summ =+ it[0]
    return summ/len(hist)


def get_histogram_directory(path):
    list_files = listdir(path)
    total_files = len(list_files)
    step = int(len(list_files)/20)
    f_count = 0
    hist_array = {'fnames':[], 'values':[]}
    for f in sorted(listdir(path)):
        f_count = f_count + 1
#set limit of files to 500 for testing it tajes a long time
        # if f_count == 500:
        #     break
        if f_count % step == 0:
            print(' '.join([str(f_count), 'of', str(total_files), '...']))
        full_name = join(path, f)
        if isfile(full_name):
            hist_val = int(trunc_val(get_image_histogram(cv2.imread(full_name))))
            #hist_array.append({full_name:hist_val})
            hist_array['fnames'].append(full_name)
            hist_array['values'].append(hist_val)
    print('[DONE]')
    return hist_array


def get_frame_rate(video_path):
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        #print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        #print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
    video.release()
    return fps

def check_if_in_range(treshold, value):
    if value >= treshold['min'] and value <= treshold['max']:
        return True
    else:
        False


def trunc_val(val):
    s_val = str(val).split('.')
    if len(s_val) == 0:
        return int(val)
    else:
        return int(s_val[0])

def check_range(check_list, startpos, num_elements, criteria, criteria_count, cmpr):

    max_r = startpos + num_elements
    if max_r > len(check_list):
        max_r = len(check_list)
    criteria_local_count = 0
    for i in range(startpos, max_r):
        if cmpr == 0:
            if check_list[i] >= criteria:
                criteria_local_count = criteria_local_count + 1
        else:
            if check_list[i] < criteria:
                criteria_local_count = criteria_local_count + 1

    if criteria_local_count >= criteria_count:
        return True
    else:
        return False

def make_video(pattern, path_to_save, frame_rate_normal, frame_rate_slow):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    print('pattern ' + pattern)
    cmd = ['ffmpeg', '-r', str(frame_rate_normal), '-y', '-pattern_type', 'glob', '-i', '/'.join(path_to_save.split('/')[:-1])+'/af_jpg/'+pattern+'_af_*.jpg', '-c:v', 'copy', path_to_save+'/'+pattern+'_normal.avi']
    exec_subproc(cmd)
    cmd = ['ffmpeg', '-r', str(frame_rate_slow), '-y', '-pattern_type', 'glob', '-i', '/'.join(path_to_save.split('/')[:-1])+'/af_jpg/'+pattern+'_lf_*.jpg', '-c:v', 'copy', path_to_save+'/'+pattern+'_slow.avi']
    exec_subproc(cmd)

def exec_subproc(cmd, show_info=1):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if show_info:
        if p.returncode == 0:
            print( '[OK] ' + ' '.join(cmd))
        else:
            print( ' [ERROR] ' + ' '.join(cmd))
            print(err)
    # for lines in out.splitlines():
    #     print('\t', lines)
    return p.returncode

def save_histogram_list(hist, save_path):
    f = open(save_path+'/histogram.csv' ,'w')
    for i in range(len(hist['fnames'])):
        str_line = hist['fnames'][i] +'\t'+ str(hist['values'][i])+'\n'
        f.write(str_line)
    f.close()

def load_histogram_list(load_path):
    hists = {'fnames':[], 'values':[]}
    f = open(load_path ,'r')
    lines = f.readlines()
    for line in lines:
        elem = line.split('\t')
        if len(elem) == 2:
            hists['fnames'].append(elem[0])
            hists['values'].append(int(elem[1]))
    #print(hists['fnames'])
    #print(hists['values'])
    f.close()
    return hists

#hists = load_histogram_list()

def frame_number_to_time(frame_number, fps):
    if fps <= 0:
        return '00:00:00'
    hour = 0
    minutes = 0
    sec = int(frame_number / fps)
    if sec > 59:
        minutes = int(sec / 60)
        sec = sec - minutes * 60
        if minutes > 59:
            hour = int(minutes/60)
            minutes = minutes - hour * 60
    time_str = '{0:02d}:{1:02d}:{2:02d}'.format(hour,minutes,sec)
    return time_str


def get_working_path(path):
    return '/'.join(path.split('/')[:-1] + [get_short_name(path)])


def get_jpg_path(path):
    return '/'.join([get_working_path(path)] + ['jpg'])


def get_af_path_path(path):
    return '/'.join([get_working_path(path)] + ['af_jpg'])


def get_video_path(path):
    return '/'.join([get_working_path(path)] + ['video'])


def get_short_name(path):
    return path.split('/')[-1].split('.')[0]


def get_cmd_cut(start_frame, end_frame, fps, video_path, episode_count, video_type ):
    tstart = frame_number_to_time(start_af, fps)
    if end_frame - start_frame < fps:
        tduration = '00:00:01'
    else:
        tduration = frame_number_to_time(end_frame - start_frame, fps)
    video_short_name = get_short_name(video_path)
    w_path = get_working_path(video_path)
    episode_file_name = '/'.join([w_path, 'video', video_short_name + '_' + '{0:04d}'.format(episode_count) + '_'+video_type+'.mkv'])
    cmd = ['ffmpeg', '-ss', tstart, '-t', tduration, '-i', video_path, '-vcodec', 'copy', '-acodec', 'copy',
           episode_file_name]
    return cmd


if len(sys.argv) > 1:
    video_path = sys.argv[1]
    load_histogram = 0
    cut_video = 0
    if len(sys.argv) >= 3:
        if sys.argv[2] == '-h':
            load_histogram = 1
        if sys.argv[3] == '-c':
            cut_video = 1

    print('video ' + video_path)
    video_short_name = get_short_name(video_path)
    w_path = get_working_path(video_path)
    af_path = get_af_path_path(video_path)
    print('working path ' + w_path)
    #if save_path+'histogram.csv'
    images_path = get_jpg_path(video_path)
    if not load_histogram:
        if not os.path.exists(w_path):
            os.makedirs(w_path)
        else:
            shutil.rmtree(w_path)
            os.makedirs(w_path)
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        print('splitting into frames...')
        cmd = ['time', 'ffmpeg', '-i', video_path, images_path+'/image%6d.jpg']
        print(' '.join(cmd))
#skip it for testing it takes a long time
        exec_subproc(cmd)
        print('getting histograms for frames...')
        hists =get_histogram_directory(images_path)
        print('saving histogram...')
        save_histogram_list(hists, w_path)
    else:
        hists = load_histogram_list(w_path+'/histogram.csv')
    if not os.path.exists(af_path):
        os.makedirs(af_path)

    print('getting most frequent element... ')
    most = most_common(hists['values'])
    print('most_common ' + str(most))
    print('getting frame rate... ')
    fps = int(round(get_frame_rate(video_path)))
    print('fps ' + str(fps))
    treshold = get_threshold(most, 2)
    most = round(most + most * 2)
    most_pow = most * 2
    episode_count = 0
    episode_list = []
    short_range_count = 0
    short_range = int(fps/4)
    min_frames = int(short_range - short_range * 0.25)
    found = 0
    end_af = 0
    print('finding lightnings...')
    for i in range(len(hists['fnames'])):
        if i % int(len(hists['fnames'])/10) == 0:
            print ( ' '.join([str(i), ' of ', str(len(hists['fnames']))]))
        #detecting beginning of the range with flashes
        if check_range(hists['values'], i, short_range, most, min_frames, 0) and not found:
            if hists['values'][i] >= most:
                start_lf = i
                found = 1
        if check_range(hists['values'], i, short_range, most, min_frames, 1) and found:
            end_lf = i + short_range
            start_af = int(start_lf - fps * 2 if (start_lf - fps * 2) > 0 else 0)
            end_af = int(i + fps * 2 if (i + fps * 2) < len(hists['values']) else len(hists['fnames']))
            print(start_af, start_lf, end_af, end_lf)
            episode_count = episode_count + 1
            episode_list.append('e_{eindex}'.format(eindex=episode_count))
            if cut_video == 1:
                cmd = get_cmd_cut(start_af,end_af,fps,video_path,episode_count, 'normal')
                exec_subproc(cmd)
                cmd = get_cmd_cut(start_lf,end_lf,fps,video_path,episode_count, 'slow')
                exec_subproc(cmd)
            else:
                for j in range(start_af, end_af):
                    findex_str = hists['fnames'][j].split('/')[-1].split('.')[0][5:]
                    if j < start_lf or j > end_lf:
                        str_prefix = 'af'
                    else:
                        new_fname = af_path + '/e_{eindex}_lf_{findex}.jpg'.format(eindex=str(episode_count),
                                                                                        findex=findex_str)
                        cmd = ['cp', hists['fnames'][j], new_fname]
                        exec_subproc(cmd, 0)

                    new_fname = af_path + '/e_{eindex}_af_{findex}.jpg'.format(eindex=str(episode_count),
                                                                                        findex=findex_str)
                    cmd = ['cp', hists['fnames'][j], new_fname]
                    exec_subproc(cmd, 0)

            start_af = 0
            end_lf = 0
            start_lf = 0
            found = 0
        if i < end_af:
            continue
        elif i == end_af:
            end_af = 0
    print('making videos...')
    print('total videos ' + str(len(episode_list)))
    if not cut_video:
        for episode in episode_list:
            make_video(episode, get_video_path(video_path),int(fps),int(fps/5))



        # if i < end_episode_index:
        #     continue
        # if hists['values'][i] >= most or short_range_count:
        #     if short_range_count == 0:
        #         store_index = i
        #     short_range_count =+ 1
        #     if short_range_count >= min_frames and (i - store_index) >= short_range:
        #         episode_count =+ 1
        #         episode_list.append('/e_{eindex}'.format(eindex=str(episode_count)))
        #
        #         # detecting ending of the range with flashes
        #         short_range_count = 0
        #         end_episode_index = 0
        #         for j in range(store_index, len(hists['fnames'])):
        #             if hists['values'][j] < most or short_range_count:
        #                 if short_range_count == 0:
        #                     end_episode_index = j
        #                 short_range_count =+ 1
        #                 if short_range_count >= min_frames and (j - end_episode_index) >= short_range:
        #                     break
        #         if end_episode_index == 0:
        #             end_episode_index = len(hists['fnames'])
        #         #calculating ranges
        #         start_af = store_index - fps * 2
        #         end_af = end_episode_index + fps * 2 + 1
        #         if end_af > len(hists['fnames']):
        #             end_af = len(hists['fnames'])
        #         if start_af <= 0:
        #             start_af = 1
        #
        #         start_lf = store_index
        #         end_lf = end_episode_index
        #
        #         #rename images
        #         for j in range(start_af, end_af):
        #             if j < start_lf and j > end_lf:
        #                 str_prefix = 'af'
        #             else:
        #                 str_prefix = 'lf'
        #             new_fname = w_path+'/e_{eindex}_{str_prefix}_{findex}'.format(eindex=str(episode_count), str_prefix=str_prefix, findex=j)
        #             cmd = ['mv', hists['fnames'][i], new_fname]
        #             print(cmd)
#                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#                    out, err = p.communicate()
#                     for lines in out.splitlines():
#                         print('\t', lines)
#                     if p.returncode <> 0:
#                         print(err)
#                         exit(1)
#                 short_range_count = 0
#             elif short_range_count < min_frames and (i - store_index) >= short_range:
#                 short_range_count = 0

        # else:
        #     episode_list.append(0)
    # print(episode_list)


# for i in range(1,86):
#     img_name = '/home/volodymyr/video/image{index}.jpg'.format(index=str(i))
#     #print(img_name)
#     plot_name = '/home/volodymyr/video/hist/image{index}.png'.format(index=str(i))
#     img = cv2.imread(img_name)
#     print( 'image{ind}.jpg\t{hist_val}'.format(ind=str(i), hist_val = str(get_image_histogram(img)) ))
#
#    plt.clf()
#     hist = []
#     for j,col in enumerate(color):
#         hist.append(cv2.calcHist([img],[j],None,[256],[0,256]))
#     histr = [ (hist[0][k] + hist[1][k] + hist[2][k] ) for k in range(len(hist[0])) ]
#     #crc_hist = sum(histr)
#     summ = 0
#     for j in range(len(histr)):
#         summ =+ histr[j][0]
#     print( 'image{ind}.jpg\t{hist_val}'.format(ind=str(i), hist_val = str(summ/len(histr)) ))
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
#     plt.savefig(plot_name)