__author__ = 'Volodymyr Varchuk'



import cv2
import itertools
import operator
import sys
import os
import subprocess
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
        if f_count == 500:
            break
        if f_count % step == 0:
            print(f_count, 'from', total_files)
        full_name = join(path, f)
        if isfile(full_name):
            hist_val = int(trunc_val(get_image_histogram(cv2.imread(full_name))))
            #hist_array.append({full_name:hist_val})
            hist_array['fnames'].append(full_name)
            hist_array['values'].append(hist_val)
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
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    cmd = ['ffmpeg', '-r', str(frame_rate_normal), '-y', '-i', '/'.join(path_to_save.split('/')[:-1])+'/jpg/'+pattern+'_*.jpg', path_to_save+'/'+pattern+'_normal.mpg']
    print(cmd)
    exec_subproc(cmd)
    cmd = ['ffmpeg', '-r', str(frame_rate_slow), '-y', '-i', '/'.join(path_to_save.split('/')[:-1])+'/jpg/'+path_to_save+'/'+pattern+'_lf*.jpg', path_to_save+'/'+pattern+'_slow.mpg']
    exec_subproc(cmd)
    print(cmd)

def exec_subproc(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    for lines in out.splitlines():
        print('\t', lines)
    return p.returncode


if len(sys.argv) > 1:
    video_path = sys.argv[1]
    print('video ', video_path)
    w = '/'.join( video_path.split('/')[:-1])
    f_name = video_path.split('/')[-1]

    w_path = '/'.join([w, f_name.split('.')[0]])
    print(w_path)
    if not os.path.exists(w_path):
        os.makedirs(w_path)
    images_path = w_path+'/jpg'

    if not os.path.exists(images_path):
        os.makedirs(images_path)
    print('splitting into frames...')
    cmd = ['ffmpeg', '-i', video_path, images_path+'/image%6d.jpg']
    # if exec_subproc(cmd) <> 0:
    #     exit(1)
    print('getting histograms for frames...')
    hists =get_histogram_directory(images_path)
    most = most_common(hists['values'])
    fps = int(round(get_frame_rate(video_path)))
    treshold = get_threshold(most, 0.5)
    most = round(most + most * 0.5)
    most_pow = most * 2
    print(hists['values'])
    print(hists['fnames'])
    print('most_common', most_common(hists))
    print('fps', fps)
    episode_count = 0
    episode_list = []
    short_range_count = 0
    short_range = 15
    min_frames = 10
    found = 0
    end_af = 0
    for i in range(len(hists['fnames'])):
        if i % int(len(hists['fnames'])/10) == 0:
            print (i, ' of ', len(hists['fnames']))
        #detecting beginning of the range with flashes
        if check_range(hists['values'], i, short_range, most, min_frames, 0):
            start_lf = i
            found = 1
        if check_range(hists['values'], i, short_range, most, min_frames, 1) and found:
            end_lf = i
            start_af = int(start_lf - fps * 2 if (start_lf - fps * 2) > 0 else 0)
            end_af  = int(i + fps * 2 if (i + fps * 2) < len(hists['values']) else len(hists['fnames']))
            episode_count = episode_count + 1
            episode_list.append( 'e_{eindex}'.format(eindex=episode_count) )
            print(start_af, end_af, range(start_af, end_af))
            for j in range(start_af, end_af):
                if j < start_lf and j > end_lf:
                    str_prefix = 'af'
                else:
                    str_prefix = 'lf'
                findex_str = hists['fnames'][j].split('/')[-1].split('.')[0][5:]
                new_fname = w_path+'/e_{eindex}_{str_prefix}_{findex}.jpg'.format(eindex=str(episode_count), str_prefix=str_prefix, findex=findex_str)
                cmd = ['cp', hists['fnames'][j], new_fname]
                exec_subproc(cmd)
                #print(j,  hists['fnames'][j])
                #print(cmd)
            start_af = 0
            end_lf = 0
            start_lf = 0
            found = 0
        if i < end_af:
            continue
        elif i == end_af:
            end_af = 0
    for episode in episode_list:
        print (episode)
        make_video(episode, w_path+'/video',int(fps),int(fps/5))



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
