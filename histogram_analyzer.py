__author__ = 'Volodymyr Varchuk'



import cv2
import itertools
import operator
import sys
import os
import subprocess
import shutil
#import numpy as np
from matplotlib import pyplot as plt

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
            print (out)
        else:
            print( ' [ERROR] ' + ' '.join(cmd))
            print (out)
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
    f.close()
    return hists

#hists = load_histogram_list()

def frame_number_to_time(frame_number, fps, precision = 0):
    if fps <= 0:
        return '00:00:00'
    hour = 0
    minutes = 0
    sec = int(frame_number / fps)
    microsec = int((frame_number - sec * fps) * (1000/fps))
    if sec > 59:
        minutes = int(sec / 60)
        sec = sec - minutes * 60
        if minutes > 59:
            hour = int(minutes/60)
            minutes = minutes - hour * 60
    if precision == 1:
        time_str = '{0:02d}:{1:02d}:{2:02d}.{3:03d}'.format(hour,minutes,sec, microsec)
    else:
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


def get_video_name(path):
    return path.split('/')[-1]



def get_cmd_cut(start_frame, end_frame, fps, video_path, episode_count, video_type ):
    tstart = frame_number_to_time(start_frame, fps)
    if end_frame - start_frame < fps:
        tduration = frame_number_to_time(fps, fps)
    else:
        tduration = frame_number_to_time(end_frame - start_frame, fps)
    video_short_name = get_short_name(video_path)
    w_path = get_working_path(video_path)
    episode_file_name = '/'.join([w_path, 'video', video_short_name + '_' + '{0:04d}'.format(episode_count) + '_'+video_type+'.mkv'])
    cmd = ['ffmpeg', '-i', video_path, '-ss', tstart, '-t', tduration, '-vcodec', 'copy', '-acodec', 'copy',
           episode_file_name]
    return cmd



if len(sys.argv) > 1:
    video_path = sys.argv[1]
    load_histogram = 0
    cut_video = 0
    show_histogram = 0
    make_plot = 0
    for param in sys.argv:
        if param == '-h':
            load_histogram = 1
        if param == '-c':
            cut_video = 1
        if param == 's':
            show_histogram = 1
        if param == '-p':
            make_plot = 1

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
        exec_subproc(cmd)
        print('getting histograms for frames...')
        hists =get_histogram_directory(images_path)
        print('saving histogram...')
        save_histogram_list(hists, w_path)
    else:
        print('load histograms from {0}'.format(w_path+'/histogram.csv'))
        hists = load_histogram_list(w_path+'/histogram.csv')
    print('create dir for copying image with lightnings...')
    if not os.path.exists(af_path):
        os.makedirs(af_path)
    print('create dir for copying video fragments...')
    if not os.path.exists(get_video_path(video_path)):
        os.makedirs(get_video_path(video_path))
    else:
        shutil.rmtree(get_video_path(video_path))
        os.makedirs(get_video_path(video_path))


    #making plot for distribution for video
    if make_plot:
        print('creating plot..')
        plt.clf()
        #y =
        x = [ count for count in range(len(hists['values'])) ]
        xlabels = []
        mod_val = int(len(hists['values'])/5)
        for count in range(len(hists['values'])):
            if count % mod_val == 0:
                #print(count, frame_number_to_time(count, 1))
                xlabels.append(frame_number_to_time(count, 1))
            else:
                continue
        print(xlabels)
        #xlabels.append(frame_number_to_time(len(hists['values']), 1))
        plt.xticks (x, xlabels, rotation='vertical')
        plt.locator_params(axis='x', nbins=5)
        print('plot histogram...')
        plt.plot(x, hists['values'])
        print('plotting [DONE]')
        plt.xlabel('time (s)')
        plt.ylabel('hist value')
        plt.title('Video {0}'.format(get_video_name(video_path)))
        plt.grid(True)
        #plt.autoscale(enable=True,axis='both',tight=True)
        plot_save_path = '{0}/{1}.png'.format( w_path, get_short_name(video_path))
        if os._exists(plot_save_path):
             os.remove(plot_save_path)
        plt.savefig(plot_save_path)
        if show_histogram:
            plt.show()

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
                cmd = get_cmd_cut(start_af+fps*2,end_af + fps*2, fps,video_path,episode_count, 'normal')
                exec_subproc(cmd)
                cmd = get_cmd_cut(start_lf,end_lf+ fps*3, fps,video_path,episode_count, 'slow')
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
    print(get_video_path(video_path))
    if not cut_video:
        for episode in episode_list:
            make_video(episode, get_video_path(video_path),int(fps),int(fps/5))

