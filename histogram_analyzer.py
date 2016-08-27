import json

__author__ = 'Volodymyr Varchuk'



import itertools
import operator
import sys
import shutil
from matplotlib import pyplot as plt
import numpy


from os import listdir
from os.path import isfile, join

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from utils import *

file_type_episode = 'mp4'
type_short_video = 'short'
type_long_video = 'long'
type_slow_video = 'slow'


#command line parameters
load_histogram = 0
cut_video = 0
show_histogram = 0
make_plot = 0
make_slow_copy = 0
ratio = 10
concat_videos = 0
concat_dir = 0
reload_histogram = 0
slow_from_frame = 0
log_only = 0
make_from_farames = 0
batch_execute = 0
load_episodes = 0




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
    return max(groups, key=_auxfun)[0]


def get_distribution(array):
    distribution_list = {}
    for i in array:
        if str(i) in distribution_list.keys():
            distribution_list[str(i)] =+ distribution_list[str(i)] + 1
        else:
            distribution_list[str(i)] = 0
    return distribution_list


def distribution_print (array):
    for i in array:
        print('{0}\t{1}'.format(i, array[i]))


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
        if f_count % step == 0:
            print(' '.join([str(f_count), 'of', str(total_files), '...']))
        full_name = join(path, f)
        if isfile(full_name):
            hist_val = getHistGrayscale(cv2.imread(full_name, 0))# int(trunc_val(get_image_histogram(cv2.imread(full_name))))
            hist_array['fnames'].append(full_name)
            hist_array['values'].append(hist_val)
    print('[DONE]')
    return hist_array


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


def get_episode_file_name(video_path, episode_count, video_type):
    w_path = get_working_path(video_path)
    video_short_name = get_short_name(video_path)
    episode_file_name = '/'.join([w_path, 'video', video_short_name + '_' + '{0:04d}'.format(episode_count) + '_'+video_type+'.'+file_type_episode])
    return episode_file_name


def get_cmd_cut(start_frame, end_frame, fps, video_path, episode_count, video_type ):
    tstart = frame_number_to_time(start_frame if start_frame >= 0 else 0, fps, 1)
    tduration = frame_number_to_time(end_frame - start_frame, fps, 1)
    cmd = ['ffmpeg', '-i', video_path, '-ss', tstart, '-t', tduration, '-y', '-vcodec', 'copy', get_episode_file_name(video_path,episode_count,video_type)]
    return cmd


def get_cmd_slow(video_path, episode_count, ratio):
    episode_file_name_in = get_episode_file_name(video_path,episode_count,type_short_video)
    episode_file_name_out = get_episode_file_name(video_path,episode_count,type_slow_video)
    cmd = ['ffmpeg', '-y', '-i', episode_file_name_in, '-filter:v', 'setpts={0}*PTS'.format(int(ratio)), episode_file_name_out]
    return cmd


def find_episodes(histrogars_array, most_common_element, fps, extremum_coefficient):
    extremum_element = most_common_element * extremum_coefficient
    last_episode = {'start':0, 'end':0}
    episodes = []
    part_count = int(len(histrogars_array['values'])/10)
    for i in range(len(histrogars_array['fnames'])):
        if i % part_count == 0:
            end_part = i+part_count if i+part_count < len(histrogars_array['values']) else len(histrogars_array['values'])
            extremum_element = most_common(histrogars_array['values'][i:end_part]) * extremum_coefficient
            print ('extremum_element {0}'.format(extremum_element))
        if last_episode['end'] > i:
            continue
        if histrogars_array['values'][i] != 0 and  histrogars_array['values'][i] >= extremum_element:
            s = i - int(1 * fps)
            e = i + int(2 * fps)
            start_frame = s if s >= 0 else 0
            end_frame = e if e <= len(histrogars_array['values']) else len(histrogars_array['values'])

            if start_frame != i:
                start_localMFE = most_common( histrogars_array['values'][start_frame:i])
            else:
                start_localMFE = most_common_element

            if i != end_frame:
                end_localMFE = most_common( histrogars_array['values'][i:end_frame] )
            else:
                end_localMFE = most_common_element
            start_pos = -1
            end_pos = -1

            for j in range(start_frame, i + 1):
                if histrogars_array['values'][j] > (start_localMFE * extremum_coefficient):
                    start_pos = j
                    break
            for j in reversed (range(i, end_frame)):

                if histrogars_array['values'][j] > (end_localMFE * extremum_coefficient):
                    end_pos = j + 2
                    break
            if end_pos < 0 or start_pos < 0:
                continue
            else:
                last_episode = {'start':start_pos, 'end':end_pos}
                episodes.append({'{0:03d}'.format(len(episodes)):last_episode})
                print(last_episode, extremum_element)
    return episodes


def get_localMFE(hists_values, frame, fps):
    s = frame - 2 * fps
    e = frame + 2 * fps
    start_frame = s if s >= 0 else 0
    end_frame = e if e >= 0 else 0
    new_array = hists_values[start_frame:end_frame]
    #print (start_frame, end_frame, new_array)
    return most_common(new_array)


def check_seq_end_pos(histrogars_array, index, sequ_check, sequ_frames, treshold):
    end_index = index + sequ_check
    sequ_frames_count = 0
    if end_index > len(histrogars_array['values']):
         end_index = len(histrogars_array['values'])
    for i in range(index, end_index):
        if histrogars_array['values'][i] <= treshold:
            sequ_frames_count = sequ_frames_count + 1
    if sequ_frames_count >= sequ_frames:
        return index
    else:
        return -1


def concatenate_videos(path):
    print(path)
    files = sorted(listdir(path))
    videos_list_file = path + '/videos.txt'
    concat_video = path+'/videos.mp4'
    f = open(videos_list_file, 'w')
    print(files)
    for file in files:
        str_line = "file '{0}'\n".format(file)
        print(str_line)
        f.write(str_line)
    f.close()
    cmd = ['ffmpeg', '-f', 'concat', '-i', videos_list_file, '-c', 'copy', concat_video]
    return cmd

def concatenate_dir(path):
    print(path + '/video')
    files = sorted(listdir(path + '/video'))
    videos_list_file = path + '/video/videos.txt'
    concat_video = path+'/videos.mp4'
    f = open(videos_list_file, 'w')
    for file in files:
        str_line = "file '{0}/video/{1}'\n".format(path, file)
        f.write(str_line)
    f.close()
    cmd = ['ffmpeg', '-f', 'concat', '-i', videos_list_file, '-c', 'copy', concat_video]
    return cmd

def getHistGrayscale(img):
#    img = cv2.imread(img,0)
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten().astype(int)
    summa = numpy.int64(0)
    i = 0
    for e in hist:
        summa = summa + (numpy.int64(e)/256) * i
        i = i + 1
    return summa


def invert_h_values(hist_values):
    inverted = []
    max_val = max(hist_values)
    min_val = min(hist_values)
    for el in hist_values:
        inverted.append(max_val - el - min_val)
    return inverted




def copy_frames(episode, video_path, filelist, episode_count):
    video_name = get_short_name(video_path)
    af_path = get_af_path_path(video_path)
    font_size = 16
    font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerif.ttf", font_size)
    fps = get_frame_rate(video_path)

    for i in range(episode['start'] - 2, episode['end'] + 1):
        #print(filelist[i])
        img_ = Image.open(filelist[i])
        width, height = img_.size
        draw = ImageDraw.Draw(img_)
        draw.text((0,0), '{0} {1}'.format(video_name, str(episode_count)) ,(255,255,255),font=font)
        draw.text((0, height - font_size - 10), 'Volodymyr Varchuk' ,(255,255,255, 128),font=font)
        jpg_name = af_path +'/img_{0:03d}_{1:05d}.jpg'.format(episode_count, i)
        img_.save(jpg_name)
    f = open(w_path+'/log.csv' ,'a')
    logstr = '\n*************************{0}*************************\n'.format(episode_count)
    jpg_pattern = af_path +'/img_{0:03d}_%05d.jpg'.format(episode_count)
    episode_video_name = '{0}/{1}_{2:04d}_{3}.{4}'.format(get_video_path(video_path), video_name,episode_count, type_short_video,file_type_episode)
    cmd = ['ffmpeg', '-r', str(int(fps)), '-y', '-start_number', '{0:05d}'.format(episode['start'] - 2),  '-i', jpg_pattern, '-c:v', 'libx264', '-r', str(int(fps)), episode_video_name]

    loginfo = ' '.join(cmd)
    logstr = loginfo + '\n'
    exec_subproc(cmd)

    if make_slow_copy:
        cmd_slow = get_cmd_slow(video_path,episode_count,10)
        loginfo = ' '.join(cmd_slow)
        logstr = logstr + loginfo + '\n'
        exec_subproc(cmd_slow)

    if slow_from_frame:
        episode_video_name = '{0}/{1}_{2:04d}_{3}_frames.{4}'.format(get_video_path(video_path), video_name,episode_count, type_slow_video,file_type_episode)
        cmd = ['ffmpeg', '-r', str(int(fps)), '-y', '-start_number', '{0:05d}'.format(episode['start'] - 2),  '-i', jpg_pattern, '-c:v', 'libx264', '-r', str(int(fps)), episode_video_name]
        loginfo = ' '.join(cmd)
        logstr = logstr + loginfo + '\n'
        exec_subproc(cmd)
    f.write(logstr)
    f.close()

    return 0
timer = TimeCounter()
sys.stdout = Unbuffered(sys.stdout)

if len(sys.argv) > 1:
    video_path = sys.argv[1]
    for i, param in enumerate(sys.argv):
        #load histograms from csv file
        if param == '-h':
            print('{0} \t-load histograms for images from previously created "historamms.csv" file'.format(param))
            load_histogram = 1
        #cut video to fragments
        if param == '-c':
            print('{0} \t-use "cuting video" method with ffmpeg to create episodes'.format(param))
            cut_video = 1
        #make from frames
        if param == '-m':
            print('{0} \t-use "concatenating frames" method with ffmpeg to create episodes'.format(param))
            make_from_farames = 1
        #show histogram in window
        if param == '-s':
            print('{0} \t-show histogramm plot (works only with -p parameter)'.format(param))
            show_histogram = 1
        #make plot
        if param == '-p':
            print('{0} \t-make histogramm plot and save to *.png file'.format(param))
            make_plot = 1
        #cut video and make slow episode from short one
        if param == '-slow':
            print('{0} \t-make slow copy'.format(param))
            make_slow_copy = 1
        #ratio. set slowness factor
        if param == '-r':
            print('{0} \t-make slow copy with ratio'.format(param))
            ratio = int(sys.argv[i + 1])
            print(ratio)
        #concatenate episodes to a single file
        if param == '-concat':
            print('{0} \t-concatenate video files from directory to single video file, as base directory uses filename path'.format(param))
            concat_videos = 1
        #concatenate episodes located in directory to a single file
        if param == '-cd':
            print('{0} \t-concatenate video files from directory to single video file '.format(param))
            concat_dir = 1
            concat_dir_path = str(sys.argv[i + 1])
            print(concat_dir_path)
        #just reload histograms from existant jpgs
        if param == '-rh':
            print('{0} \t-reload historgamms from images, make new histogramm.csv'.format(param))
            reload_histogram = 1
        if param == '-f':
            print('{0} \t-make slow copy from frames'.format(param))
            slow_from_frame = 1
        #only logging info
        if param == '-l':
            print('{0} \t-only logging, without executing command'.format(param))
            log_only = 1
        #batch execute
        if param == '-b':
            batch_name = sys.argv[i + 1]
        #load episodes
        if param == '-le':
            load_episodes = 1

    print('video ' + video_path)
    video_short_name = get_short_name(video_path)
    w_path = get_working_path(video_path)

    if concat_dir:
        print('conctenating videos...')
        cmd = concatenate_videos(concat_dir_path)
        print(' '.join(cmd))
        exit (exec_subproc(cmd, 1))

    if batch_execute:
        batch_execute_proc(batch_name)

    af_path = get_af_path_path(video_path)
    print('working path ' + w_path)
    print('getting frame rate... ')
    fps = int(round(get_frame_rate(video_path)))
    print('fps ' + str(fps))
    images_path = get_jpg_path(video_path)
    if not load_histogram:
        if reload_histogram:
            print('getting histograms for frames...')
            hists =get_histogram_directory(images_path)
            print('saving histogram...')
            save_histogram_list(hists, w_path)
        else:
            if not os.path.exists(w_path):
                os.makedirs(w_path)
            else:
                shutil.rmtree(w_path)
                os.makedirs(w_path)
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            print('splitting into frames...')
            if not os.path.exists(video_path):
                print('File {0} not found'.format(video_path))
                exit(0)
            cmd = ['time', 'ffmpeg', '-i', video_path ,'-qscale:v', '2', images_path+'/image%6d.jpg']
            exec_subproc(cmd)

            print('getting histograms for frames...')
            timer.start_count()
            hists =get_histogram_directory(images_path)
            timer.end_count()
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

    #making plot for distribution for video
    if make_plot:
        timer.start_count()
        print('creating plot..')
        plt.clf()
        x = [ count for count in range(len(hists['values'])) ]
        xlabels = []
        mod_val = int(len(hists['values'])/5)
        for count in range(len(hists['values'])):
            if count % mod_val == 0:
                xlabels.append(frame_number_to_time(count, fps, 1))
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
        timer.end_count()

    print('getting most frequent element... ')
    most = most_common(hists['values'])
    print('most_common ' + str(most))
    treshold = get_threshold(most, 2)
    #most = round(most + most * 2)
    most_pow = most * 2
    episode_count = 0
    episode_list = []
    short_range_count = 0
    short_range = int(fps/4)
    min_frames = int(short_range - short_range * 0.25)
    found = 0
    end_af = 0

    print('finding lightnings...')
    timer.start_count()
    if load_episodes == 1:
        episodes_path = p.join(w_path, 'episodes_list.txt')
        print('epath' ,episodes_path)
        e_file = open(episodes_path, 'r')
        episodes = json.load(e_file)
    else:
        episodes = find_episodes (hists,most,fps,2)
        print(episodes)
        with open(p.join(w_path, 'episodes_list.txt'), 'w') as ofile:
            json.dump(episodes, ofile)
        ofile.close()
    timer.end_count()

    print('making videos...')
    print('total videos ' + str(len(episodes)))
    if cut_video:
        episode_count = 0
        f = open(w_path+'/log.csv' ,'w')
        for episode_key in episodes:
            episode = episode_key.itervalues().next()
            str_text = '\n'
            print(episode, frame_number_to_time(episode['start'], fps, 1), frame_number_to_time(episode['end'], fps, 1))
            str_text = str_text + ' #### '.join([str(episode_count), str(episode['start']), str(episode['end']), frame_number_to_time(episode['start'], fps, 1), frame_number_to_time(episode['end'], fps, 1), '\n\t'])

            cmd = get_cmd_cut( episode['start'] ,episode['end'], fps,video_path,episode_count, type_short_video)
            print( ' '.join(cmd))
            str_text = str_text + ' '.join(cmd) + '\n\t'
            if not log_only:
                exec_subproc(cmd, 0)

            cmd = get_cmd_cut( episode['start'] - fps * 2 if episode['start'] - fps * 2 > 0 else 0 ,episode['end'] + fps * 2 if episode['end'] + fps < len(hists['values']) else len(hists['values']), fps,video_path,episode_count, type_long_video)
            print( ' '.join(cmd))
            str_text = str_text + ' '.join(cmd) + '\n'
            if not log_only:
                exec_subproc(cmd, 0)

            if make_slow_copy:
                cmd = get_cmd_slow(video_path, episode_count, ratio)
                print( ' '.join(cmd))
                str_text = str_text + '\t' + ' '.join(cmd) + '\n'
                if not log_only:
                    exec_subproc(cmd, 0)
                if slow_from_frame:
                    cmd = ''
                    frame_rate_normal = str(int(fps / ratio))
                    #cmd = ['ffmpeg', '-r', frame_rate_normal, '-y', '-pattern_type', 'glob', '-i', '/'.join(path_to_save.split('/')[:-1])+'/af_jpg/'+pattern+'_af_*.jpg', '-c:v', 'copy', path_to_save+'/'+pattern+'_normal.avi']

            f.write(str_text)
            episode_count = episode_count + 1
        f.close()
    if make_from_farames:
        for count, episode_key in enumerate(episodes):
            episode = episode_key.itervalues().next()
            copy_frames(episode,video_path,hists['fnames'],count)

    if concat_videos:
        print('concatenate videos to single file...')
        cmd = concatenate_videos(w_path)
        print( ' '.join(cmd))
        exec_subproc(cmd, 1)
