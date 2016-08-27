from __future__ import division
__author__ = 'Volodymyr Varchuk'




from PIL import Image
from PIL.ExifTags import TAGS
import pprint
import fnmatch
import os
import exif
import json


def get_files(dir_path):
    matches = []
    for root, dirnames, filenames in os.walk(dir_path):
        print(root)
        # print(dirnames)
        print(len(filenames))
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '*.nef'):
            matches.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '*.JPG'):
            matches.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '*.NEF'):
            matches.append(os.path.join(root, filename))
    return matches

def get_exif(i):
    ret = {}
    info = i._getexif()
    if info is not None:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == 'FocalLength':
                v1, v2 = value
                value = int(v1/v2)
            if decoded == 'ExposureTime':
                v1, v2 = value
                if v1 < v2:
                    value = str('1/{0}'.format(int(v2/v1)))
                else:
                    value = v1/v2
            ret[decoded] = str(value)
    return ret

def get_nikon_exif(path_name):
    f = open(path_name, 'rb')
    tags = exif.process_file(f)
    f.close()
    return tags


def get_exif_param(exif_dict, param):
    if param in exif_dict.keys():
        return str(exif_dict[param]).strip()
    else:
        return 'None'


def get_exif_re(exif_dict, tag_value):
    if tag_value in exif_dict.keys():
        return str(exif_dict[tag_value])

def get_files_stats(search_dir):
    files  = get_files(search_dir)
    stats = []
    for file_name in files:
        s_path = file_name.split(os.path.sep)
        skip = False
        for p_element in s_path:
            if p_element.startswith('convert'):
                skip = True
                # print(file_name, 'SKIPPED')
        if not skip:
            im = Image.open(file_name)
            if file_name.endswith('.nef') or file_name.endswith('.NEF'):
                exif_dict = get_nikon_exif(file_name)
                d_element = {}
                d_element['dir'] = os.path.sep.join(file_name.split(os.path.sep)[:-1])
                d_element['file'] = file_name
                d_element['FocalLength'] = get_exif_re(exif_dict, 'EXIF FocalLength')
                d_element['ExposureTime'] = get_exif_re(exif_dict, 'EXIF ExposureTime')
                d_element['ISOSpeedRatings'] = get_exif_re(exif_dict, 'EXIF ISOSpeedRatings')
                d_element['Make'] = get_exif_re(exif_dict, 'Image Make')
                d_element['Model'] = get_exif_re(exif_dict, 'Image Model')
                d_element['Date'] = get_exif_re(exif_dict, 'EXIF DateTimeDigitized')
                d_element['Orientation'] = 'Horizontal' if get_exif_param(exif_dict, 'Image Orientation').startswith('Horizontal') else 'Vertical'
            else:
                exif_dict = get_exif(im)
                d_element = {}
                d_element['dir'] = os.path.sep.join(file_name.split(os.path.sep)[:-1])
                d_element['file'] = file_name
                d_element['FocalLength'] = get_exif_param(exif_dict, 'FocalLength')
                d_element['ExposureTime'] = get_exif_param(exif_dict, 'ExposureTime')
                d_element['ISOSpeedRatings'] = get_exif_param(exif_dict, 'ISOSpeedRatings')
                d_element['Make'] = get_exif_param(exif_dict,'Make')
                d_element['Model'] = get_exif_param(exif_dict,'Model')
                d_element['Date'] = get_exif_param(exif_dict, 'DateTimeDigitized')
                d_element['Orientation'] = 'Horizontal' if get_exif_param(exif_dict, 'Orientation') in [1,2,3,4] else 'Vertical'
            stats.append(d_element)
    return stats


def group_data(stats):
    grouped_data = {}
    for image_data in stats:
        for param_name in image_data:
            if param_name == 'file' or param_name == 'dir' or param_name == 'Date':
                continue
            if not param_name in grouped_data.keys():
                grouped_data[param_name] = {image_data[param_name]:1}
            if image_data[param_name] in grouped_data[param_name].keys():
                grouped_data[param_name][image_data[param_name]] = grouped_data[param_name][image_data[param_name]]+1
            else:
                grouped_data[param_name][image_data[param_name]] = 1
    return grouped_data


def group_data2(stats, total_count):
    grouped_data2 = {}

def dump_to_file(stats, file_name):
    f_out =  open(file_name, 'w')
    headers = stats[0].keys()
    sss = ';'.join(headers) + ';\n'
    f_out.write(sss)
    for row in stats:
        sss = ';'.join(row.values()) + ';\n'
        f_out.write(sss)
    f_out.close()


nef_file = '/media/sf_share_linux/p/DSC_0034.NEF'
jpg_file = '/media/sf_share_linux/p/DSC_0011.JPG'

csv_file = '/media/sf_share_linux/p/raw.txt'
stat_file = '/media/sf_share_linux/p/stat.txt'

print(get_nikon_exif(nef_file))
print(get_exif(Image.open(jpg_file)))
pp = pprint.PrettyPrinter(indent=4)
# search_dir = '/media/sf_All_Fotos/US Foto'
search_dir = u'/media/sf_All_Fotos/US Foto/LV/Allfotos/2016-03-12_01_Road_To_SR'

stats = get_files_stats(search_dir)
dump_to_file(stats, csv_file)
grouped_data = group_data(stats)
pp.pprint(grouped_data)

st_file = open(stat_file, 'w')
json.dump(grouped_data, st_file)
st_file.close()