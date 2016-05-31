__author__ = 'Volodymyr Varchuk'

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import math

import cv2
import argparse


def is_point_inside_circle(center, point, radius):
    x0, y0 = center
    x, y = point
    if (x - x0)*(x - x0) + (y - y0)*(y - y0) <= radius * radius:
        return True
    else:
        False


def is_point_acceptable(point, img_grayscale, treshold, radius):
    max_width, max_height = img_grayscale.shape
    max_i, max_j, max_val = point
    wrong_point = False
    sq_radius = int(round(radius, 0))
    sq_radius2 = sq_radius * 5
    for i in range (max(0,max_i - sq_radius), min(max_width, max_i + sq_radius)):
        for j in range (max(0, max_j - sq_radius), min(max_height, max_j + sq_radius)):
            if is_point_inside_circle((max_i, max_j), (i, j), radius):
                if img_grayscale[i,j] < int(max_val - max_val * treshold):
                    wrong_point = True
    if wrong_point:
        return False
    else:
        wrong_point2 = False
        for i in range (max(0, max_i - sq_radius2), min(max_width, max_i + sq_radius2)):
            for j in range (max(0, max_j - sq_radius2), min(max_height, max_j + sq_radius2)):
                if is_point_inside_circle((max_i, max_j), (i, j), sq_radius2):
                    if img_grayscale[i,j] < int(max_val - max_val * treshold):
                        return True
        if not wrong_point2:
            return False


def get_max_point(img_grayscale, point_radius):
    max_i = 100
    max_j = 100
    max_val = 220
    indent = 150
    im_h, im_w = img_grayscale.shape
    max_points = []
    for i in range(indent, im_h-indent):
        for j in range(indent, im_w-indent):
            #print (img_grayscale[i,j], max_val)
            if img_grayscale[i,j] >= max_val:
                #print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if is_point_acceptable ((i, j, img_grayscale[i,j]), img_grayscale, 0.1, point_radius):
                    max_val = img_grayscale[i,j]
                    max_i = i
                    max_j = j

    return (max_i, max_j, max_val)


def open_img(image_path):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # grayscale
    return gray_img


def round_points(center, radius):
    X0, Y0, max_value = center
    coordinates = []
    cirlce_points = []
    cp_step = 360.0/1200.0
    for i in range(1200):
        cp_val = i*cp_step
        cirlce_points.append(cp_val)
    for i in cirlce_points:
        Y1 = math.cos(math.radians(i)) * radius + Y0
        X1 = math.sin(math.radians(i)) * radius + X0
        point = (int(Y1), int(X1))
        if not point in coordinates:
            coordinates.append(point)

    return coordinates


def find_chains(whites_point, image_gary, iterations, draw, step_circle):
    success_count = 0
    x0, y0, max_val = whites_point
    check_reverse = 0
    direction = 0 #0 - East(45-134), 1 - North(135-224), 2 - West(225-314), 3 - South(315-44)
    points = []
    angle = 0
    step_angle = 190/40
    for i in range(iterations):
        # print (i)
        # start from 0 degree to 360
        if check_reverse == 0:
            range_cust = xrange(0, 180, (190 / 40))
        else:
            range_cust = xrange(180, 360, (190 / 40))
        print (range_cust)
        find_in_circle = 0
        for deg in range_cust:
            x = int(math.cos(math.radians(deg)) * step_circle + x0)
            y = int(math.sin(math.radians(deg)) * step_circle + y0)
            point_to_check = (x,y)
            if draw is not None:
                draw.point((y,x), 128)
            points.append(point_to_check)
            if is_point_acceptable((x,y,max_val),image_gary,0.1,0.5):
                x0, y0 = point_to_check
                success_count = success_count + 1
                find_in_circle = 1
                # print (x0,y0)
                # angle = deg
                break
        if find_in_circle == 0:
            if check_reverse == 1:
                return False
            else:
                x0, y0, max_val = whites_point
                check_reverse = 1
                # angle = 180
                # print ('reverse')
    print (success_count, iterations)
    if success_count == iterations:
        return True
    else:
        return False

#img_055_19541.jpg
#img_105_67039.jpg

#-f /media/sf_share_linux/video/2016-05-18/SJCM0005/af_jpg/img_017_08231.jpg

#-f /media/sf_share_linux/video/2016-05-18/SJCM0004/af_jpg/img_080_45696.jpg
#-f /media/sf_share_linux/video/2016-05-18/SJCM0004/af_jpg/img_105_67039.jpg
#-f /media/sf_share_linux/video/img_055_19541.jpg
# -f /tmp/SJCM0003/jpg/image007822.jpg



# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--file", required=True, help="Path to the image file")
# args = vars(ap.parse_args())
#
# print('read_image')
# gray_img = open_img(args['file'])
# max_point = (0,0,0)
# print('getting max point')
# max_point = get_max_point(gray_img, 3)
# print (max_point)
#
#
# rrr = round_points(max_point, 4)
# im = Image.open(args['file'])
# draw = ImageDraw.Draw(im)
#
# for it in rrr:
#     draw.point(it, 128)
#
#
# found = find_chains(max_point,gray_img,20,draw,10)
# print (found)
# # for it in points:
# #     draw.point(it, 128)
# del draw
#
# # find_white_points_chain(max_point, 5, 5, gray_img, 255, draw)
# im.save("output.png")
# im.show()

#print (rrr)