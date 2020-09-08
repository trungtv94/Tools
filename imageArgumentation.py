# -*- coding:utf-8 -*-
# version 1.4


import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import argparse
import os
import cv2
import difflib

import numpy as np
from PIL import ImageFont, ImageDraw, Image
from shutil import copyfile

#import random
from random import *

from tqdm import tqdm
import fnmatch
import glob

import logging
import optparse

import random
import math
import re
import shutil

import json # import json module

def gt_file_write( gt_file , int_list , chartype , string ):
    gt_file.write(
        str( int_list[0] ) + ',' + str( int_list[1] ) + ',' +
        str( int_list[2] ) + ',' + str( int_list[3] ) + ',' +
        str( int_list[4] ) + ',' + str( int_list[5] ) + ',' +
        str( int_list[6] ) + ',' + str( int_list[7] ) + ',' +
        #chartype + ',' +
        string   + '\n'
    )



def drawbox( image , box  , color ) :
    image = cv2.line( image , ( box[0] , box[1] ) , ( box[2] , box[3] ) , color , 2 )
    image = cv2.line( image , ( box[2] , box[3] ) , ( box[4] , box[5] ) , color , 2 )
    image = cv2.line( image , ( box[4] , box[5] ) , ( box[6] , box[7] ) , color , 2 )
    image = cv2.line( image , ( box[0] , box[1] ) , ( box[6] , box[7] ) , color , 2 )

    return image


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def remove_last_slash(path) :
    if ( path[-1] == '/' ):
        path = path[:-1]
    return path

def make_folder( folder_name ):
    if ( os.path.isdir( folder_name ) == True ) :
        shutil.rmtree( folder_name )
    os.mkdir( folder_name )    

def rotate_pixel(  point , Angle ) :
    angle = math.pi / 180 * Angle * (-1)
    x = math.cos(angle) * point[0] + math.sin(angle) * point[1]
    y = math.sin(angle) * point[0] * (-1) + math.cos(angle) * point[1]

    return int(x) , int(y)


def box_rotation(location , angle):

    location[0] , location[1] = rotate_pixel(( location[0], location[1] ), angle)
    location[2] , location[3] = rotate_pixel(( location[2], location[3] ), angle)
    location[4] , location[5] = rotate_pixel(( location[4], location[5] ), angle)
    location[6] , location[7] = rotate_pixel(( location[6], location[7] ), angle)

    return location


def make_relative_location( center_point , location ) :
    location[0] = int(location[0] - center_point[0])
    location[1] = int(location[1] - center_point[1])
    location[2] = int(location[2] - center_point[0])
    location[3] = int(location[3] - center_point[1])
    location[4] = int(location[4] - center_point[0])
    location[5] = int(location[5] - center_point[1])
    location[6] = int(location[6] - center_point[0])
    location[7] = int(location[7] - center_point[1])
    return location



def make_abs_location( center_point , location) :
    location[0] = int(location[0] + center_point[0])
    location[1] = int(location[1] + center_point[1])
    location[2] = int(location[2] + center_point[0])
    location[3] = int(location[3] + center_point[1])
    location[4] = int(location[4] + center_point[0])
    location[5] = int(location[5] + center_point[1])
    location[6] = int(location[6] + center_point[0])
    location[7] = int(location[7] + center_point[1])
    return location



def main( args ) :

   def_ROOT_FOLDER = remove_last_slash(args.root_folder)
   make_folder(def_ROOT_FOLDER)

   def_INPUT_FOLDER  = remove_last_slash(args.input_folder)
   def_GENERATE_FILE_COUNT = args.generate_count

   def_SCALE  = args.scale


   # Check Number of file count
   NUM_OF_FILES = len(fnmatch.filter(os.listdir(def_INPUT_FOLDER), '*.json'))
   #DUMP_LOG("[0000] number of files in plate folder : " + str( NUM_OF_FILES ))
   json_file_names = glob.glob(def_INPUT_FOLDER + '/' + '*.json')

   def_TRAINING_IMAGES_FOLDER     = def_ROOT_FOLDER + '/' +  "ch4_training_images"
   def_TRAINING_GT_FOLDER         = def_ROOT_FOLDER + '/' +  "ch4_training_localization_transcription_gt"
   def_TRAIN_VOC_PER_IMAGE_FOLDER = def_ROOT_FOLDER + '/' +  "ch4_training_vocabularies_per_image"

   def_TEST_IMAGES_FOLDER         = def_ROOT_FOLDER + '/' +  "ch4_test_images"
   def_TEST_GT_FOLDER             = def_ROOT_FOLDER + '/' +  "Challenge4_Test_Task4_GT"
   def_TEST_VOC_PER_IMAGE_FOLDER  = def_ROOT_FOLDER + '/' +  "ch4_test_vocabularies_per_image"

   def_BOXED_IMG_FOLDER           = def_ROOT_FOLDER + '/' +  "boxed_img_folder"

   def_RANDOM_ANLGE_RANGE         = args.random_angle_range

   print("[2000] Random angle range : " + str(def_RANDOM_ANLGE_RANGE))

   make_folder( def_TRAINING_IMAGES_FOLDER )
   make_folder( def_TRAINING_GT_FOLDER     )
   make_folder( def_TRAIN_VOC_PER_IMAGE_FOLDER )

   make_folder( def_TEST_IMAGES_FOLDER )
   make_folder( def_TEST_GT_FOLDER     )
   make_folder( def_TEST_VOC_PER_IMAGE_FOLDER )
   
   make_folder( def_BOXED_IMG_FOLDER )


   print("[1000] Number of JSON files : " + str(NUM_OF_FILES))

   for count in tqdm ( range ( 0 , def_GENERATE_FILE_COUNT ) ):

       file_count = uniform( 0, NUM_OF_FILES -1 )
       file_count = int(file_count)

       print("[1001] file_name : " + str(json_file_names[file_count]))
       image_file_name = json_file_names[file_count].replace('json','jpg')
       print("[1002] file_name : " + str(image_file_name))

       file_name_base    =  os.path.split( image_file_name )
       image_file_name_base = file_name_base[1]     

       print("[1020] file_name_base " + str(image_file_name_base))
       src_image = cv2.imread(image_file_name, cv2.IMREAD_COLOR)

       with open(json_file_names[file_count]) as json_file:

           json_data   = json.load(json_file)
           json_shapes = json_data["shapes"]

           print("[1004] json_shapes : " + str(json_shapes))
           print("[1005] size of json_shapes : " + str(len(json_shapes)))

           min_ratio         = max( 1 - def_SCALE , 0.8 )
           horizontal_size_change_ratio = uniform(min_ratio  , 1.0 + def_SCALE)
           print("[1009] horizontal_size_change_ratio : " + str(horizontal_size_change_ratio))
           vertical_size_change_ratio = uniform(min_ratio  , 1.0 + def_SCALE)
           print("[1009] vertical_size_change_ratio : " + str(vertical_size_change_ratio))

           dest_image  = cv2.resize(src_image, dsize=(0, 0), 
                                    fx = horizontal_size_change_ratio , 
                                    fy= vertical_size_change_ratio, 
                                    interpolation=cv2.INTER_CUBIC)

           file_gen_ratio = uniform( 0  , 0.999)
           image_file_name_base_copy = image_file_name_base

           rotation_angle = uniform( def_RANDOM_ANLGE_RANGE * (-1) , def_RANDOM_ANLGE_RANGE )
           print("[2001] Rotation Angle : " + str(rotation_angle))
               

           ratio_text = "_{0:.3f}".format( rotation_angle ) + \
                        "_{0:.3f}".format( horizontal_size_change_ratio ) + \
                        "x{0:.3f}.jpg".format(vertical_size_change_ratio) 
           image_file_name_base_copy= image_file_name_base_copy.replace('.jpg' , ratio_text )

           if file_gen_ratio > args.test_ratio :
               def_IMAGE_FOLDER = def_TRAINING_IMAGES_FOLDER + '/' + image_file_name_base_copy
               def_GT_FOLDER = def_TRAINING_GT_FOLDER + '/' + 'gt_' + image_file_name_base_copy.replace('.jpg' , '.txt')
               def_VOC_PER_IMAGE_FOLDER = def_TRAIN_VOC_PER_IMAGE_FOLDER + '/' + image_file_name_base_copy.replace('.jpg' , '.txt') 
           else:
               def_IMAGE_FOLDER = def_TEST_IMAGES_FOLDER + '/' + image_file_name_base_copy
               def_GT_FOLDER = def_TEST_GT_FOLDER + '/' + 'gt_' + image_file_name_base_copy.replace('.jpg' , '.txt')
               def_VOC_PER_IMAGE_FOLDER = def_TEST_VOC_PER_IMAGE_FOLDER + '/' + image_file_name_base_copy.replace('.jpg' , '.txt')


           height, width, channel = dest_image.shape
           center_point = [width/2, height/2]

           matrix = cv2.getRotationMatrix2D((width/2, height/2), rotation_angle, 1)
           dest_image = cv2.warpAffine( dest_image , matrix, (width, height))

           cv2.imwrite( def_IMAGE_FOLDER , dest_image )

           gt_file  = open( def_GT_FOLDER ,'w')
           
           for json_shapes_count in range ( 0 , len(json_shapes)) :

               point_0 = ( json_shapes[json_shapes_count]["points"][0][0] * horizontal_size_change_ratio , 
                           json_shapes[json_shapes_count]["points"][0][1] * vertical_size_change_ratio )
               point_1 = ( json_shapes[json_shapes_count]["points"][1][0] * horizontal_size_change_ratio ,
                           json_shapes[json_shapes_count]["points"][1][1] * vertical_size_change_ratio )
               point_2 = ( json_shapes[json_shapes_count]["points"][2][0] * horizontal_size_change_ratio ,
                           json_shapes[json_shapes_count]["points"][2][1] * vertical_size_change_ratio )
               point_3 = ( json_shapes[json_shapes_count]["points"][3][0] * horizontal_size_change_ratio ,
                           json_shapes[json_shapes_count]["points"][3][1] * vertical_size_change_ratio )

               box_pos = ( point_0[0] , point_0[1] , point_1[0] , point_1[1] , point_2[0] , point_2[1] , point_3[0] , point_3[1] )
               box_pos = np.array( box_pos ).astype(int)

               box_pos_rel = make_relative_location( center_point , box_pos )
               box_pos_rel = box_rotation( box_pos , rotation_angle *(-1))
               box_pos     = make_abs_location( center_point , box_pos_rel )

               dest_image = drawbox( dest_image ,  box_pos , (255,0,255))

               #def gt_file_write( gt_file , int_list , chartype , string ):
               gt_file_write( gt_file , box_pos , 'Korea' , json_shapes[json_shapes_count]['label'])
         
           gt_file.close()

           cv2.imwrite( def_BOXED_IMG_FOLDER + '/' + image_file_name_base_copy , dest_image ) 


if __name__ == '__main__':
    print("################################################################################")
    print(" ")
    print("                             Data argumentation ")
    print(" ")
    print("################################################################################")

    print(" --input_folder : ")
    print(" --root_folder : ")
    print(" --scale : ")
    print("        0.2  : scale 0.8~1.2")
    print(" --image_bottom_limit : ")
    print("        320  : 320 미만의 크기는 더이상 터치 하지 않는다.")
    print(" --image_upper_limit  : ")
    print("        1200 : 1200 이상의 크기는 더이상 터치 하지 않는다.")
    print(" --vertial_scale    : ")
    print("        0.3  : scale 0.7~1.3")
    print(" --random_angle     : ")
    print("        10   :  -10 ~ +10")

    print(" --generate_count   : ")
    print("        생성할 파일의 수 ")
    
    parser = argparse.ArgumentParser(description='PyTorch Template')

    # NICE의 번호판 폴더
    parser.add_argument('--input_folder'       , type=str ,  help='nice plate folder')
    parser.add_argument('--root_folder'      , type=str ,  help='nice plate folder')
    parser.add_argument('--random_angle_range'  , type=int, default=0, help='Random angle')
    parser.add_argument('--generate_count'      , type=int, default=100, help='generate_count')
    parser.add_argument('--scale'               , type=float, default=0.2, help='scale change')
    parser.add_argument('--test_ratio'          , type=float, default=0.9, help='Test Image Ratio')


    args = parser.parse_args()
    main(args)


