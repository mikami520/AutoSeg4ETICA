'''
python3 split_data.py RegisteredImageFolderPath RegisteredLabelFolderPath

Given the parameter as the path to the registered images, 
function creates two folders in the base directory (same level as this script), randomly putting in
70 percent of images into the train and 30 percent to the test
'''
import os
import glob
import random
import shutil

from typing import Tuple
import numpy as np
from collections import OrderedDict
import json
import argparse


"""
creates a folder at a specified folder path if it does not exists
folder_path : relative path of the folder (from cur_dir) which needs to be created
over_write :(default: False) if True overwrite the existing folder 
 """


def parse_command_line():
    print('---'*10)
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(
        description='pipeline for dataset split')
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="Absolute path of the base directory")
    parser.add_argument('-ip', metavar='image path', type=str,
                        help="Relative path of the image directory")
    parser.add_argument('-sp', metavar='segmentation path', type=str,
                        help="Relative path of the image directory")
    parser.add_argument('-sl', metavar='segmentation information list', type=str, nargs='+',
                        help='a list of label name and corresponding value')
    parser.add_argument('-ti', metavar='task id', type=str,
                        help='task id number')
    parser.add_argument('-tn', metavar='task name', type=str,
                        help='task name')
    argv = parser.parse_args()
    return argv


def make_if_dont_exist(folder_path, overwrite=False):

    if os.path.exists(folder_path):
        if not overwrite:
            print(f'{folder_path} exists.')
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


def rename(location, oldname, newname):

    os.rename(os.path.join(location, oldname), os.path.join(location, newname))


def main():
    args = parse_command_line()
    base = args.bp
    reg_data_path = args.ip
    lab_data_path = args.sp
    task_id = args.ti
    Name = args.tn
    seg_list = args.sl
    base_dir = "/home/ameen"
    #os.chdir(base_dir)
    task_name = "Task" + task_id + "_" + Name  # MODIFY
    nnunet_dir = "nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data"
    task_folder_name = os.path.join(base_dir, nnunet_dir, task_name)
    train_image_dir = os.path.join(task_folder_name, 'imagesTr')
    train_label_dir = os.path.join(task_folder_name, 'labelsTr')
    test_dir = os.path.join(task_folder_name, 'imagesTs')
    main_dir = os.path.join(base_dir, 'nnUNet/nnunet')

    make_if_dont_exist(task_folder_name)
    make_if_dont_exist(train_image_dir)
    make_if_dont_exist(train_label_dir)
    make_if_dont_exist(test_dir)
    make_if_dont_exist(os.path.join(main_dir, 'nnUNet_preprocessed'))
    make_if_dont_exist(os.path.join(main_dir, 'nnUNet_trained_models'))

    os.environ['nnUNet_raw_data_base'] = os.path.join(
        main_dir, 'nnUNet_raw_data_base')
    os.environ['nnUNet_preprocessed'] = os.path.join(
        main_dir, 'nnUNet_preprocessed')
    os.environ['RESULTS_FOLDER'] = os.path.join(
        main_dir, 'nnUNet_trained_models')

    random.seed(0)
    cur_path = os.getcwd()  # current working

    image_list = glob.glob(os.path.join(base, reg_data_path) + "/*.nii.gz")
    label_list = glob.glob(os.path.join(base, lab_data_path) + "/*.nii.gz")
    num_images = len(image_list)
    # Dataset Split (70 / 30):
    num_train = int(round(num_images*0.7))
    num_test = int(round(num_images*0.3))
    print("Number of training subjects: ", num_train,
          "\nNumber of testing subjects:", num_test, "\nTotal:", num_images)

    random.shuffle(image_list)
    for i in range(num_images):
        filename1 = os.path.basename(image_list[i]).split(".")[0]
        number = filename1[-3:]
        if (i < num_train):
            # put this image to the training folder
            shutil.copy(image_list[i], train_image_dir)
            filename = os.path.basename(image_list[i])
            rename(train_image_dir, filename, Name + "_" + number + "_0000.nii.gz")

            for label_dir in label_list:
                if label_dir.endswith(os.path.basename(image_list[i])):
                    shutil.copy(label_dir, train_label_dir)
                    rename(train_label_dir, filename, Name + "_" + number + '.nii.gz')
                    break
        else:
            # put this image to the test folder
            shutil.copy(image_list[i], test_dir)
            filename = os.path.basename(image_list[i])
            filename1 = os.path.basename(image_list[i]).split(".")[0]
            number = filename1[-3:]
            rename(test_dir, filename, Name +  "_" + number + "_0000.nii.gz")

    # create json file
    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = Name
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "MODIFY"
    json_dict['licence'] = "MODIFY"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }
    json_dict['labels'] = {
        "0": "background",
    }
    for i in range(0, len(seg_list), 2):
        assert(seg_list[i].isdigit() == True)
        assert(seg_list[i + 1].isdigit() == False)
        json_dict['labels'].update({
            seg_list[i]: seg_list[i + 1]
        })
    train_ids = os.listdir(train_image_dir)
    test_ids = os.listdir(test_dir)
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)
    json_dict['training'] = [{'image': "./imagesTr/%s" % (i[:i.find(
        "_0000")]+'.nii.gz'), "label": "./labelsTr/%s" % (i[:i.find("_0000")]+'.nii.gz')} for i in train_ids]
    json_dict['test'] = ["./imagesTs/%s" %
                         (i[:i.find("_0000")]+'.nii.gz') for i in test_ids]

    with open(os.path.join(task_folder_name, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    if os.path.exists(os.path.join(task_folder_name, 'dataset.json')):
        print("new json file created!")


if __name__ == '__main__':
    main()
