import numpy as np
import glob
import ants
import nibabel as nib
import os
import argparse
import sys
from pathlib import Path

def parse_command_line():
    parser = argparse.ArgumentParser(
        description='pipeline for data preprocessing')
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="absolute path of the base directory")
    parser.add_argument('-ip', metavar='image path', type=str,
                        help="relative path of the image directory")
    parser.add_argument('-sp', metavar='segmentation path', type=str,
                        help="relative path of the image directory")
    parser.add_argument('-op', metavar='preprocessing result output path', type=str, default='output',
                        help='relative path of the preprocessing result directory')
    argv = parser.parse_args()
    return argv

def flip(nib_img, nib_seg, ants_img, ants_seg, seg_fomat):
    img = nib_img.get_fdata()
    if seg_fomat == 'nii.gz' or seg_fomat == 'nii':
        seg = nib_seg.get_fdata()
    else:
        seg = nib_seg[0]
    gem = ants.label_geometry_measures(ants_seg, ants_img)
    low_x = min(list(gem.loc[:, 'BoundingBoxLower_x']))
    upp_x = max(list(gem.loc[:, 'BoundingBoxUpper_x']))
    low_y = min(list(gem.loc[:, 'BoundingBoxLower_y']))
    upp_y = max(list(gem.loc[:, 'BoundingBoxUpper_y']))
    low_z = min(list(gem.loc[:, 'BoundingBoxLower_z']))
    upp_z = max(list(gem.loc[:, 'BoundingBoxUpper_z']))
    # Compute mid point
    mid_x = int((low_x + upp_x) / 2)

    left_seg = seg[:mid_x, :, :]
    left_img = img[:mid_x, :, :]
    right_seg = seg[mid_x:, :, :]
    right_img = img[mid_x:, :, :]
    flipped_right_seg = np.flip(right_seg, axis=0)
    flipped_right_img = np.flip(right_img, axis=0)
    print("finish flip")
    return left_img, left_seg, flipped_right_img, flipped_right_seg

def load_data(img_path, seg_path):
    nib_seg = nib.load(seg_path)
    nib_img = nib.load(img_path)
    ants_seg = ants.image_read(seg_path)
    ants_img = ants.image_read(img_path)
    return nib_img, nib_seg, ants_img, ants_seg


def crop_flip_save_file(left_img, left_seg, flipped_right_img, flipped_right_seg, nib_img, nib_seg, output_img, output_seg, scan_id):
    left_img_nii = nib.Nifti1Image(
        left_img, affine=nib_img.affine, header=nib_img.header)
    left_seg_nii = nib.Nifti1Image(
        left_seg, affine=nib_seg.affine, header=nib_seg.header)
    right_img_nii = nib.Nifti1Image(
        flipped_right_img, affine=nib_img.affine, header=nib_img.header)
    right_seg_nii = nib.Nifti1Image(
        flipped_right_seg, affine=nib_seg.affine, header=nib_seg.header)
    left_img_nii.to_filename(os.path.join(
        output_img, scan_id + '1.nii.gz'))
    left_seg_nii.to_filename(os.path.join(
        output_seg, scan_id + '1.nii.gz'))
    right_img_nii.to_filename(os.path.join(
        output_img, scan_id + '0.nii.gz'))
    right_seg_nii.to_filename(os.path.join(
        output_seg, scan_id + '0.nii.gz'))


def main():
    args = parse_command_line()
    base_path = args.bp
    image_path = os.path.join(base_path, args.ip)
    seg_path = os.path.join(base_path, args.sp)
    output_path = os.path.join(base_path, args.op)
    output_img = os.path.join(output_path, 'images')
    output_seg = os.path.join(output_path, 'labels')
    try:
        os.mkdir(output_path)
    except:
        print(f'{output_path} is already existed')

    try:
        os.mkdir(output_img)
    except:
        print(f'{output_img} is already existed')

    try:
        os.mkdir(output_seg)
    except:
        print(f'{output_seg} is already existed')

    for i in sorted(glob.glob(image_path + '/*nii.gz')):
        id = os.path.basename(i).split('.')[0]
        label_path = os.path.join(seg_path, id + '.nii.gz')
        nib_img, nib_seg, ants_img, ants_seg = load_data(i, label_path)
        left_img, left_seg, flipped_right_img, flipped_right_seg = flip(nib_img, nib_seg, ants_img, ants_seg, 'nii.gz')
        print('Scan ID: ' + id + f', img & seg before cropping: {nib_img.get_fdata().shape}, after flipping: {left_img.shape} and {flipped_right_img.shape}')
        crop_flip_save_file(left_img, left_seg, flipped_right_img, flipped_right_seg, nib_img, nib_seg, output_img, output_seg, id)

if __name__ == '__main__':
    main()