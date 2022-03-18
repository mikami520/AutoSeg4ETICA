import numpy as np
import nibabel as nib
import os
from glob import glob
import argparse


def parse_command_line():
    print('---'*10)
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(
        description='Defacing protocol')
    parser.add_argument('-sc', metavar='Scans', type=str,
                        help="An integer belonging to the scan ids you wish to choose as template")
    parser.add_argument('-mk', metavar='Masks', type=str,
                        help="An integer belonging to the scan ids you wish to choose as template segmentation id")
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="Absolute path of the base directory")
    argv = parser.parse_args()
    return argv


def deface(input_file, mask_file, output_file=None, suffix=" (masked)", write=True):
    # Load the original CT volume
    input = nib.load(input_file)

    # Load the segmentation mask
    segmentation = nib.load(mask_file)

    input_array = input.get_fdata()
    segmentation_array = segmentation.get_fdata()
    mask = 1-segmentation_array  # 0's inside the mask, 1's outside

    # Create the masked CT volume
    output_array = input_array * mask
    output = nib.Nifti1Image(output_array, input.affine, input.header)

    # Save the masked CT volume
    if write:
        if output_file is None:  # Save in same folder but with suffix
            output_file = input_file.split(".")[0] + suffix + ".nii.gz"
            output.to_filename(output_file)
        else:
            # Otherwise, save to specified path
            output.to_filename(output_file)
    return output


def main():
    args = parse_command_line()
    base = args.bp
    images = args.sc
    masks = args.mk

    CT_images = sorted(glob(os.path.join(base, images, '*.nii.gz')))
    mask_images = sorted(glob(os.path.join(base, masks, '*.nii.gz')))

    num = len(CT_images)
    print(num)

    for i in range(num):
        deface(CT_images[i], mask_images[i], output_file=None,
               suffix=' (masked)', write=True)


if __name__ == '__main__':
    main()
