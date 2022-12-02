import os
import ants
import nrrd
import numpy as np
import glob
import slicerio
import shutil
import argparse


def parse_command_line():
    print('---'*10)
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(
        description='pipeline for dataset nnUNet preprocessing')
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="Absolute path of the base directory")
    parser.add_argument('-ip', metavar='image path', type=str,
                        help="Relative path of the image directory")
    parser.add_argument('-sp', metavar='segmentation path', type=str,
                        help="Relative path of the image directory")
    parser.add_argument('-sl', metavar='segmentation information list', type=str, nargs='+',
                        help='a list of label name and corresponding value')
    argv = parser.parse_args()
    return argv


def split_and_registration(template, target, base, images_path, seg_path, fomat, checked=False, has_label=False):
    print('---'*10)
    print('Creating file paths')
    # Define the path for template, target, and segmentations (from template)
    fixed_path = os.path.join(base, images_path, template + '.' + fomat)
    moving_path = os.path.join(base, images_path, target + '.' + fomat)
    images_output = os.path.join(base, 'imagesRS/', target + '.nii.gz')
    print('---'*10)
    print('Reading in the template and target image')
    # Read the template and target image
    template_image = ants.image_read(fixed_path)
    target_image = ants.image_read(moving_path)
    print('---'*10)
    print('Performing the template and target image registration')
    transform_forward = ants.registration(fixed=template_image, moving=target_image,
                                          type_of_transform="Similarity", verbose=False)
    if has_label:
        segmentation_path = os.path.join(
            base, seg_path, target + '.nii.gz')
        segmentation_output = os.path.join(
            base, 'labelsRS/', target + '.nii.gz')
        print('---'*10)
        print('Reading in the segmentation')
        # Split segmentations into individual components
        segment_target = ants.image_read(segmentation_path)
        print('---'*10)
        print('Applying the transformation for label propagation and image registration')
        predicted_targets_image = ants.apply_transforms(
            fixed=template_image,
            moving=segment_target,
            transformlist=transform_forward["fwdtransforms"],
            interpolator="genericLabel",
            verbose=False)
        predicted_targets_image.to_file(segmentation_output)

    reg_img = ants.apply_transforms(
        fixed=template_image,
        moving=target_image,
        transformlist=transform_forward["fwdtransforms"],
        interpolator="linear",
        verbose=False)
    print('---'*10)
    print("writing out transformed template segmentation")
    reg_img.to_file(images_output)
    print('Label Propagation & Image Registration complete')


def convert_to_one_hot(data, header, segment_indices=None):
    print('---'*10)
    print("converting to one hot")

    layer_values = get_layer_values(header)
    label_values = get_label_values(header)

    # Newer Slicer NRRD (compressed layers)
    if layer_values and label_values:

        assert len(layer_values) == len(label_values)
        if len(data.shape) == 3:
            x_dim, y_dim, z_dim = data.shape
        elif len(data.shape) == 4:
            x_dim, y_dim, z_dim = data.shape[1:]

        num_segments = len(layer_values)
        one_hot = np.zeros((num_segments, x_dim, y_dim, z_dim))

        if segment_indices is None:
            segment_indices = list(range(num_segments))

        elif isinstance(segment_indices, int):
            segment_indices = [segment_indices]

        elif not isinstance(segment_indices, list):
            print("incorrectly specified segment indices")
            return

        # Check if NRRD is composed of one layer 0
        if np.max(layer_values) == 0:
            for i, seg_idx in enumerate(segment_indices):
                layer = layer_values[seg_idx]
                label = label_values[seg_idx]
                one_hot[i] = 1*(data == label).astype(np.uint8)

        else:
            for i, seg_idx in enumerate(segment_indices):
                layer = layer_values[seg_idx]
                label = label_values[seg_idx]
                one_hot[i] = 1*(data[layer] == label).astype(np.uint8)

    # Binary labelmap
    elif len(data.shape) == 3:
        x_dim, y_dim, z_dim = data.shape
        num_segments = np.max(data)
        one_hot = np.zeros((num_segments, x_dim, y_dim, z_dim))

        if segment_indices is None:
            segment_indices = list(range(1, num_segments + 1))

        elif isinstance(segment_indices, int):
            segment_indices = [segment_indices]

        elif not isinstance(segment_indices, list):
            print("incorrectly specified segment indices")
            return

        for i, seg_idx in enumerate(segment_indices):
            one_hot[i] = 1*(data == seg_idx).astype(np.uint8)

    # Older Slicer NRRD (already one-hot)
    else:
        return data

    return one_hot


def get_layer_values(header, indices=None):
    layer_values = []
    num_segments = len([key for key in header.keys() if "Layer" in key])
    for i in range(num_segments):
        layer_values.append(int(header['Segment{}_Layer'.format(i)]))
    return layer_values


def get_label_values(header, indices=None):
    label_values = []
    num_segments = len([key for key in header.keys() if "LabelValue" in key])
    for i in range(num_segments):
        label_values.append(int(header['Segment{}_LabelValue'.format(i)]))
    return label_values


def get_num_segments(header, indices=None):
    num_segments = len([key for key in header.keys() if "LabelValue" in key])
    return num_segments


def checkCorrespondence(segmentation, base, paired_list, filename):
    print(filename)
    assert type(paired_list) == list
    data, tempSeg = nrrd.read(os.path.join(base, segmentation, filename))
    seg_info = slicerio.read_segmentation_info(
        os.path.join(base, segmentation, filename))
    output_voxels, output_header = slicerio.extract_segments(
        data, tempSeg, seg_info, paired_list)
    output = os.path.join(base, 'MatchedSegs/' +
                          filename)
    nrrd.write(output, output_voxels, output_header)
    print('---'*10)
    print('Check the label names and values')
    print(slicerio.read_segmentation_info(output))
    return output


def checkSegFormat(base, segmentation, paired_list, check=False):
    path = os.path.join(base, segmentation)
    save_dir = os.path.join(base, 're-format_labels')
    try:
        os.mkdir(save_dir)
    except:
        print(f'{save_dir} already exists')

    for file in os.listdir(path):
        name = file.split('.')[0]
        if file.endswith('seg.nrrd') or file.endswith('nrrd'):
            if check:
                output_path = checkCorrespondence(
                    segmentation, base, paired_list, file)
                ants_img = ants.image_read(output_path)
                header = nrrd.read_header(output_path)
            else:
                ants_img = ants.image_read(os.path.join(path, file))
                header = nrrd.read_header(os.path.join(path, file))
            segmentations = True
            filename = os.path.join(save_dir, name + '.nii.gz')
            nrrd2nifti(ants_img, header, filename, segmentations)
        elif file.endswith('nii'):
            image = ants.image_read(os.path.join(path, file))
            image.to_file(os.path.join(save_dir, name + '.nii.gz'))
        elif file.endswith('nii.gz'):
            shutil.copy(os.path.join(path, file), save_dir)

    return save_dir


def nrrd2nifti(img, header, filename, segmentations=True):
    img_as_np = img.view(single_components=segmentations)
    if segmentations:
        data = convert_to_one_hot(img_as_np, header)
        foreground = np.max(data, axis=0)
        labelmap = np.multiply(np.argmax(data, axis=0) + 1,
                               foreground).astype('uint8')
        segmentation_img = ants.from_numpy(
            labelmap, origin=img.origin, spacing=img.spacing, direction=img.direction)
        print('-- Saving NII Segmentations')
        segmentation_img.to_file(filename)
    else:
        print('-- Saving NII Volume')
        img.to_file(filename)


def find_template(base, image_path, fomat):
    scans = sorted(glob.glob(os.path.join(base, image_path) + '/*' + fomat))
    template = os.path.basename(scans[0]).split('.')[0]
    return template


def find_template_V2(base, image_path, fomat):
    maxD = -np.inf
    for i in glob.glob(os.path.join(base, image_path) + '/*' + fomat):
        id = os.path.basename(i).split('.')[0]
        img = ants.image_read(i)
        thirdD = img.shape[2]
        if thirdD > maxD:
            template = id
            maxD = thirdD

    return template


def path_to_id(path, fomat):
    ids = []
    for i in glob.glob(path + '/*' + fomat):
        id = os.path.basename(i).split('.')[0]
        ids.append(id)
    return ids


def checkFormat(base, images_path):
    path = os.path.join(base, images_path)
    for file in os.listdir(path):
        if file.endswith('.nii'):
            ret = 'nii'
            break
        elif file.endswith('.nii.gz'):
            ret = 'nii.gz'
            break
        elif file.endswith('.nrrd'):
            ret = 'nrrd'
            break
        elif file.endswith('.seg.nrrd'):
            ret = 'seg.nrrd'
            break
    return ret


def main():
    args = parse_command_line()
    base = args.bp
    images_path = args.ip
    segmentation = args.sp
    label_list = args.sl
    images_output = os.path.join(base, 'imagesRS')
    labels_output = os.path.join(base, 'labelsRS')
    fomat = checkFormat(base, images_path)
    fomat_seg = checkFormat(base, segmentation)
    template = find_template_V2(base, images_path, fomat)
    label_lists = path_to_id(os.path.join(base, segmentation), fomat_seg)
    if label_list is not None:
        matched_output = os.path.join(base, 'MatchedSegs')
        try:
            os.mkdir(matched_output)
        except:
            print(f"{matched_output} already exists")

    try:
        os.mkdir(images_output)
    except:
        print(f"{images_output} already exists")

    try:
        os.mkdir(labels_output)
    except:
        print(f"{labels_output} already exists")

    paired_list = []
    if label_list is not None:
        for i in range(0, len(label_list), 2):
            if not label_list[i].isdigit():
                print(
                    "Wrong order of input argument for pair-wising label value and its name !!!")
                return
            else:
                value = label_list[i]
                if not label_list[i+1].isdigit():
                    key = label_list[i+1]
                    ele = tuple((key, value))
                    paired_list.append(ele)
                else:
                    print(
                        "Wrong input argument for pair-wising label value and its name !!!")
                    return

            # print(new_segmentation)
        seg_output_path = checkSegFormat(
            base, segmentation, paired_list, check=True)
        for j in sorted(glob.glob(os.path.join(base, images_path) + '/*' + fomat)):
            id = os.path.basename(j).split('.')[0]
            if id == template:
                pass
            else:
                target = id
                if id in label_lists:
                    split_and_registration(
                        template, target, base, images_path, seg_output_path, fomat, checked=True, has_label=True)
                else:
                    split_and_registration(
                        template, target, base, images_path, seg_output_path, fomat, checked=True, has_label=False)

        image = ants.image_read(os.path.join(
            base, images_path, template + '.' + fomat))
        image.to_file(os.path.join(base, images_output, template + '.nii.gz'))
        fomat = 'nii.gz'
        images_path = os.path.join(base, 'imagesRS/')
        if template in label_lists:
            split_and_registration(
                target, template, base, images_path, seg_output_path, fomat, checked=True, has_label=True)
        else:
            split_and_registration(
                target, template, base, images_path, seg_output_path, fomat, checked=True, has_label=False)

    else:
        seg_output_path = checkSegFormat(
            base, segmentation, paired_list, check=False)

        for i in sorted(glob.glob(os.path.join(base, images_path) + '/*' + fomat)):
            id = os.path.basename(i).split('.')[0]
            if id == template:
                pass
            else:
                target = id
                if id in label_lists:
                    split_and_registration(
                        template, target, base, images_path, seg_output_path, fomat, checked=False, has_label=True)
                else:
                    split_and_registration(
                        template, target, base, images_path, seg_output_path, fomat, checked=False, has_label=False)

        image = ants.image_read(os.path.join(
            base, images_path, template + '.' + fomat))
        image.to_file(os.path.join(base, images_output, template + '.nii.gz'))

        images_path = os.path.join(base, 'imagesRS/')
        fomat = 'nii.gz'
        if template in label_lists:
            split_and_registration(
                target, template, base, images_path, seg_output_path, fomat, checked=True, has_label=True)
        else:
            split_and_registration(
                target, template, base, images_path, seg_output_path, fomat, checked=True, has_label=False)


if __name__ == '__main__':
    main()
