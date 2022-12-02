import numpy as np
import nibabel as nib
import ants
import argparse
import pandas as pd
import glob
import os
import surface_distance
import nrrd
import shutil
import distanceVertex2Mesh
import textwrap


def parse_command_line():
    print('---'*10)
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(
        description='Inference evaluation pipeline for image registration-segmentation', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="Absolute path of the base directory")
    parser.add_argument('-gp', metavar='ground truth path', type=str,
                        help="Relative path of the ground truth segmentation directory")
    parser.add_argument('-pp', metavar='predicted path', type=str,
                        help="Relative path of predicted segmentation directory")
    parser.add_argument('-sp', metavar='save path', type=str,
                        help="Relative path of CSV file directory to save, if not specify, default is base directory")
    parser.add_argument('-vt', metavar='validation type', type=str, nargs='+',
                        help=textwrap.dedent('''Validation type:
                dsc: Dice Score
                ahd: Average Hausdorff Distance
                whd: Weighted Hausdorff Distance
                        '''))
    parser.add_argument('-pm', metavar='probability map path', type=str,
                        help="Relative path of text file directory of probability map")
    parser.add_argument('-fn', metavar='file name', type=str,
                        help="name of output file")
    parser.add_argument('-reg', action='store_true',
                        help="check if the input files are registration predictions")
    parser.add_argument('-tp', metavar='type of segmentation', type=str,
                        help=textwrap.dedent('''Segmentation type:
                ET: Eustachian Tube
                NC: Nasal Cavity
                HT: Head Tumor
                        '''))
    parser.add_argument('-sl', metavar='segmentation information list', type=str, nargs='+',
                        help='a list of label name and corresponding value')
    parser.add_argument('-cp', metavar='current prefix of filenames', type=str,
                        help='current prefix of filenames')
    argv = parser.parse_args()
    return argv


def rename(prefix, filename):
    name = filename.split('.')[0][-3:]
    name = prefix + '_' + name
    return name

def dice_coefficient_and_hausdorff_distance(filename, img_np_pred, img_np_gt, num_classes, spacing, probability_map, dsc, ahd, whd, average_DSC, average_HD):
    df = pd.DataFrame()
    data_gt, bool_gt = make_one_hot(img_np_gt, num_classes)
    data_pred, bool_pred = make_one_hot(img_np_pred, num_classes)
    for i in range(1, num_classes):
        df1 = pd.DataFrame([[filename, i]], columns=[
            'File ID', 'Label Value'])
        if dsc:
            if data_pred[i].any():
                volume_sum = data_gt[i].sum() + data_pred[i].sum()
                if volume_sum == 0:
                    return np.NaN

                volume_intersect = (data_gt[i] & data_pred[i]).sum()
                dice = 2*volume_intersect / volume_sum
                df1['Dice Score'] = dice
                average_DSC[i-1] += dice
            else:
                dice = 0.0
                df1['Dice Score'] = dice
                average_DSC[i-1] += dice
        if ahd:
            if data_pred[i].any():
                avd = average_hausdorff_distance(bool_gt[i], bool_pred[i], spacing)
                df1['Average Hausdorff Distance'] = avd
                average_HD[i-1] += avd
            else:
                avd = np.nan
                df1['Average Hausdorff Distance'] = avd
                average_HD[i-1] += avd
        if whd:
            # wgd = weighted_hausdorff_distance(gt, pred, probability_map)
            # df1['Weighted Hausdorff Distance'] = wgd
            pass

        df = pd.concat([df, df1])
    return df, average_DSC, average_HD


def make_one_hot(img_np, num_classes):
    img_one_hot_dice = np.zeros(
        (num_classes, img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.int8)
    img_one_hot_hd = np.zeros(
        (num_classes, img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=bool)
    for i in range(num_classes):
        a = (img_np == i)
        img_one_hot_dice[i, :, :, :] = a
        img_one_hot_hd[i, :, :, :] = a

    return img_one_hot_dice, img_one_hot_hd


def average_hausdorff_distance(img_np_gt, img_np_pred, spacing):
    surf_distance = surface_distance.compute_surface_distances(
        img_np_gt, img_np_pred, spacing)
    gp, pg = surface_distance.compute_average_surface_distance(surf_distance)
    return (gp + pg) / 2


def checkSegFormat(base, segmentation, type, prefix):
    path = os.path.join(base, segmentation)
    if type == 'gt':
        save_dir = os.path.join(base, 'gt_reformat_labels')
    else:
        save_dir = os.path.join(base, 'pred_reformat_labels')
    try:
        os.mkdir(save_dir)
    except:
        print(f'{save_dir} already exists')

    for file in os.listdir(path):
        if type == 'gt':
            name = rename(prefix, file)
        else:
            name = file.split('.')[0]

        if file.endswith('seg.nrrd'):
            ants_img = ants.image_read(os.path.join(path, file))
            header = nrrd.read_header(os.path.join(path, file))
            filename = os.path.join(save_dir, name + '.nii.gz')
            nrrd2nifti(ants_img, header, filename)
        elif file.endswith('nii'):
            image = ants.image_read(os.path.join(path, file))
            image.to_file(os.path.join(save_dir, name + '.nii.gz'))
        elif file.endswith('nii.gz'):
            shutil.copy(os.path.join(path, file), os.path.join(save_dir, name + '.nii.gz'))

    return save_dir


def nrrd2nifti(img, header, filename):
    img_as_np = img.view(single_components=True)
    data = convert_to_one_hot(img_as_np, header)
    foreground = np.max(data, axis=0)
    labelmap = np.multiply(np.argmax(data, axis=0) + 1,
                           foreground).astype('uint8')
    segmentation_img = ants.from_numpy(
        labelmap, origin=img.origin, spacing=img.spacing, direction=img.direction)
    print('-- Saving NII Segmentations')
    segmentation_img.to_file(filename)


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


def get_layer_values(header):
    layer_values = []
    num_segments = len([key for key in header.keys() if "Layer" in key])
    for i in range(num_segments):
        layer_values.append(int(header['Segment{}_Layer'.format(i)]))
    return layer_values


def get_label_values(header):
    label_values = []
    num_segments = len([key for key in header.keys() if "LabelValue" in key])
    for i in range(num_segments):
        label_values.append(int(header['Segment{}_LabelValue'.format(i)]))
    return label_values


def main():
    args = parse_command_line()
    base = args.bp
    gt_path = args.gp
    pred_path = args.pp
    if args.sp is None:
        save_path = base
    else:
        save_path = args.sp
    validation_type = args.vt
    probability_map_path = args.pm
    filename = args.fn
    reg = args.reg
    seg_type = args.tp
    label_list = args.sl
    current_prefix = args.cp
    if probability_map_path is not None:
        probability_map = np.loadtxt(os.path.join(base, probability_map_path))
    else:
        probability_map = None
    dsc = False
    ahd = False
    whd = False
    for i in range(len(validation_type)):
        if validation_type[i] == 'dsc':
            dsc = True
        elif validation_type[i] == 'ahd':
            ahd = True
        elif validation_type[i] == 'whd':
            whd = True
        else:
            print('wrong validation type, please choose correct one !!!')
            return

    filepath = os.path.join(base, save_path, 'output_' + filename + '.csv')
    save_dir = os.path.join(base, save_path)
    gt_output_path = checkSegFormat(base, gt_path, 'gt', current_prefix)
    pred_output_path = checkSegFormat(base, pred_path, 'pred', current_prefix)
    try:
        os.mkdir(save_dir)
    except:
        print(f'{save_dir} already exists')
    
    try:
        os.mknod(filepath)
    except:
        print(f'{filepath} already exists')

    DSC = pd.DataFrame()
    file = glob.glob(os.path.join(base, gt_output_path) + '/*nii.gz')[0]
    seg_file = ants.image_read(file)
    num_class = np.unique(seg_file.numpy().ravel()).shape[0]
    average_DSC = np.zeros((num_class-1))
    average_HD = np.zeros((num_class-1))
    k = 0
    for i in glob.glob(os.path.join(base, pred_output_path) + '/*nii.gz'):
        k += 1
        pred_img = ants.image_read(i)
        pred_spacing = list(pred_img.spacing)
        if reg and seg_type == 'ET':
            file_name = os.path.basename(i).split('.')[0].split('_')[4] + '_' + os.path.basename(
                i).split('.')[0].split('_')[5] + '_' + os.path.basename(i).split('.')[0].split('_')[6]
            file_name1 = os.path.basename(i).split('.')[0]
        elif reg and seg_type == 'NC':
            file_name = os.path.basename(i).split(
                '.')[0].split('_')[3] + '_' + os.path.basename(i).split('.')[0].split('_')[4]
            file_name1 = os.path.basename(i).split('.')[0]
        elif reg and seg_type == 'HT':
            file_name = os.path.basename(i).split('.')[0].split('_')[2]
            file_name1 = os.path.basename(i).split('.')[0]
        else:
            file_name = os.path.basename(i).split('.')[0]
            file_name1 = os.path.basename(i).split('.')[0]
        gt_seg = os.path.join(base, gt_output_path, file_name + '.nii.gz')
        gt_img = ants.image_read(gt_seg)
        gt_spacing = list(gt_img.spacing)

        if gt_spacing != pred_spacing:
            print(
                "Spacing of prediction and ground_truth is not matched, please check again !!!")
            return

        ref = pred_img
        data_ref = ref.numpy()

        pred = gt_img
        data_pred = pred.numpy()

        num_class = len(np.unique(data_pred))
        ds, aver_DSC, aver_HD = dice_coefficient_and_hausdorff_distance(
            file_name1, data_ref, data_pred, num_class, pred_spacing, probability_map, dsc, ahd, whd, average_DSC, average_HD)
        DSC = pd.concat([DSC, ds])
        average_DSC = aver_DSC
        average_HD = aver_HD

    avg_DSC = average_DSC / k
    avg_HD = average_HD / k
    print(avg_DSC)
    with open(os.path.join(base, save_path, "metric.txt"), 'w') as f:
        f.write("Label Value  Label Name  Average Dice Score  Average Mean HD\n")
        for i in range(len(avg_DSC)):
            f.write(f'{str(i+1):^12}{str(label_list[2*i+1]):^12}{str(avg_DSC[i]):^20}{str(avg_HD[i]):^18}\n')
    DSC.to_csv(filepath)


if __name__ == '__main__':
    main()
