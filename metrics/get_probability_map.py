import numpy as np
import pyvista as pv
import argparse
import os
import glob
import skeletor as sk
import trimesh
import navis


def parse_command_line():
    print('---'*10)
    print('Parsing Command Line Arguments')
    parser = argparse.ArgumentParser(description='Defacing protocol')
    parser.add_argument('-bp', metavar='base path', type=str,
                        help="Absolute path of the base directory")
    parser.add_argument('-gp', metavar='ground truth path', type=str,
                        help="Relative path of the ground truth model")
    parser.add_argument('-pp', metavar='prediction path', type=str,
                        help="Relative path of the prediction model")
    parser.add_argument('-rr', metavar='ratio to split skeleton', type=int, nargs='+',
                        help="Ratio to split the skeleton")
    parser.add_argument('-ps', metavar='probability sequences', type=float, nargs='+',
                        help="Proability sequences for each splitted region")
    argv = parser.parse_args()
    return argv


def distanceVertex2Path(mesh, skeleton, probability_map):
    if len(probability_map) == 0:
        print('empty probability_map !!!')
        return np.inf

    if not mesh.is_all_triangles():
        print('only triangulations is allowed (Faces do not have 3 Vertices)!')
        return np.inf

    if hasattr(mesh, 'points'):
        points = np.array(mesh.points)
    else:
        print('mesh structure must contain fields ''vertices'' and ''faces''!')
        return np.inf

    if hasattr(skeleton, 'vertices'):
        vertex = skeleton.vertices
    else:
        print('skeleton structure must contain fields ''vertices'' !!!')
        return np.inf

    numV, dim = points.shape
    numT, dimT = vertex.shape

    if dim != dimT or dim != 3:
        print('mesh and vertices must be in 3D space!')
        return np.inf

    d_min = np.matrix(np.ones((numV, 1)) * np.inf)
    min_idx = np.matrix(np.zeros((numV, 1)))
    pm = []
    # first check: find closest distance from vertex to vertex
    for i in range(numV):
        for j in range(numT):
            v1 = points[i, :]
            v2 = vertex[j, :]
            d = distance3DV2V(v1, v2)
            if d < d_min[i]:
                d_min[i] = d
                min_idx[i] = j

        pm.append(probability_map[min_idx[i]])

    print("check is finished !!!")
    return pm


def generate_probability_map(skeleton, split_ratio, probability):
    points = skeleton.vertices
    center = skeleton.skeleton.centroid
    x = sorted(points[:, 0])
    left = []
    right = []
    for i in range(len(x)):
        if x[i] < center[0]:
            left.append(x[i])
        else:
            right.append(x[i])

    right_map = []
    left_map = []
    sec_old = 0
    for j in range(len(split_ratio)):
        if j == len(split_ratio) - 1:
            sec_len = len(left) - sec_old
        else:
            sec_len = int(round(len(left) * split_ratio[j] / 100))

        for k in range(sec_old, sec_old + sec_len):
            left_map.append(probability[j])

        sec_old += sec_len

    sec_old = 0
    for j in range(len(split_ratio)-1, -1, -1):
        if j == 0:
            sec_len = len(right) - sec_old
        else:
            sec_len = int(round(len(right) * split_ratio[j] / 100))

        for k in range(sec_old, sec_old + sec_len):
            right_map.append(probability[j])

        sec_old += sec_len

    final_map = []
    row = points.shape[0]
    assert len(left) + len(right) == row
    for m in range(row):
        ver_x = points[m, 0]
        if ver_x in left:
            index = left.index(ver_x)
            final_map.append(left_map[index])
        else:
            index = right.index(ver_x)
            final_map.append(right_map[index])

    return final_map


def skeleton(mesh):
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    trmesh = trimesh.Trimesh(mesh.points, faces_as_array)
    fixed = sk.pre.fix_mesh(trmesh, remove_disconnected=5, inplace=False)
    skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
    # Create a neuron from your skeleton
    n = navis.TreeNeuron(skel, soma=None)
    # keep only the two longest linear section in your skeleton
    long2 = navis.longest_neurite(n, n=2, from_root=False)

    # This renumbers nodes
    swc = navis.io.swc_io.make_swc_table(long2)
    # We also need to rename some columns
    swc = swc.rename({'PointNo': 'node_id', 'Parent': 'parent_id', 'X': 'x',
                      'Y': 'y', 'Z': 'z', 'Radius': 'radius'}, axis=1).drop('Label', axis=1)
    # Skeletor excepts node IDs to start with 0, but navis starts at 1 for SWC
    swc['node_id'] -= 1
    swc.loc[swc.parent_id > 0, 'parent_id'] -= 1
    # Create the skeletor.Skeleton
    skel2 = sk.Skeleton(swc)
    return skel2


def distance3DV2V(v1, v2):
    d = np.linalg.norm(v1-v2)
    return d


def main():
    args = parse_command_line()
    base = args.bp
    gt_path = args.gp
    pred_path = args.pp
    area_ratio = args.rr
    prob_sequences = args.ps
    output_dir = os.path.join(base, 'output')
    try:
        os.mkdir(output_dir)
    except:
        print(f'{output_dir} already exists')

    for i in glob.glob(os.path.join(base, gt_path) + '/*.vtk'):
        scan_name = os.path.basename(i).split('.')[0].split('_')[1]
        scan_id = os.path.basename(i).split('.')[0].split('_')[2]
        output_sub_dir = os.path.join(
            base, 'output', scan_name + '_' + scan_id)
        try:
            os.mkdir(output_sub_dir)
        except:
            print(f'{output_sub_dir} already exists')

        gt_mesh = pv.read(i)
        pred_mesh = pv.read(os.path.join(
            base, pred_path, 'pred_' + scan_name + '_' + scan_id + '.vtk'))
        pred_skel = skeleton(pred_mesh)
        prob_map = generate_probability_map(
            pred_skel, area_ratio, prob_sequences)
        pm = distanceVertex2Path(pred_mesh, pred_skel, prob_map)
        if(pm == np.Inf):
            print('something with mesh, probability map and skeleton are wrong !!!')
            return
        np.savetxt(os.path.join(base, output_sub_dir, scan_id + '.txt'), pm)


if __name__ == '__main__':
    main()
