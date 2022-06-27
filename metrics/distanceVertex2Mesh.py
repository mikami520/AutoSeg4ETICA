import numpy as np
import pyvista as pv
import argparse
import os
import glob
import trimesh


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
    argv = parser.parse_args()
    return argv


def distanceVertex2Mesh(mesh, vertex):
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    mesh_box = trimesh.Trimesh(vertices=mesh.points,
                               faces=faces_as_array)
    cp, cd, ci = trimesh.proximity.closest_point(mesh_box, vertex)
    return cd


def main():
    args = parse_command_line()
    base = args.bp
    gt_path = args.gp
    pred_path = args.pp
    output_dir = os.path.join(base, 'output')
    try:
        os.mkdir(output_dir)
    except:
        print(f'{output_dir} already exists')

    for i in glob.glob(os.path.join(base, gt_path) + '/*.vtk'):
        filename = os.path.basename(i).split('.')[0]
        #side = os.path.basename(i).split('.')[0].split('_')[0]
        #scan_name = os.path.basename(i).split('.')[0].split('_')[0]
        #scan_id = os.path.basename(i).split('.')[0].split('_')[1]
        output_sub_dir = os.path.join(
            base, 'output', filename)
        try:
            os.mkdir(output_sub_dir)
        except:
            print(f'{output_sub_dir} already exists')

        gt_mesh = pv.read(i)
        pred_mesh = pv.read(os.path.join(
            base, pred_path, filename + '.vtk'))
        pred_vertices = np.array(pred_mesh.points)
        cd = distanceVertex2Mesh(gt_mesh, pred_vertices)
        pred_mesh['dist'] = cd
        pred_mesh.save(os.path.join(output_sub_dir, filename + '.vtk'))


if __name__ == '__main__':
    main()
