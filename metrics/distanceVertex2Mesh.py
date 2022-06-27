import numpy as np
import pyvista as pv
import argparse
import os
import glob


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
    if not mesh.is_all_triangles():
        print('only triangulations is allowed (Faces do not have 3 Vertices)!')
        return np.Inf

    if hasattr(mesh, 'faces') and hasattr(mesh, 'points'):
        points = np.array(mesh.points)
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    else:
        print('mesh structure must contain fields ''vertices'' and ''faces''!')

    numV, dim = points.shape
    numF, numVF = faces.shape
    numT, dimT = vertex.shape

    if dim != dimT or dim != 3:
        print('mesh and vertices must be in 3D space!')
        return np.Inf

    d_min = np.matrix(np.ones((numT, 1)) * np.inf)

    # first check: find closest distance from vertex to vertex
    for i in range(numT):
        for j in range(numV):
            v1 = vertex[i, :]
            v2 = points[j, :]
            d = distance3DV2V(v1, v2)
            if d < d_min[i]:
                d_min[i] = d

    print("first check finished !!!")

    # Second check: find closest distance from vertex to edge
    for i in range(numT):
        for j in range(numF):
            for k in range(2):
                for m in range(k + 1, 3):
                    v1 = vertex[i, :]
                    v2 = points[faces[j, k], :]
                    v3 = points[faces[j, m], :]
                    if (np.minimum(v2, v3) < v1 + d_min[i]).all() and (np.maximum(v2, v3) > v1 - d_min[i]).all():
                        d = distance3DV2E(v1, v2, v3)
                        if d < d_min[i]:
                            d_min[i] = d

    print("second check finished !!!")
    # Third check: find closest distance from vertex to face
    for i in range(numT):
        for j in range(numF):
            v1 = vertex[i, :]
            v2 = points[faces[j, 0], :]
            v3 = points[faces[j, 1], :]
            v4 = points[faces[j, 2], :]
            v5 = np.vstack((v2, v3, v4))
            if (np.min(v5, axis=0) < v1 + d_min[i]).all() and (np.max(v5, axis=0) > v1 - d_min[i]).all():
                d = distance3DV2F(v1, v2, v3, v4)
                if d < d_min[i]:
                    d_min[i] = d

    print("third check finished !!!")
    return d_min


def distance3DV2V(v1, v2):
    d = np.linalg.norm(v1-v2)
    return d


def distance3DV2E(v1, v2, v3):
    cross_V = np.cross(v3-v2, v2-v1)
    norm_V = np.linalg.norm(cross_V)
    norm_util = np.linalg.norm(v3-v2)
    d = norm_V / norm_util
    s = (-np.dot((v2-v1), (v3-v2))) / (norm_util ** 2)
    if s >= 0 and s <= 1:
        dist = d
    else:
        dist = np.inf

    return dist


def distance3DV2F(v1, v2, v3, v4):
    cross_V = np.cross(v4-v2, v3-v2)
    norm_V = np.linalg.norm(cross_V)
    n = cross_V / norm_V
    d = abs(np.dot(n, v1-v2))
    f1 = v1 + d * n
    f2 = v1 - d * n
    m = np.vstack((v3-v2, v4-v2)).T
    try:
        r1 = np.linalg.lstsq(m, (f1-v2).T)[0]
    except:
        r1 = np.array((np.inf, np.inf))

    try:
        r2 = np.linalg.lstsq(m, (f2-v2).T)[0]
    except:
        r2 = np.array((np.inf, np.inf))

    sum_r1 = np.sum(r1)
    sum_r2 = np.sum(r2)
    if (sum_r1 <= 1 and sum_r1 >= 0 and (r1 >= 0).all() and (r1 <= 1).all()) or (sum_r2 <= 1 and sum_r2 >= 0 and (r2 >= 0).all() and (r2 <= 1).all()):
        dist = d
    else:
        dist = np.inf

    return dist


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
        d = distanceVertex2Mesh(gt_mesh, pred_vertices)
        if(d.any() == np.Inf):
            print('something with mesh is wrong !!!')
            return
        np.savetxt(os.path.join(base, output_sub_dir, filename + '.txt'), d)


if __name__ == '__main__':
    main()
