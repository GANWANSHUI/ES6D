import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../..')))
import argparse
from tqdm import tqdm
import numpy as np
import math
import torch
import json
import open3d as o3d
from lib.meanshift import mean_shift as ms
from models.pointnet_util import Rmatrix_from_rotate_axis, knn_one_point
from tools.grouped_primitives_tools.rotation_space_sampler import symmetric_axis_angle_sampler
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tless', help='only tless is supported!')
parser.add_argument('--pcd_dir', type=str, default='./datasets/tless/3d_models', help='the directory of 3d models')
parser.add_argument('--out_dir', type=str, default='./tools/grouped_primitives_tools/tless', help='the directory of results')
opt = parser.parse_args()

box3d = np.asarray(
        [[-1, 1, 1],
         [-1, 1, -1],
         [-1, -1, 1],
         [-1, -1, -1],
         [1, 1, 1],
         [1, 1, -1],
         [1, -1, 1],
         [1, -1, -1]])

color_map = [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.],
             [0.7, 0.3, 0.5],
             [0., 1., 1.],
             [1., 0., 1.],
             [0.8, 0.4, 1.]]


def main():
    aa_num = 30000  # number of axis-angle samples
    aa_min_angle = 30  # minimum angle of interest
    auxiliary_axis_scale = 0.8  # auxiliary axis is used in construction of the category 2 of symmetric object
    batch_sz = 50

    trim_gp = True  # trim grouped primitives of asymmetric objects
    scale_gp_by_shape = True  # scale raw grouped primitives by object shape
    normalize_gp = True  # normalize grouped primitives
    pre_name = 'tsn'

    dataset = opt.dataset
    pcd_dir = opt.pcd_dir
    out_dir = opt.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    ball_ids, symmetry_ids, errs = [], [], []

    if dataset == 'tless':
        # tless config
        ball_ids = []  # the class indexes of spherical objects
        symmetry_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28,
                        29]  # the class indexes of other symmetric objects, start with 0

        errs = [.015, .015, .015, .015, .015, .015, .015, 0.02, .015, .015, 0.02, 0.04, .015, .015, .015, .015, .015,
                .015, .015, .015, .015, .015, 0.02, .015, .015, 0.02, .015]
    elif dataset == 'ycb':
        print("not yet support!")
        return
        # ycb-video config
        ball_ids = []  # the class indexes of Spherical objects
        symmetry_ids = [8, 11, 26, 27, 28]  # start with 0
        errs = [0.03, 0.04, 0.03, 0.03, 0.03]
    else:
        print("not yet support!")
        return

    print("------------------------------------------------------------------------------------")
    print("dataset: {}".format(dataset))
    print("pcd_dir: {}".format(pcd_dir))
    print("out_dir: {}".format(out_dir))
    print("ball_ids: {}".format(ball_ids))
    print("symmetry_ids: {}".format(symmetry_ids))
    print("errs: {}".format(errs))
    print("aa_num: {}".format(aa_num))
    print("aa_min_angle: {}".format(aa_min_angle))
    print("auxiliary_axis_scale: {}".format(auxiliary_axis_scale))
    print("batch_sz: {}".format(batch_sz))
    print("------------------------------------------------------------------------------------")

    # get axis-angles from rotation space aas: [nv, 4]
    aas = torch.from_numpy(np.asarray(symmetric_axis_angle_sampler(aa_num, aa_min_angle), dtype='float')).cuda()
    aa_num, _ = aas.size()
    iter_num = aa_num // batch_sz

    # transform axis-angles to rotation matrices, Rms: [nv, 3, 3]
    Rms = Rmatrix_from_rotate_axis(aas)

    # get 3d model of objects from pcd_dir, mds: [nm, 3, np]
    mds, raw_mds, raw_ori, raw_radius = load_3d_models(pcd_dir, dataset, symmetry_ids)
    avg_radius = np.mean(np.asarray(raw_radius))

    mds_saas = []
    for i in range(len(mds)):
        print('md: ', i)
        if i in symmetry_ids:
            # 1. find symmetric axis-angles of object_i [ns, 4]
            sym_id = symmetry_ids.index(i)
            saa_ids = []
            min_adds_ls = []
            for j in tqdm(range(iter_num)):
                batch_Rms = Rms[(j * batch_sz): ((j + 1) * batch_sz)]
                batch_ids, min_adds = find_symmetric_rotation_matrices(mds[i], batch_Rms, errs[sym_id])
                batch_ids = batch_ids + j * batch_sz
                saa_ids.append(batch_ids)
                min_adds_ls.append(min_adds)

            if iter_num * batch_sz < aa_num:
                batch_Rms = Rms[(iter_num * batch_sz):]
                batch_ids, min_adds = find_symmetric_rotation_matrices(mds[i], batch_Rms, errs[sym_id])
                batch_ids = batch_ids + iter_num * batch_sz
                saa_ids.append(batch_ids)
                min_adds_ls.append(min_adds)

            print('min_adds: ', min(min_adds_ls))

            saa_ids = torch.cat(saa_ids, 0)
            print('num of raw symmetric axis-angles: ', saa_ids.size()[0])

            saa_ids = saa_ids.view(-1, 1).repeat((1, 4))
            cur_saas = torch.gather(aas, 0, saa_ids)

            if i not in ball_ids:
                # 2. simplify saas
                cur_saas = simplify_symmetric_axis_angle(cur_saas, errs[sym_id])
                print('num of final symmetric axis-angles: ', cur_saas.shape[0])

                # 3. construct group primitives using symmetric axis-angles
                cur_grps = construct_group_primitives(cur_saas, aa_min_angle, errs[sym_id], auxiliary_axis_scale)
                cur_grps = [[np.asarray([0., 0., 0.], dtype='float')]] + cur_grps
            else:
                cur_grps = [[np.asarray([0., 0., 0.], dtype='float')]]
        else:
            cur_grps = []
            cur_grps.append([np.asarray([0., 0., 0.], dtype='float')])
            cur_grps.append([np.asarray([1., 0., 0.], dtype='float')])
            cur_grps.append([np.asarray([0., 1., 0.], dtype='float')])
            cur_grps.append([np.asarray([0., 0., 1.], dtype='float')])

        # 4. postprocessing
        if scale_gp_by_shape and i not in ball_ids:
            cur_grps = scale_symmetric_grouped_primitives(cur_grps, raw_mds[i])
        if trim_gp and i not in symmetry_ids:
            cur_grps = trimming_asymmetric_grouped_primitives(cur_grps)
        if normalize_gp and i not in ball_ids:
            cur_grps = normalize_symmetric_grouped_primitives(cur_grps)

        print('ori:', list(raw_ori[i]))
        mds_saas.append({'obj_id': i,
                         'groups': [[list(p) for p in grp] for grp in cur_grps]})

        # draw plots
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(raw_mds[i][:, 0], raw_mds[i][:, 1], raw_mds[i][:, 2], marker='.', s=1)
        for k, grp in enumerate(cur_grps):
            for srv in grp:
                ppx = [0, srv[0]]
                ppy = [0, srv[1]]
                ppz = [0, srv[2]]
                ax.plot(ppx, ppy, ppz, c=color_map[k])
                ax.scatter([srv[0]], [srv[1]], [srv[2]], marker='o', s=400, c=[color_map[k]])
        ax.scatter(box3d[:, 0], box3d[:, 1], box3d[:, 2], marker='.', s=1)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_box_aspect((1., 1., 1.))
        plt.savefig(f'{out_dir}/{pre_name}_{i}.jpg')

    # scale by avg_radius
    for i, saas in enumerate(mds_saas):
        for j, grp in enumerate(saas['groups']):
            for v in grp:
                if j == 0:
                    v[0] = raw_ori[i][0]
                    v[1] = raw_ori[i][1]
                    v[2] = raw_ori[i][2]
                else:
                    v[0] = v[0] * avg_radius + raw_ori[i][0]
                    v[1] = v[1] * avg_radius + raw_ori[i][1]
                    v[2] = v[2] * avg_radius + raw_ori[i][2]

    with open(f'{out_dir}/{pre_name}_gp.json', 'w') as f:
        json.dump(mds_saas, f)
        print('finish.')
    return


def load_3d_models(models_dir, dataset, sym_ids):
    """
    Load point cloud of object 3d model
    - models_dir:
     - pcd_5000: Each object has 5000 sample points.
     - simple: We will use sparse representative points of each object to accelerate computation.
               sparse points are selected from pcd_5000.

    Args:
        models_dir: The directory with 3d models
        dataset: 'tless' or 'ycb'
        sym_ids: The indexes of symmetric objects in dataset

    Returns:
        cld: Normalized sparse representative points
        raw: Normalized raw points
        raw_ori: Mean x, y, z of raw points
        raw_radius: Radius of raw points

    """

    cld = []
    raw = []
    raw_ori = []
    raw_radius = []

    if dataset == 'tless':
        class_file = open('datasets/tless/classes.txt')
    elif dataset == 'ycb':
        print("not yet support!")
    else:
        print("not yet support!")
        return

    while 1:
        class_input = class_file.readline()
        if not class_input:
            break

        input_cloud = o3d.io.read_point_cloud('{0}/pcd_5000/{1}.pcd'.format(models_dir, class_input[:-1]))
        raw_xyz = torch.tensor(np.asarray(input_cloud.points), dtype=torch.float32).T

        # normalization
        raw_xyz *= 0.001  # transform unit to meter
        min_xyz, _ = torch.min(raw_xyz, dim=1)
        max_xyz, _ = torch.max(raw_xyz, dim=1)
        radius = torch.norm(max_xyz - min_xyz).item() / 2.
        mean_xyz = torch.mean(raw_xyz, dim=1, keepdim=True)
        raw_xyz = (raw_xyz - mean_xyz) / radius

        if len(cld) in sym_ids:
            # If this object is a symmetric object,
            # we will use sparse representative points to accelerate computation.
            simple_cloud = o3d.io.read_point_cloud('{0}/simple/{1}.pcd'.format(models_dir, class_input[:-1]))
            sim_xyz = torch.tensor(np.asarray(simple_cloud.points), dtype=torch.float32).T * 0.001
            sim_xyz = (sim_xyz - mean_xyz) / radius
        else:
            sim_xyz = mean_xyz

        cld.append(sim_xyz.view(3, -1).contiguous().cuda())
        raw.append(raw_xyz.T.cpu().numpy())
        raw_ori.append(mean_xyz.double().view(3).cpu().numpy())
        raw_radius.append(radius)

    return cld, raw, raw_ori, raw_radius


def find_symmetric_rotation_matrices(pts, Rms, err=0.01):
    """
    Find the symmetric rotation matrices of the point cloud from Rms

    Args:
        pts: The point cloud in object surface
        Rms: The candidate rotation matrices
        err: We will select rotation matrices, whose add-s are smaller than err

    Returns:
        ids: The indexes of symmetric rotation matrices in Rms

    """
    _, p_num = pts.size()
    R_num, _, _ = Rms.size()
    md_xyz = pts.view(1, 3, p_num).repeat(R_num, 1, 1)
    pr_xyz = torch.bmm(Rms, md_xyz)  # R_num, 3, p_num


    inds = knn_one_point(pr_xyz.permute(0, 2, 1), md_xyz.permute(0, 2, 1))  # R_num, p_num
    inds = inds.view(R_num, 1, p_num).repeat(1, 3, 1)
    tar_tmp = torch.gather(md_xyz, 2, inds)
    # add-s
    distance = torch.mean(torch.norm(pr_xyz - tar_tmp, dim=1), dim=1)  # [nv]
    # distance, _ = torch.max(torch.norm(pr_xyz - tar_tmp, dim=1), dim=1)  # [nv]
    ids = torch.where(distance < err)[0]
    return ids, torch.min(distance, dim=0)[0].item()


def simplify_symmetric_axis_angle(saas, err=0.01):
    """
    Raw symmetric axes are distributed around the target symmetric axis, because noise exists in calculation.
    Raw symmetric angle are multiple, but we only focus on the greatest common divisor of these angles.
    For example:
    target axis-angle:
        (1, 0, 0, 60)
    raw axis-angles:
        (0.99, 0.002, 0.08, 60)
        (0.98, 0.01, 0.01, 120)
        (0.992, 0.001, 0.07, 120)
        (0.994, 0.002, 0.06, 180)

    Raw symmetric axis-angles are simplified with following steps:
    1. using mean shift to get target axis from raw axes
    2. find the greatest common divisor in raw symmetric angles

    Args:
        saas: symmetric axis angles
        err: kernel_bandwidth in mean_shift

    Returns:
        res: [(x1, y1, z1, angle1, order1), ...]

    """
    axes = saas[:, :3].contiguous().cpu().numpy()
    angle = saas[:, 3].contiguous().cpu().numpy()
    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(axes, kernel_bandwidth=err)

    cluster_assignments = mean_shift_result.cluster_ids
    num_cluster = int(cluster_assignments.max() + 1)
    shifted_points = mean_shift_result.shifted_points

    symmetric_angle_list = []
    for i in range(1, 181):
        if 360 % i == 0:
            symmetric_angle_list.append(i)

    res = []
    for i in range(num_cluster):
        ids = np.where(cluster_assignments == i)[0]

        # calculate target axis
        cluster_i = shifted_points[ids]
        axis_i = np.mean(cluster_i, axis=0)
        axis_i = axis_i / (np.sqrt(np.sum(axis_i * axis_i)))

        # get the greatest common divisor
        ag = angle[ids] * 180. / math.pi
        ag = np.min(ag)
        min_diff = 9999
        res_ag = ag
        for sa in symmetric_angle_list:
            if abs(sa-ag) < min_diff:
                min_diff = abs(sa-ag)
                res_ag = sa

        res.append((axis_i[0], axis_i[1], axis_i[2], res_ag, 360//res_ag))

    return np.asarray(res, dtype='float')


def construct_group_primitives(saas, angle_thr=1, err=0.01, scale=0.5):
    """
    Construct group primitives using symmetric axis-angles

    Args:
        saas: symmetric_axis_angle list
        angle_thr: angle threshold
        err: group distance threshold
        scale: scale auxiliary axes

    Returns:

    """
    n = saas.shape[0]
    assert n % 2 == 0
    tensor_saas = saas.copy()
    tensor_saas[:, 3] = tensor_saas[:, 3] * math.pi / 180.
    tensor_saas = torch.tensor(tensor_saas[:, :4])
    Rs = Rmatrix_from_rotate_axis(tensor_saas).cpu().numpy()

    if n == 2:
        if saas[0][3] <= angle_thr:
            # handle the category 1 of symmetric object
            return [[saas[0][:3]]]
        else:
            # handle the category 2 of symmetric object
            g1 = [saas[1][:3]]
            g2 = []
            ph = saas[1][:3]
            pv = np.asarray([1., 0., 0.], dtype='float')
            pv = pv - np.dot(pv, ph) * ph / np.dot(ph, ph)
            Rh = Rs[0]
            for i in range(int(saas[1][4])):
                cur_pv = pv.copy()
                for j in range(i+1):
                    cur_pv = Rh @ cur_pv
                cur_pv = cur_pv / np.sqrt(np.sum(cur_pv*cur_pv)) # auxiliary axis
                g2.append(cur_pv * scale)
            return [g1, g2]
    elif n > 2:
        ps = []
        for i in range(n):
            if saas[i][3] <= angle_thr:
                ps.append(saas[i][:3])
        if len(ps) > 0:
            if len(ps) > 2:
                # handle the category 3 of symmetric object
                return []
            if len(ps) == 2:
                # handle the category 4 of symmetric object
                return [ps]
        else:
            # handle the category 5 of symmetric object
            grps = [[saas[0][:3]]]
            for i in range(1, n):
                in_g = False
                for j in range(len(grps)):
                    gp = grps[j][0]
                    for k in range(n):
                        if k == j:
                            continue
                        p = saas[i][:3].copy()
                        for m in range(int(saas[k][4])):
                            p = Rs[k] @ p
                            dist = np.sqrt(np.dot(p-gp, p-gp))
                            if dist < err:
                                grps[j].append(saas[i][:3])
                                in_g = True
                                break
                        if in_g:
                            break
                    if in_g:
                        break
                if in_g == False:
                    grps.append([saas[i][:3]])

            return grps
    else:
        return []


def scale_symmetric_grouped_primitives(cur_grps, raw_md):
    new_grps = []
    for i, grp in enumerate(cur_grps):
        scaled_grp = []
        for j, sa in enumerate(grp):
            projs = np.zeros(raw_md.shape[0], dtype='float')
            for k in range(raw_md.shape[0]):
                projs[k] = np.abs(np.dot(sa, raw_md[k]))
            scale_sa = sa * projs.max()
            scaled_grp.append(scale_sa)
        new_grps.append(scaled_grp)

    return new_grps


def trimming_asymmetric_grouped_primitives(cur_grps):
    new_grps = []
    new_grps.append(cur_grps[0])  # center point
    _, min_id = torch.min(torch.tensor([cur_grps[1][0][0], cur_grps[2][0][1], cur_grps[3][0][2]]).view(3, 1), dim=0)
    if min_id.size()[0] > 1:
        min_id = min_id[0]
    for i in range(len(cur_grps)-1):
        if i != min_id:
            new_grps.append(cur_grps[i+1])

    return new_grps


def normalize_symmetric_grouped_primitives(cur_grps):
    new_grps = []
    max_l = 0
    for i, grp in enumerate(cur_grps):
        for j, srv in enumerate(grp):
            l = np.sqrt(np.sum(srv * srv))
            if l > max_l:
                max_l = l

    scale = 1. / max_l
    for i, grp in enumerate(cur_grps):
        n_grp = []
        for j, srv in enumerate(grp):
            n_grp.append(srv * scale)
        new_grps.append(n_grp)

    return new_grps


if __name__ == '__main__':
    main()