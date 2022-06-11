# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
import numpy.ma as ma
import scipy.io as scio
from PIL import Image
from multiprocessing import Pool
import os
import json
import shutil, cv2
import random
import argparse
from matplotlib import pyplot as plt


# read file list
def read_file_list(file_path, sample_step=1):
    print('start read')
    file_list = []
    file_real = []
    file_syn = []
    input_file = open(file_path)
    step = 0
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        step += 1
        if step % sample_step != 0: continue
        if input_line[:5] == 'data/':
            file_real.append(input_line)
        else:
            file_syn.append(input_line)
        file_list.append(input_line)
    input_file.close()
    print('end read')
    return file_list, file_real, file_syn

def get_bbox(label):
    img_length = label.shape[1]
    img_width = label.shape[0]
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    wid = max(r_b, c_b)
    extend_wid = int(wid / 8)
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(wid / 2) - extend_wid
    rmax = center[0] + int(wid / 2) + extend_wid
    cmin = center[1] - int(wid / 2) - extend_wid
    cmax = center[1] + int(wid / 2) + extend_wid
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def get_data(trian_root_path, gt_mask):

    data_package = []

    files = os.listdir(trian_root_path)
    idx = 0
    for dir_name in files:

        print(dir_name)
        scene_path = os.path.join(trian_root_path, dir_name)
        scene_json_path = scene_path + '/scene_gt.json'

        with open(scene_json_path, 'r', encoding='UTF-8') as f:
            scene_gt_info = json.load(f)

        camera_info_path = scene_path + '/scene_camera.json'

        with open(camera_info_path, 'r', encoding='UTF-8') as f:
            camera_info = json.load(f)

        if 'test' in scene_path and not gt_mask:
            mask_visib_dir = scene_path + '/mask_visib_pred_correct_eroded/'

        else:
            mask_visib_dir = scene_path + '/mask_visib/'

        depth_dir =  scene_path + '/depth/'
        rgb_dir =  scene_path + '/rgb/'

        visiblemask = os.listdir(mask_visib_dir)

        for instance_mask in visiblemask:

            if 'test' in rgb_dir:
                instance_rgb_path = rgb_dir + instance_mask[:6] + '.png'  # test

            else:
                instance_rgb_path = rgb_dir + instance_mask[:6] + '.jpg'   # train

            instance_depth_path = depth_dir + instance_mask[:6] + '.png'

            instance_mask_path = mask_visib_dir + instance_mask

            if 'test_primesense' in instance_mask_path:

                if not gt_mask:
                    instance_save_dir = instance_mask_path.replace("test_primesense", "test_primesense_mat").replace("/mask_visib_pred_correct_eroded", "")[:-4]

                else:
                    instance_save_dir = instance_mask_path.replace("test_primesense", "test_primesense_gt_mask_mat").replace("/mask_visib", "")[:-4]
            else:
                instance_save_dir = instance_mask_path.replace("train_pbr", "train_pbr_mat").replace("/mask_visib", "")[:-4]


            instance_gt_info = scene_gt_info["{}".format(int(instance_mask[:6]))][int(instance_mask[8:-4])]
            # cam_R_m2c, cam_t_m2c, obj_id

            instance_camera_info = camera_info["{}".format(int(instance_mask[:6]))]

            idx += 1
            data_package.append([instance_rgb_path, instance_depth_path, instance_mask_path, instance_gt_info, instance_camera_info, instance_save_dir, idx])


    print("Data preparation finish! In total : {}".format(len(data_package)) )

    # exit()
    return data_package


def prepare_data(input_data):

    instance_rgb_path, instance_depth_path, instance_mask_path, instance_gt_info, instance_camera_info, instance_save_dir, idx = input_data

    cam_scale = instance_camera_info['depth_scale']
    img = Image.open(instance_rgb_path)
    depth = np.array(Image.open(instance_depth_path)).astype(np.float32)

    depth *= cam_scale*0.001

    mask = np.array(Image.open(instance_mask_path))  # MASK

    rows, cols = depth.shape
    ymap = np.array([[j for i in range(cols)] for j in range(rows)]).astype(np.float32)
    xmap = np.array([[i for i in range(cols)] for j in range(rows)]).astype(np.float32)


    # record camera intrinsic
    '''
    fx  0   cx
    0   fy  cy
    0   0   1
    '''

    cam_cx = instance_camera_info['cam_K'][2]
    cam_cy = instance_camera_info['cam_K'][5]
    cam_fx = instance_camera_info['cam_K'][0]
    cam_fy = instance_camera_info['cam_K'][4]


    # loop for objects in one image
    obj_id = instance_gt_info['obj_id']#.astype(np.int32)

    mask_thershold = 0.1  # 可视mask的阈值

    if len(mask.nonzero()[0]) >= mask_thershold:

        rmin, rmax, cmin, cmax = get_bbox(mask)
        img_crop = np.array(img)[:, :, :3][rmin:rmax, cmin:cmax, :]
        mask_crop = mask[rmin:rmax, cmin:cmax].astype(np.float32) / 255.
        depth_crop = depth[rmin:rmax, cmin:cmax, np.newaxis].astype(np.float32)

        # choose num_pt from the selected object or pad to num_pt
        xmap_masked = xmap[rmin:rmax, cmin:cmax, np.newaxis]
        ymap_masked = ymap[rmin:rmax, cmin:cmax, np.newaxis]

        pt2 = depth_crop
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy

        depth_xyz = np.concatenate((pt0, pt1, pt2), axis=2)
        depth_mask_xyz = depth_xyz * mask_crop[:, :, np.newaxis]
        choose = depth_mask_xyz[:, :, 2].flatten().nonzero()[0]

        if choose.size <= 0:
            # print(instance_rgb_path)
            return None

        mask_x = depth_xyz[:, :, 0].flatten()[choose][:, np.newaxis]
        mask_y = depth_xyz[:, :, 1].flatten()[choose][:, np.newaxis]
        mask_z = depth_xyz[:, :, 2].flatten()[choose][:, np.newaxis]
        mask_xyz = np.concatenate((mask_x, mask_y, mask_z), axis=1)

        mean_xyz = mask_xyz.mean(axis=0).reshape(3)

        tar_t = np.array(instance_gt_info["cam_t_m2c"]).reshape(3)*0.001

        if np.sqrt(np.dot(mean_xyz-tar_t, mean_xyz-tar_t)) > 0.1:

            # remove the data mean point cloud center is large than the tar_t over 0.1
            return None

        mean_xyz = mean_xyz.reshape((1, 1, 3))

        # print('mean_xyz:', mean_xyz)
        depth_xyz = (depth_xyz - mean_xyz) * mask_crop[:, :, np.newaxis]

        cur_meta = {}
        cur_meta['mean_xyz'] = mean_xyz.astype('float32')
        cur_meta['rgb'] = img_crop.astype('uint8')
        cur_meta['mask'] = mask_crop.astype('uint8')
        cur_meta['xyz'] = depth_xyz

        #
        R = np.array(instance_gt_info["cam_R_m2c"]).reshape(3,3)
        T = np.array(instance_gt_info["cam_t_m2c"]).reshape(3,1)

        pose = np.concatenate((R, T), axis= 1)

        pose[:3, 3] = pose[:3, 3]*0.001 - mean_xyz.reshape(3)

        cur_meta['poses'] = pose
        cur_meta['cls_indexes'] = obj_id

        # discard the dirty training data
        if 'train' in instance_save_dir:

            rgb = cur_meta['rgb'].astype(np.float32)
            mask = cur_meta['mask'].astype(np.float32)
            xyz = cur_meta['xyz'].astype(np.float32)
            target = cur_meta['poses'].astype(np.float32)

            dis_xyz = np.sqrt(xyz[:, :, 0] * xyz[:, :, 0] + xyz[:, :, 1] * xyz[:, :, 1] + xyz[:, :, 2] * xyz[:, :, 2])
            mask_xyz = np.where(dis_xyz > 0.01, 1.0, 0.0).astype(np.float32)

            if np.isnan(xyz).sum() == 0 and np.isnan(rgb).sum() == 0 and np.isnan(target).sum() == 0 and np.isinf(
                    xyz).sum() == 0 and np.isinf(rgb).sum() == 0 and np.isinf(target).sum() == 0 and \
                    np.isnan(mask).sum() == 0 and np.isinf(mask).sum() == 0 and mask_xyz.sum() > 50:

                pass

            else:
                return None

        print('instance_save_dir:', instance_save_dir[:-13])
        if not os.path.isdir(instance_save_dir[:-13]):
            os.makedirs(instance_save_dir[:-13])
        scio.savemat('{}.mat'.format(instance_save_dir), cur_meta)

    return None


def erode_mask(instance_mask_path):

    mask = cv2.imread(instance_mask_path, 0)
    kernel = np.ones((7, 7), dtype=np.uint8)
    erode_mask = cv2.erode(mask, kernel)

    instance_mask_dilate_path = instance_mask_path.replace('mask_visib_pred_correct', 'mask_visib_pred_correct_eroded')

    if not os.path.isdir(instance_mask_dilate_path[:-17]):
        os.makedirs(instance_mask_dilate_path[:-17])

    cv2.imwrite(instance_mask_dilate_path, erode_mask)

    return 1


def translate_gt_mask2_pred_mask(data_path):

    '''
    Some index of the mask is incorrect on the test_primesense prediction set.
    e.g.,
    /tless/test_primesense/000001/mask_visib_pred/000017_000003.png
    /tless/test_primesense/000001/mask_visib/000017_000003.png

    we correct the index by translate the index from the mask_visib to mask_visib_pred by the maximum IOU.

    '''

    test_primesense_path = os.path.join(data_path, 'test_primesense')

    with open(os.path.join(data_path, 'test_primesense_gt_mask.txt'), 'a') as file_write:
        for f in os.listdir(test_primesense_path):
            g = os.walk(os.path.join(test_primesense_path, f , 'mask_visib'))
            for path, dir_list, file_list in g:
                for index in os.listdir(path):
                    print(os.path.join(path, index))
                    file_write.write(os.path.join(path, index))
                    file_write.write('\n')

    with open(os.path.join(data_path, 'test_primesense_pred_mask.txt'), 'a') as file_write:
        for f in os.listdir(test_primesense_path):
            g = os.walk(os.path.join(test_primesense_path, f , 'mask_visib_pred'))
            for path, dir_list, file_list in g:
                for index in os.listdir(path):
                    print(os.path.join(path, index))
                    file_write.write(os.path.join(path, index))
                    file_write.write('\n')

    print('Finishing get list')

    visib_gt_mask_path = open(os.path.join(data_path, 'test_primesense_gt_mask.txt'))
    visib_pred_mask_path = open(os.path.join(data_path, 'test_primesense_pred_mask.txt'))

    gt_mask_path = []
    pred_mask_path = []

    while 1:
        instance_visib_gt_mask_path = visib_gt_mask_path.readline()
        if not instance_visib_gt_mask_path:
            break

        if instance_visib_gt_mask_path[-1:] == '\n':
            instance_visib_gt_mask_path = instance_visib_gt_mask_path[:-1]
            gt_mask_path.append(instance_visib_gt_mask_path)

    while 1:

        instance_visib_pred_mask_path = visib_pred_mask_path.readline()
        if not instance_visib_pred_mask_path:
            break

        if instance_visib_pred_mask_path[-1:] == '\n':
            instance_visib_pred_mask_path = instance_visib_pred_mask_path[:-1]

            pred_mask_path.append(instance_visib_pred_mask_path)

    for gt_mask in gt_mask_path:

        instance_gt_mask = np.array(Image.open(gt_mask))  # MASK

        mask_thershold = 50  # ignore the mask less than 50 pixels

        if len(instance_gt_mask.nonzero()[0]) >= mask_thershold:

            mask_image = []
            for pred_mask in pred_mask_path:
                if gt_mask.replace('mask_visib', 'mask_visib_pred')[:-10] in pred_mask:
                    mask_image.append(pred_mask)

            match_score = []
            for instance_mask_path in mask_image:
                instance_mask = np.array(Image.open(instance_mask_path))
                overlap_mask = instance_mask*instance_gt_mask
                match_score.append(len(overlap_mask.nonzero()[0]))

            best_match = mask_image[match_score.index(max(match_score))]  # find the maximum IOU and save the renamed pred mask

            cp_name = gt_mask.replace('/mask_visib', '/mask_visib_pred_correct')

            target_path = cp_name[:-17]

            if not os.path.isdir(target_path):
                os.makedirs(target_path)

            shutil.copy(best_match, cp_name)
    #
    print('Finishing correct the mask')

    with open(os.path.join(data_path, 'test_primesense_pred_correct_mask.txt'), 'a') as file_write:
        for f in os.listdir(test_primesense_path):
            g = os.walk(os.path.join(test_primesense_path, f , 'mask_visib_pred_correct'))
            for path, dir_list, file_list in g:
                for index in os.listdir(path):
                    print(os.path.join(path, index))
                    file_write.write(os.path.join(path, index))
                    file_write.write('\n')

    correct_mask_list = open(os.path.join(data_path, 'test_primesense_pred_correct_mask.txt'))

    while 1:
        instance_visib_pred_mask_path = correct_mask_list.readline()
        if not instance_visib_pred_mask_path:
            break

        if instance_visib_pred_mask_path[-1:] == '\n':
            instance_visib_pred_mask_path = instance_visib_pred_mask_path[:-1]
            print(instance_visib_pred_mask_path)
            erode_mask(instance_visib_pred_mask_path)

    print('Finishing erode the correct_mask')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tless_path', type=str, default='', help='tless path')
    parser.add_argument('--gt_mask', type=bool, default=True, help='gt mask or estimation mask from stablepose')
    parser.add_argument('--train_set', type=bool, default=True, help='gt mask or estimation mask from stablepose')


    opt = parser.parse_args()

    tless_path = opt.tless_path
    gt_mask = opt.gt_mask



    if not opt.train_set:

        # prepare test set
        # translate_gt_mask2_pred_mask(tless_path) # correct the pred mask and erode it

        print("begin the gt mask testing set preparation!")

        tless_test_path = os.path.join(tless_path, 'test_primesense')
        data_package = get_data(tless_test_path, gt_mask = gt_mask)

        print("length data:", len(data_package))

        print('----start----')
        with Pool(processes=20) as pl:
            train_file_list = pl.map(prepare_data, data_package)


        mat_path = os.path.join(tless_path, 'test_primesense_gt_mask_mat')

        f = open(os.path.join(tless_path, 'test_primesense_gt_mask_mat.txt'), 'a')
        for root, dirs, files in os.walk(mat_path, topdown=False):
            for name in files:
                if name[-4:] != '.mat':
                    continue
                file_path = os.path.join(root, name)[-24:-4]

                f.write(file_path)
                f.write('\n')

        f.close()
        print("finish the gt mask testing set preparation!")



    else:

        # # prepare train set
        print("start training set preparation!")

        tless_train_path = os.path.join(tless_path, 'train_pbr')
        data_package = get_data(tless_train_path, gt_mask = True)

        print("length data:", len(data_package))

        print('----start----')
        with Pool(processes=20) as pl:
            train_file_list = pl.map(prepare_data, data_package)


        mat_path = os.path.join(tless_path, 'train_pbr_mat')
        f = open(os.path.join(tless_path, 'train_pbr_mat.txt'), 'a')
        for root, dirs, files in os.walk(mat_path, topdown=False):
            for name in files:
                if name[-4:] != '.mat':
                    continue
                # print(os.path.join(root, name))
                file_path = os.path.join(root, name)[-24:-4]

                f.write(file_path)
                f.write('\n')

        f.close()
        print("finish train preparation!")

        # exit()

