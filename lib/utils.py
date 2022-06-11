import logging
import torch
import torch.nn.functional as F
import numpy as np
from lib.transformations import quaternion_matrix
import json
import pandas as pd
import pdb


def warnup_lr(cur_iter, end_iter, start_lr, end_lr):
    if(cur_iter < end_iter):
        return start_lr + (end_lr - start_lr) * cur_iter / end_iter
    else:
        return end_lr


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l


def save_pred_and_gt_json(rt_list, total_instance_list, gt_rt_list, gt_cls_list, save_path):


    rt = np.stack(rt_list)
    gt = np.stack(gt_rt_list)
    cls = np.stack(gt_cls_list)

    jdict = {'pred_rt': rt.tolist(), 'gt_rt': gt.tolist(), 'cls': cls.tolist()}
    file_hd = open(save_path + '/results.json', 'w', encoding='utf-8')
    jobj = json.dump(jdict, file_hd)
    file_hd.close()

    scene_id_ls = []
    im_id_ls = []
    obj_id_ls = []
    score_ls = []
    time_ls = []

    for i in range (len(total_instance_list)):

        scene_id_ls.append(total_instance_list[i][0][0])
        im_id_ls.append(total_instance_list[i][0][1])
        obj_id_ls.append(cls.tolist()[i][0])

        score_ls.append(1)
        time_ls.append(-1)

    # exit()
    r_ls = []
    t_ls =[]

    for instance_pred in rt.tolist():

        instance_pred = np.array(instance_pred)
        pred_R = instance_pred[:, 0:3]

        # pred_R = pred_R.reshape((1, -1))[0].tolist()
        pred_R = pred_R.reshape((1, -1))[0].tolist()


        if len(pred_R) != 9:

            exit()

        pred_R = str(pred_R)[1:-1].replace(',', ' ')



        pred_T = instance_pred[:, 3:4].reshape((1, -1))[0] * 1000
        pred_T = pred_T.tolist()


        pred_T = str(pred_T)[1:-1].replace(',', ' ')


        r_ls.append(pred_R)
        t_ls.append(pred_T)

        # print("pred_R:", pred_R)
        # print("pred_T:", pred_T)

    # save_csv
    dataframe = pd.DataFrame({'scene_id': scene_id_ls, 'im_id': im_id_ls, 'obj_id': obj_id_ls, 'score': score_ls,
                            'R': r_ls, 't': t_ls, 'time': time_ls})
    dataframe.to_csv(save_path+ "/test.csv", index=False, sep=',')





def post_processing_ycb_1(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask and score
    :param preds: output of pose net ['pred_x'][bs, 3, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_x'].size()
    obj_ids = torch.tensor([i for i in range(b)]).long().cuda()
    sr = preds['score'].view(b, -1)
    st = preds['score'].view(b, -1)
    px = preds['pred_x'].view(b, 3, -1)
    py = preds['pred_y'].view(b, 3, -1)
    pz = preds['pred_z'].view(b, 3, -1)
    pt = preds['pred_t'].view(b, 3, -1)
    mask = preds['mask']

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        res_px = px[i][:, valid_pixels]
        res_py = py[i][:, valid_pixels]
        res_pz = pz[i][:, valid_pixels]
        res_pt = pt[i][:, valid_pixels]
        res_sr = sr[i][valid_pixels]
        res_st = st[i][valid_pixels]

        # res_px[3, nv] res_ux[nv]
        res_sr = torch.topk(res_sr, min(num_val, 32), dim=0, largest=True)
        res_st = torch.topk(res_st, min(num_val, 32), dim=0, largest=True)

        r_ids = res_sr.indices.unsqueeze(dim=0).repeat(3, 1)
        t_ids = res_st.indices.unsqueeze(dim=0).repeat(3, 1)

        res_sr = res_sr.values
        res_st = res_st.values

        res_px = torch.gather(res_px, dim=1, index=r_ids)
        res_py = torch.gather(res_py, dim=1, index=r_ids)
        res_pz = torch.gather(res_pz, dim=1, index=r_ids)
        res_pt = torch.gather(res_pt, dim=1, index=t_ids)
        # res_px[3, 32]
        res_px = torch.sum(res_sr * res_px, dim=1) / (torch.sum(res_sr) + 0.000001)
        res_py = torch.sum(res_sr * res_py, dim=1) / (torch.sum(res_sr) + 0.000001)
        res_pz = torch.sum(res_sr * res_pz, dim=1) / (torch.sum(res_sr) + 0.000001)
        res_pt = torch.sum(res_st * res_pt, dim=1) / (torch.sum(res_st) + 0.000001)
        res_sr = torch.sum(res_sr)
        res_st = torch.sum(res_st)

        res_px = res_px / torch.norm(res_px, dim=0).unsqueeze(dim=0)
        res_py = res_py / torch.norm(res_py, dim=0).unsqueeze(dim=0)
        res_pz = res_pz / torch.norm(res_pz, dim=0).unsqueeze(dim=0)


        if cls_ids[i].item() in sym_list:
            if cls_ids[i].item() == 12:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 15:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 18:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
            if cls_ids[i].item() == 19:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 1, 2])
            if cls_ids[i].item() == 20:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
        else:
            res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 1, 0])
        res_T.append(torch.cat([res_r, res_pt.view(3, 1)], dim=1))

    return torch.stack(res_T, dim=0)


def post_processing_ycb_2(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask and score
    :param preds: output of pose net ['pred_x'][bs, 3, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_x'].size()
    obj_ids = torch.tensor([i for i in range(b)]).long().cuda()
    ux = preds['scor_x'].view(b, -1)
    uy = preds['scor_y'].view(b, -1)
    uz = preds['scor_z'].view(b, -1)
    ut = preds['scor_t'].view(b, -1)
    px = preds['pred_x'].view(b, 3, -1)
    py = preds['pred_y'].view(b, 3, -1)
    pz = preds['pred_z'].view(b, 3, -1)
    pt = preds['pred_t'].view(b, 3, -1)
    mask = preds['mask']

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        res_px = px[i][:, valid_pixels]
        res_py = py[i][:, valid_pixels]
        res_pz = pz[i][:, valid_pixels]
        res_pt = pt[i][:, valid_pixels]
        res_ux = ux[i][valid_pixels]
        res_uy = uy[i][valid_pixels]
        res_uz = uz[i][valid_pixels]
        res_ut = ut[i][valid_pixels]

        # res_px[3, nv] res_ux[nv]
        res_ux = torch.topk(res_ux, min(num_val, 32), dim=0, largest=True)
        res_uy = torch.topk(res_uy, min(num_val, 32), dim=0, largest=True)
        res_uz = torch.topk(res_uz, min(num_val, 32), dim=0, largest=True)
        res_ut = torch.topk(res_ut, min(num_val, 32), dim=0, largest=True)

        x_ids = res_ux.indices.unsqueeze(dim=0).repeat(3, 1)
        y_ids = res_uy.indices.unsqueeze(dim=0).repeat(3, 1)
        z_ids = res_uz.indices.unsqueeze(dim=0).repeat(3, 1)
        t_ids = res_ut.indices.unsqueeze(dim=0).repeat(3, 1)

        # res_ux = torch.mean(res_ux.values)
        # res_uy = torch.mean(res_uy.values)
        # res_uz = torch.mean(res_uz.values)
        # res_ut = torch.mean(res_ut.values)
        res_ux = res_ux.values
        res_uy = res_uy.values
        res_uz = res_uz.values
        res_ut = res_ut.values

        res_px = torch.gather(res_px, dim=1, index=x_ids)
        res_py = torch.gather(res_py, dim=1, index=y_ids)
        res_pz = torch.gather(res_pz, dim=1, index=z_ids)
        res_pt = torch.gather(res_pt, dim=1, index=t_ids)
        # res_px[3, 32]
        # res_px = torch.mean(res_px, dim=1)
        # res_py = torch.mean(res_py, dim=1)
        # res_pz = torch.mean(res_pz, dim=1)
        # res_pt = torch.mean(res_pt, dim=1)
        res_px = torch.sum(res_ux * res_px, dim=1) / (torch.sum(res_ux) + 0.000001)
        res_py = torch.sum(res_uy * res_py, dim=1) / (torch.sum(res_uy) + 0.000001)
        res_pz = torch.sum(res_uz * res_pz, dim=1) / (torch.sum(res_uz) + 0.000001)
        res_pt = torch.sum(res_ut * res_pt, dim=1) / (torch.sum(res_ut) + 0.000001)
        res_ux = torch.sum(res_ux)
        res_uy = torch.sum(res_uy)
        res_uz = torch.sum(res_uz)
        res_ut = torch.sum(res_ut)

        res_px = res_px / torch.norm(res_px, dim=0).unsqueeze(dim=0)
        res_py = res_py / torch.norm(res_py, dim=0).unsqueeze(dim=0)
        res_pz = res_pz / torch.norm(res_pz, dim=0).unsqueeze(dim=0)

        if cls_ids[i].item() in sym_list:
            if cls_ids[i].item() == 12:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 15:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 18:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 2, 0])
            if cls_ids[i].item() == 19:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 2, 1])
            if cls_ids[i].item() == 20:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
        else:
            if res_ux >= res_uy and res_ux >= res_uz:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 2, 1])
            if res_uy >= res_ux and res_uy >= res_uz:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 2, 0])
            if res_uz >= res_ux and res_uz >= res_uy:
                if res_ux > res_uy:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 0, 1])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 1, 0])
        res_T.append(torch.cat([res_r, res_pt.view(3, 1)], dim=1))

    return torch.stack(res_T, dim=0)


def post_processing_ycb_3(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask
    :param preds: output of pose net ['pred_x'][bs, 3, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_x'].size()
    px = preds['pred_x'].view(b, 3, -1)
    py = preds['pred_y'].view(b, 3, -1)
    pz = preds['pred_z'].view(b, 3, -1)
    pt = preds['pred_t'].view(b, 3, -1)
    mask = preds['mask']

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        res_px = px[i][:, valid_pixels]
        res_py = py[i][:, valid_pixels]
        res_pz = pz[i][:, valid_pixels]
        res_pt = pt[i][:, valid_pixels]

        # get voting score
        res_ux = vote_strategy(res_px, dist_thr=0.05)
        res_uy = vote_strategy(res_py, dist_thr=0.05)
        res_uz = vote_strategy(res_pz, dist_thr=0.05)
        res_ut = vote_strategy(res_pt, dist_thr=0.025)

        # res_px[3, nv] res_ux[nv]
        res_ux = torch.topk(res_ux, min(num_val, 32), dim=0, largest=True)
        res_uy = torch.topk(res_uy, min(num_val, 32), dim=0, largest=True)
        res_uz = torch.topk(res_uz, min(num_val, 32), dim=0, largest=True)
        res_ut = torch.topk(res_ut, min(num_val, 32), dim=0, largest=True)

        x_ids = res_ux.indices.unsqueeze(dim=0).repeat(3, 1)
        y_ids = res_uy.indices.unsqueeze(dim=0).repeat(3, 1)
        z_ids = res_uz.indices.unsqueeze(dim=0).repeat(3, 1)
        t_ids = res_ut.indices.unsqueeze(dim=0).repeat(3, 1)

        res_ux = torch.mean(res_ux.values)
        res_uy = torch.mean(res_uy.values)
        res_uz = torch.mean(res_uz.values)
        res_ut = torch.mean(res_ut.values)

        res_px = torch.gather(res_px, dim=1, index=x_ids)
        res_py = torch.gather(res_py, dim=1, index=y_ids)
        res_pz = torch.gather(res_pz, dim=1, index=z_ids)
        res_pt = torch.gather(res_pt, dim=1, index=t_ids)
        # res_px[3, 32]
        res_px = torch.mean(res_px, dim=1)
        res_py = torch.mean(res_py, dim=1)
        res_pz = torch.mean(res_pz, dim=1)
        res_pt = torch.mean(res_pt, dim=1)

        res_px = res_px / torch.norm(res_px, dim=0).unsqueeze(dim=0)
        res_py = res_py / torch.norm(res_py, dim=0).unsqueeze(dim=0)
        res_pz = res_pz / torch.norm(res_pz, dim=0).unsqueeze(dim=0)

        if cls_ids[i].item() in sym_list:
            if cls_ids[i].item() == 12:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 15:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 18:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 2, 0])
            if cls_ids[i].item() == 19:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 2, 1])
            if cls_ids[i].item() == 20:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
        else:
            if res_ux >= res_uy and res_ux >= res_uz:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 2, 1])
            if res_uy >= res_ux and res_uy >= res_uz:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 2, 0])
            if res_uz >= res_ux and res_uz >= res_uy:
                if res_ux > res_uy:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 0, 1])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 1, 0])
        res_T.append(torch.cat([res_r, res_pt.view(3, 1)], dim=1))

    return torch.stack(res_T, dim=0)


def primitives_to_rotation_sym(prim_list, order_list=[0, 1, 2]):
    p = []
    p.append(prim_list[order_list[0]])
    p.append(prim_list[order_list[1]])
    p[1] = p[1] - torch.dot(p[0], p[1]) * p[0]
    p[1] = p[1] / torch.norm(p[1])
    p3 = torch.zeros(p[1].size()).cuda().float()
    p3[0] = p[0][1] * p[1][2] - p[0][2] * p[1][1]
    p3[1] = - p[0][0] * p[1][2] + p[0][2] * p[1][0]
    p3[2] = p[0][0] * p[1][1] - p[0][1] * p[1][0]
    p.append(p3)

    if order_list[0] == 0 and order_list[1] == 1:
        px = p[0]
        py = p[1]
        pz = p[2]
    if order_list[0] == 0 and order_list[1] == 2:
        px = p[0]
        py = -p[2]
        pz = p[1]
    if order_list[0] == 1 and order_list[1] == 0:
        px = p[1]
        py = p[0]
        pz = -p[2]
    if order_list[0] == 1 and order_list[1] == 2:
        px = p[2]
        py = p[0]
        pz = p[1]
    if order_list[0] == 2 and order_list[1] == 0:
        px = p[1]
        py = p[2]
        pz = p[0]
    if order_list[0] == 2 and order_list[1] == 1:
        px = -p[2]
        py = p[1]
        pz = p[0]

    res_R = torch.cat([px.unsqueeze(dim=1), py.unsqueeze(dim=1), pz.unsqueeze(dim=1)], dim=1)
    return res_R


def primitives_to_rotation(prim_list, order_list=[0, 1, 2]):
    AA = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    m = AA.shape[1]
    BB = torch.stack(prim_list, dim=0).detach().cpu().numpy().astype(np.float32)
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    return torch.from_numpy(R).cuda()


def vote_strategy(pred_x, dist_thr=0.1):
    c, n = pred_x.size()
    # 1. calculate distance between every pred_x[i] and pred_x[j]
    x1 = pred_x.unsqueeze(dim=0).repeat(n, 1, 1)
    x2 = pred_x.permute(1, 0).unsqueeze(dim=2).repeat(1, 1, n)
    d = torch.norm(x1 - x2, dim=1)
    # 2. count the vote of pred_x[i] by distance and the threshold
    d = d < dist_thr
    vote_num = torch.sum(d, dim=1).float()
    return vote_num


def cal_mean_std(dataset):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('==> Computing mean and std..')

    print("len dataset:", len(dataset))

    mask_0 = []

    for data in trainloader:

        inputs = data['rgb']# .cuda()
        # print("mask_flag:", data['mask_flag'])
        if data['mask_flag'] != 0:
            print("mask_flag:", data['mask_flag'])

            mask_0.append(data['mask_flag'][0])

        # print("len dataset:", len(dataset))
        #
        # print("inputs size:", inputs.shape)
        # exit()

        for i in range(3):

            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    # print('mean: {}, std: {}'.format(mean, std))

    return mean, std, mask_0

def post_processing_ycb_quaternion(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_r'].size()
    px = preds['pred_r'].detach().view(b, 4, -1)
    pt = preds['pred_t'].detach().view(b, 3, -1)
    ps = preds['pred_s'].detach().view(b, -1)
    mask = preds['xyz'][:, 0].unsqueeze(dim=1).detach()

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        q = px[i].view(4, -1)[:, valid_pixels]
        t = pt[i].view(3, -1)[:, valid_pixels]
        s = ps[i].view(-1)[valid_pixels]
        s_id = torch.argmax(s)
        _q = q[:, s_id].cpu().numpy()
        _r = quaternion_matrix(_q)[:3, :3]
        _r = torch.from_numpy(_r).cuda().float()
        _t = t[:, s_id].view(3, 1)
        res_T.append(torch.cat([_r, _t], dim=1))

    return torch.stack(res_T, dim=0)


def post_processing_ycb_quaternion_wi_vote(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask
    :param preds: output of pose net ['pred_x'][bs, 4, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_r'].size()
    px = preds['pred_r'].detach().view(b, 4, -1)
    pt = preds['pred_t'].detach().view(b, 3, -1)

    ps = preds['pred_s'].detach().view(b, -1)   # confidence score

    mask = preds['xyz'][:, 0].unsqueeze(dim=1).detach()

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []

    for i in range(b):

        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]

        if num_val < 32:

            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]

        q = px[i].view(4, -1)[:, valid_pixels]
        t = pt[i].view(3, -1)[:, valid_pixels]
        s = ps[i].view(-1)[valid_pixels]


        k_s = torch.topk(s, min(num_val, 32), dim=0, largest=True)
        s_id = k_s.indices.unsqueeze(dim=0).repeat(4, 1)
        s_v = k_s.values
        res_t = torch.gather(t, dim=1, index=s_id[:3, :])
        res_q = torch.gather(q, dim=1, index=s_id[:4, :])
        n_id = res_q[0, :] < 0
        res_q[:, n_id] = -res_q[:, n_id]


        # res_px[3, 32]
        res_t = torch.sum(s_v * res_t, dim=1) / max(torch.sum(s_v), 0.0001)
        res_q = torch.sum(s_v * res_q, dim=1) / max(torch.sum(s_v), 0.0001)
        res_q = res_q / torch.norm(res_q, dim=0)
        s_id = torch.argmax(s)

        _q = q[:, s_id].cpu().numpy()
        # _q = res_q.cpu().numpy()

        _r = quaternion_matrix(_q)[:3, :3]
        _r = torch.from_numpy(_r).cuda().float()
        _t = res_t.view(3, 1)
        res_T.append(torch.cat([_r, _t], dim=1))

    return torch.stack(res_T, dim=0)


def post_processing_translation_and_ratation(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask
    :param preds: output of pose net ['pred_x'][bs, 4, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_r'].size()
    px = preds['pred_r'].detach().view(b, 4, -1)
    pt = preds['pred_t'].detach().view(b, 3, -1)
    r_score = preds['pred_rs'].detach().view(b, -1)
    t_score = preds['pred_ts'].detach().view(b, -1)
    mask = preds['xyz'][:, 0].unsqueeze(dim=1).detach()

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        q = px[i].view(4, -1)[:, valid_pixels]
        t = pt[i].view(3, -1)[:, valid_pixels]
        rs = r_score[i].view(-1)[valid_pixels]
        ts = t_score[i].view(-1)[valid_pixels]
        rs_id = torch.argmax(rs)
        ts_id = torch.argmax(ts)
        _q = q[:, rs_id].cpu().numpy()
        _r = quaternion_matrix(_q)[:3, :3]
        _r = torch.from_numpy(_r).cuda().float()
        _t = t[:, ts_id].view(3, 1)
        res_T.append(torch.cat([_r, _t], dim=1))

    return torch.stack(res_T, dim=0)
