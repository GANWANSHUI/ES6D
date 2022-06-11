import torch.utils.data as data
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import random
import math
import scipy.io as scio
import open3d as o3d
import json

class PoseDataset(data.Dataset):

    def __init__(self, mode, num_pt, root, add_noise=False, noise_trans = 0.03):

        self.mode = mode

        if mode == 'train':
            self.path = root + '/train_pbr_mat.txt'
            self.file_path = root + '/train_pbr_mat'

        elif mode == 'test':

            self.path = root + '/test_primesense_gt_mask_mat.txt'
            self.file_path = root + '/test_primesense_gt_mask_mat'

        self.data_augmentation = True
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.list = []
        input_file = open(self.path)

        while 1:

            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]

            self.list.append(input_line)

            # if len(self.list) >= 128:
            #    break

        input_file.close()

        class_file = open('datasets/tless/classes.txt')

        self.cld = []

        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            input_cloud = o3d.io.read_point_cloud('{0}/models/{1}.pcd'.format(self.root, class_input[:-1]))
            raw_xyz = torch.tensor(np.asarray(input_cloud.points).reshape((1, -1, 3)), dtype=torch.float32)
            xyz_ids = farthest_point_sample(raw_xyz, num_pt).cpu().numpy()
            raw_xyz = np.asarray(input_cloud.points).astype(np.float32) * 0.001
            self.cld.append(raw_xyz[xyz_ids[0, :], :])


        self.length = len(self.list)

        self.prim_groups = []
        self.raw_prim_groups = []

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.resize_img_width = 128
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50


        self.norm = transforms.Normalize(mean=[0.485*255.0, 0.456*255.0, 0.406*255.0], std=[0.229*255.0, 0.224*255.0, 0.225*255.0])
        self.RandomErasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

        self.symmetry_obj_idx = [0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   18,   19,   22,   23,   24,   25,   26,   27,   28,   29]  # start with 0
        self.obj_radius = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        self.front_num = 2


        # load primitives
        with open('datasets/tless/tless_gp.json', 'r') as f:
            prim_groups = json.load(f)

            for i, prim in enumerate(prim_groups):

                tmp = []
                raw = []
                for grp in prim['groups']:
                    tmp.append(torch.tensor(grp, dtype=torch.float).permute(1, 0).contiguous() / self.obj_radius[i])
                    raw.append(torch.tensor(grp, dtype=torch.float).permute(1, 0).contiguous())

                self.prim_groups.append(tmp)
                self.raw_prim_groups.append(raw)

        print(f'total data number: {len(self.list)}')


    def __getitem__(self, index):

        meta = scio.loadmat('{0}/{1}.mat'.format(self.file_path, self.list[index]))
        rgb = meta['rgb'].astype(np.float32)
        mask = meta['mask'].astype(np.float32)
        xyz = meta['xyz'].astype(np.float32)
        cls_id = meta['cls_indexes'][0][0] # start with 1

        if np.isnan(xyz).sum()>0:
            print(index)

        if self.mode == 'train':

            noise_xyz = np.random.uniform(-0.01, 0.01, xyz.shape)
            xyz += noise_xyz
            xyz = xyz * mask[:, :, np.newaxis]

        # resize
        rgb, xyz, mask = resize(rgb, xyz, mask, self.resize_img_width, self.resize_img_width)

        # get gt pose
        target_r = meta['poses'][:, 0:3].astype(np.float32)
        target_t = np.array([meta['poses'][:, 3:4].flatten()]).reshape(3, -1).astype(np.float32)

        model_points = self.cld[cls_id - 1].T

        # filter outlier points
        dis_xyz = np.sqrt(xyz[:, :, 0] * xyz[:, :, 0] + xyz[:, :, 1]*xyz[:, :, 1] + xyz[:, :, 2]*xyz[:, :, 2])
        mask_xyz = np.where(dis_xyz > self.obj_radius[cls_id - 1], 0.0, 1.0).astype(np.float32)
        xyz = xyz * mask_xyz[:, :, np.newaxis]
        mask = mask * mask_xyz

        # add noise to xyz and target_t
        if self.mode == 'train':

            noise_t = np.asarray([np.random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)]).astype(np.float32)
            xyz += noise_t
            target_t += noise_t.reshape((3, 1))

            if self.data_augmentation:
                rgb = np.asarray(self.trancolor(Image.fromarray(rgb.astype('uint8')))).astype(np.float32)


        # normalize xyz and target_t
        xyz = xyz / self.obj_radius[cls_id - 1]
        target_t = target_t / self.obj_radius[cls_id-1]
        model_points = model_points / self.obj_radius[cls_id - 1]

        target_xyz = target_r @ model_points + target_t

        rgb = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).contiguous() #/ 255
        xyz = torch.from_numpy(xyz.astype(np.float32)).permute(2, 0, 1).contiguous()

        if(mask.sum() == 0.0):

            mask = np.ones(mask.shape, dtype=np.float32)


        #print(instance_id)

        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0)

        rgb = self.norm(rgb)

        if self.mode == 'test':
              mean_xyz = meta['mean_xyz'].astype(np.float32)
              # get instance id
              instance_id = str(self.list[index])
              # print('instance_id:', instance_id)

              # 000009/000204_000006
              instance_id = np.array([int(instance_id[:6]), int(instance_id[7:13]), int(instance_id[14:])])
              instance_id = torch.from_numpy(instance_id)

        else:
            mean_xyz = torch.from_numpy(np.array([0]))
            instance_id = torch.from_numpy(np.array([0]))


        return {
                'rgb': rgb,
                'xyz': xyz,
                'mask': mask,
                'target_r': torch.from_numpy(target_r.astype(np.float32)).view(3, 3),
                'target_t': torch.from_numpy(target_t.astype(np.float32)).view(3),
                'model_xyz': torch.from_numpy(model_points.astype(np.float32)),
                'class_id': torch.LongTensor([int(cls_id)-1]),
                'target_xyz': target_xyz,
                'instance_id': instance_id,
                'mean_xyz': mean_xyz
                
        } 


    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def resize(rgb, xyz, mask, width, height):
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    xyz = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)

    rgb = F.interpolate(rgb, size=(height, width), mode='bilinear').squeeze(dim=0).permute(1, 2, 0).contiguous()
    xyz = F.interpolate(xyz, size=(height, width), mode='nearest').squeeze(dim=0).permute(1, 2, 0).contiguous()
    mask = F.interpolate(mask, size=(height, width), mode='nearest').squeeze(dim=0).squeeze(dim=0)
    return rgb.cpu().numpy(), xyz.cpu().numpy(), mask.cpu().numpy()

def random_rotation_translation(rgb, xyz, mask, degree_range, trans_range):
    h, w, c = rgb.shape
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    xyz = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)

    angle = float(random.uniform(-degree_range, degree_range)) * math.pi / 180.0
    trans1 = random.choice([float(random.uniform(trans_range[0], trans_range[1])), -float(random.uniform(trans_range[0], trans_range[1]))])
    trans2 = random.choice([float(random.uniform(trans_range[0], trans_range[1])), -float(random.uniform(trans_range[0], trans_range[1]))])

    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), trans1],
        [math.sin(angle), math.cos(angle), trans2]
    ], dtype=torch.float)

    grid = F.affine_grid(theta.unsqueeze(0), rgb.size())
    rgb = F.grid_sample(rgb, grid).squeeze(dim=0).permute(1, 2, 0).contiguous()
    xyz = F.grid_sample(xyz, grid).squeeze(dim=0).permute(1, 2, 0).contiguous()
    mask = F.grid_sample(mask, grid, mode='nearest').squeeze(dim=0).squeeze(dim=0)

    return rgb.cpu().numpy(), xyz.cpu().numpy(), mask.cpu().numpy()

def paste_two_objects(f_rgb, f_xyz, f_mask, b_rgb, b_xyz, b_mask):
    mask = b_mask - b_mask * f_mask
    # both_mask = mask + f_mask
    rgb = b_rgb * (1 - f_mask[:, :, np.newaxis]) + f_rgb * f_mask[:, :, np.newaxis]
    xyz = b_xyz * (1 - f_mask[:, :, np.newaxis]) + f_xyz * f_mask[:, :, np.newaxis]
    return rgb, xyz, mask

def augmentation(rgb, xyz, mask, root, syn_list, real_list, real=True, addfront_p = 0.75, blur_p=0.75):
    h, w, c = rgb.shape
    min_numpoint = h * w / 20.0
    if(random.uniform(0.0, 1.0) < addfront_p):
        if (real):
            seed = random.choice(syn_list)
        else:
            seed = random.choice(real_list)
        # syn_rgb = np.asarray(Image.open('{0}/{1}.png'.format(root, seed))).astype(np.float32)
        # syn_mask = np.asarray(Image.open('{0}/{1}_mask.png'.format(root, seed))).astype(np.float32)
        syn_meta = scio.loadmat('{0}/{1}.mat'.format(root, seed))
        syn_rgb = syn_meta['rgb'].astype(np.float32)
        syn_mask = syn_meta['mask'].astype(np.float32)

        if(syn_mask.sum() > min_numpoint):
            # syn_meta = scio.loadmat('{0}/{1}.mat'.format(root, seed))
            syn_xyz = syn_meta['xyz'].astype(np.float32)

            # resize
            syn_rgb, syn_xyz, syn_mask = resize(syn_rgb, syn_xyz, syn_mask, w, h)
            z_offset = 0.05 + float(random.uniform(0.0, 0.1))
            syn_xyz[:, :, 2] = syn_xyz[:, :, 2] + z_offset

            for i in range(5):
                # rotate and translate syn_obj
                syn_rgb1, syn_xyz1, syn_mask1 = random_rotation_translation(syn_rgb, syn_xyz, syn_mask, 90, [0.4, 0.8])
                # paste synthesized object in front of real object
                new_rgb, new_xyz, new_mask = paste_two_objects(syn_rgb1, syn_xyz1, syn_mask1, rgb, xyz, mask)
                if (new_mask.sum() / mask.sum() > 0.3 and new_mask.sum() > w * h / 20):
                    rgb, xyz, mask = new_rgb, new_xyz, new_mask
                    break

    if(real==False):
        seed = random.choice(real_list)
        # syn_rgb = np.asarray(Image.open('{0}/{1}.png'.format(root, seed))).astype(np.float32)
        # syn_mask = np.asarray(Image.open('{0}/{1}_mask.png'.format(root, seed))).astype(np.float32)
        # syn_meta = scio.loadmat('{0}/{1}.mat'.format(root, seed))
        # syn_xyz = syn_meta['xyz'].astype(np.float32)
        syn_meta = scio.loadmat('{0}/{1}.mat'.format(root, seed))
        syn_rgb = syn_meta['rgb'].astype(np.float32)
        syn_mask = syn_meta['mask'].astype(np.float32)
        syn_xyz = syn_meta['xyz'].astype(np.float32)
        syn_rgb, syn_xyz, syn_mask = resize(syn_rgb, syn_xyz, syn_mask, w, h)

        z_offset = -0.05 - float(random.uniform(0.0, 0.1))
        syn_xyz[:, :, 2] = syn_xyz[:, :, 2] + z_offset
        back_mask = (rgb[:, :, 0] == 0).astype(np.float32)
        front_mask = (rgb[:, :, 0] > 0).astype(np.float32)
        rgb, xyz, _ = paste_two_objects(rgb, xyz, front_mask, syn_rgb, syn_xyz, back_mask)
        # randomly blur
        if (random.uniform(0.0, 1.0) < blur_p):
            rgb = Image.fromarray(rgb.astype('uint8')).convert('RGB')
            rgb = rgb.filter(ImageFilter.BoxBlur(random.choice([1])))
            rgb = np.asarray(rgb).astype(np.float32)

    return rgb, xyz, mask