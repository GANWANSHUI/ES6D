import numpy as np
import torch
from . import point_grouper as pg
from . import mean_shift_utils as ms_utils

MIN_DISTANCE = 0.001


class MeanShift(object):
    def __init__(self, kernel=ms_utils.gaussian_kernel):
        if kernel == 'multivariate_gaussian':
            kernel = ms_utils.multivariate_gaussian_kernel
        self.kernel = kernel

    def cluster(self, points, kernel_bandwidth, iteration_callback=None):
        if(iteration_callback):
            iteration_callback(points, 0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, kernel_bandwidth)
                dist = ms_utils.euclidean_dist(p_new, p_new_start)
                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
        point_grouper = pg.PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())
        return MeanShiftResult(points, shift_points, group_assignments)

    def _shift_point(self, point, points, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift
        points = np.array(points)

        # numerator
        point_weights = self.kernel(point-points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        # denominator
        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        return shifted_point

        # ***************************************************************************
        # ** The above vectorized code is equivalent to the unrolled version below **
        # ***************************************************************************
        # shift_x = float(0)
        # shift_y = float(0)
        # scale_factor = float(0)
        # for p_temp in points:
        #     # numerator
        #     dist = ms_utils.euclidean_dist(point, p_temp)
        #     weight = self.kernel(dist, kernel_bandwidth)
        #     shift_x += p_temp[0] * weight
        #     shift_y += p_temp[1] * weight
        #     # denominator
        #     scale_factor += weight
        # shift_x = shift_x / scale_factor
        # shift_y = shift_y / scale_factor
        # return [shift_x, shift_y]


class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids


class MeanShift_wi_Dense(object):
    def __init__(self):
        self.kernel = ms_utils.dense_kernel

    def cluster(self, samples, points, pts_dense, kernel_bandwidth, iteration_callback=None):
        if(iteration_callback):
            iteration_callback(points, 0)
        shift_points = torch.tensor(samples).float().cuda()
        points = torch.tensor(points).float().cuda()
        pts_dense = torch.tensor(pts_dense).float().cuda()
        max_min_dist = 1
        iteration_number = 0

        still_shifting = [True] * shift_points.size()[0]
        while max_min_dist > MIN_DISTANCE:
            print(iteration_number, max_min_dist)
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, shift_points.size()[0]):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, pts_dense,  kernel_bandwidth)
                dist = torch.norm(p_new-p_new_start).item()
                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
        shift_points = shift_points.cpu().numpy()
        point_grouper = pg.PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())
        return MeanShiftResult(samples, shift_points, group_assignments)

    def _shift_point(self, point, points, pts_dense, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift

        # numerator
        point_weights = self.kernel(point-points, pts_dense, kernel_bandwidth)
        _, mid = torch.max(point_weights, dim=0)
        shifted_point = points[mid]
        return shifted_point

        # ***************************************************************************
        # ** The above vectorized code is equivalent to the unrolled version below **
        # ***************************************************************************
        # shift_x = float(0)
        # shift_y = float(0)
        # scale_factor = float(0)
        # for p_temp in points:
        #     # numerator
        #     dist = ms_utils.euclidean_dist(point, p_temp)
        #     weight = self.kernel(dist, kernel_bandwidth)
        #     shift_x += p_temp[0] * weight
        #     shift_y += p_temp[1] * weight
        #     # denominator
        #     scale_factor += weight
        # shift_x = shift_x / scale_factor
        # shift_y = shift_y / scale_factor
        # return [shift_x, shift_y]

