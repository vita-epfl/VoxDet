import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES


def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[..., 0]**2 + input_xyz[..., 1]**2)
    phi = torch.atan2(input_xyz[..., 1], input_xyz[..., 0])
    return torch.stack((rho, phi, input_xyz[..., 2]), axis=-1)


@PIPELINES.register_module()
class LoadLidarPointsFromFiles_SemanticKitti(object):

    def __init__(self, data_config, is_train=False):
        super().__init__()

        self.is_train = is_train
        self.data_config = data_config

    def get_inputs(self, results):
        lidar_filenames = results['lidar_filename']
        data_lists = []

        for i in range(len(lidar_filenames)):
            lidar_filename = lidar_filenames[i]
            lidar_points = torch.tensor(np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4))

            result = [lidar_points]
            result = [x[None] for x in result]

            data_lists.append(result)

        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))

        return result_list

    def __call__(self, results):
        results['lidar_inputs'] = self.get_inputs(results)

        return results


@PIPELINES.register_module()
class LidarPointsPreProcess_SemanticKitti(object):

    def __init__(
        self,
        data_config,
        point_cloud_range,
        occ_size,
        coarse_ratio=2,
        is_train=False,
    ):
        super().__init__()

        self.is_train = is_train
        self.data_config = data_config

        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.occ_size = torch.tensor(occ_size, dtype=torch.int32)

        self.grid_size = (self.occ_size / coarse_ratio).to(torch.int32)

        self.voxel_len = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.occ_size
        self.grid_len = self.voxel_len * coarse_ratio

        self.min_bound = self.point_cloud_range[:3]
        self.max_bound = self.point_cloud_range[3:]

        # get voxel_position_grid_coarse
        x_indices, y_indices, z_indices = torch.meshgrid(
            torch.arange(self.grid_size[0]),
            torch.arange(self.grid_size[1]),
            torch.arange(self.grid_size[2]),
            indexing='ij',
        )
        coords = torch.stack((x_indices, y_indices, z_indices), dim=-1)
        self.grid_position = ((coords.to(torch.float32) + 0.5) * self.grid_len + self.min_bound).view(-1, 3)

    def get_inputs(self, results):
        lidar_points = results['lidar_inputs']
        point_lists = []
        voxel_pos_lists = []
        grid_ind_lists = []

        for i in range(len(lidar_points)):
            points = lidar_points[i]
            xyz, feat = points[..., :3], points[..., 3:]
            mask_x = (xyz[:, :, 0] > self.point_cloud_range[0]) & (xyz[:, :, 0] < self.point_cloud_range[3])
            mask_y = (xyz[:, :, 1] > self.point_cloud_range[1]) & (xyz[:, :, 1] < self.point_cloud_range[4])
            mask_z = (xyz[:, :, 2] > self.point_cloud_range[2]) & (xyz[:, :, 2] < self.point_cloud_range[5])
            mask = mask_x & mask_y & mask_z
            xyz = xyz[:, mask[0], :]
            feat = feat[:, mask[0], :]
            grid_ind = torch.floor((xyz - self.min_bound) / self.grid_len).to(torch.int32)
            voxel_centers = (grid_ind.to(torch.float32) + 0.5) * self.grid_len + self.min_bound
            return_xyz = xyz - voxel_centers
            return_feat = torch.cat((return_xyz, xyz[..., :2], feat), dim=-1)

            point_lists.append(return_feat)
            voxel_pos_lists.append(self.grid_position)
            grid_ind_lists.append(grid_ind)

        return point_lists, voxel_pos_lists, grid_ind_lists

    def __call__(self, results):
        results['points'], results['voxel_position_grid_coarse'], results['grid_ind'] = self.get_inputs(results)

        return results
