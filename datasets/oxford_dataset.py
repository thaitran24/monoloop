# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import Image

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
from .mono_pair_dataset import MonoPairDataset


class OxfordDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(OxfordDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1242, 375)
        self.K = np.array([[0.768, 0, 0.5, 0],
                           [0, 1.024, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # 983.044006, 0, 643.646973 / 1280
        # 0, 983.044006, 493.378998 / 960
        self.full_res_shape = (1280, 640)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class OxfordRawDataset(OxfordDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(OxfordRawDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        depth_path = os.path.join(self.data_path, folder+'_gt', f_str)

        img_file = Image.open(depth_path)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png), depth_path)
        # print(np.min(depth_png), ' ',np.max(depth_png))

        depth_gt = depth_png.astype(np.float) / 256.
        # depth[depth_png == 0] = -1.
        # depth = np.expand_dims(depth, -1)

        # depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])

        depth_gt = depth_gt[160:960-160,:]

        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class OxfordPairDataset(MonoPairDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(OxfordPairDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1242, 375)
        self.K = np.array([[0.768, 0, 0.5, 0],
                           [0, 1.024, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # 983.044006, 0, 643.646973 / 1280
        # 0, 983.044006, 493.378998 / 960
        self.full_res_shape = (1280, 640)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class OxfordRawPairDataset(OxfordPairDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(OxfordRawPairDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        depth_path = os.path.join(self.data_path, folder+'_gt', f_str)

        img_file = Image.open(depth_path)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png), depth_path)
        # print(np.min(depth_png), ' ',np.max(depth_png))

        depth_gt = depth_png.astype(np.float) / 256.
        # depth[depth_png == 0] = -1.
        # depth = np.expand_dims(depth, -1)

        # depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])

        depth_gt = depth_gt[160:960-160,:]

        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt