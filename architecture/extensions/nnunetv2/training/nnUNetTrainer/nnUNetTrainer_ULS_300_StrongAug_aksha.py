import numpy as np
import torch
from typing import List, Tuple, Union
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_ULS_300_StrongAug_aksha(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 2.5e-3
        self.num_epochs = 300

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            do_dummy_2d_data_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15.0 / 360.0 * 2.0 * np.pi, 15.0 / 360.0 * 2.0 * np.pi)
            else:
                rotation_for_DA = (-180.0 / 360.0 * 2.0 * np.pi, 180.0 / 360.0 * 2.0 * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                rotation_for_DA = (-180.0 / 360.0 * 2.0 * np.pi, 180.0 / 360.0 * 2.0 * np.pi)
            else:
                rotation_for_DA = (-45.0 / 360.0 * 2.0 * np.pi, 45.0 / 360.0 * 2.0 * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()
        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.65, 1.6),
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]
        self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales,
                                mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None,
                                is_cascaded=False, foreground_labels=None, regions=None,
                                ignore_label=None):
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=0, random_crop=False,
            p_elastic_deform=0.2, elastic_deform_scale=(0.04, 0.12),
            elastic_deform_magnitude=(8.0, 24.0), p_synchronize_def_scale_across_axes=0,
            p_rotation=0.25, rotation=rotation_for_DA, p_rot_per_axis=1,
            p_scaling=0.25, scaling=(0.65, 1.6), p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False, border_mode_seg="constant", padding_value_seg=-1,
        ))
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        transforms.append(RandomTransform(GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True), apply_probability=0.1))
        transforms.append(RandomTransform(GaussianBlurTransform(blur_sigma=(0.5, 1.0), synchronize_channels=False, synchronize_axes=False, p_per_channel=0.5, benchmark=True), apply_probability=0.2))
        transforms.append(RandomTransform(MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)), synchronize_channels=False, p_per_channel=1), apply_probability=0.15))
        transforms.append(RandomTransform(ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True, synchronize_channels=False, p_per_channel=1), apply_probability=0.15))
        transforms.append(RandomTransform(SimulateLowResolutionTransform(scale=(0.5, 1), synchronize_channels=False, synchronize_axes=True, ignore_axes=ignore_axes, allowed_channels=None, p_per_channel=0.5), apply_probability=0.25))
        transforms.append(RandomTransform(GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=1, synchronize_channels=False, p_per_channel=1, p_retain_stats=1), apply_probability=0.1))
        transforms.append(RandomTransform(GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=0, synchronize_channels=False, p_per_channel=1, p_retain_stats=1), apply_probability=0.3))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]], channel_idx_in_seg=0, set_outside_to=0))
        transforms.append(RemoveLabelTansform(-1, 0))
        if regions is not None:
            transforms.append(ConvertSegmentationToRegionsTransform(regions=list(regions) + [ignore_label] if ignore_label is not None else regions, channel_in_seg=0))
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        print("StrongAug transforms built successfully")
        return ComposeTransforms(transforms)
