from typing import Optional
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmpose.registry import TRANSFORMS
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union
import mmcv
import mmengine
from mmcv.image import imflip
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine import is_list_of
from mmengine.dist import get_dist_info
from scipy.stats import truncnorm
from mmpose.codecs import *  
from mmpose.registry import KEYPOINT_CODECS, TRANSFORMS
from mmpose.structures.bbox import bbox_xyxy2cs, flip_bbox
from mmpose.structures.keypoint import flip_keypoints
from mmpose.utils.typing import MultiConfig
from typing import Dict, Optional, Tuple
import cv2
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
from typing import Sequence, Union
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_seq_of
from mmpose.registry import TRANSFORMS
from mmpose.structures import MultilevelPixelData, PoseDataSample
from mmpose.datasets.transforms.formatting import image_to_tensor, keypoints_to_tensor
from mmpose.models.pose_estimators.base import BasePoseEstimator
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
import copy
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from mmpose.registry import MODELS
from mmpose.models.backbones.base_backbone import BaseBackbone
from mmpose.models.backbones.resnet import BasicBlock, Bottleneck, get_expansion
from mmpose.models.backbones.hrnet import HRModule
import torch.nn.functional as F
import numbers
from einops import rearrange
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn
from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from mmpose.models.heads.base_head import BaseHead
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from mmengine.dataset import BaseDataset, force_full_init
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.model import BaseModel

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.utils import check_and_update_config
from mmpose.utils.typing import (ConfigType, ForwardResults, OptConfigType,
                                 Optional, OptMultiConfig, OptSampleList,
                                 SampleList)
from itertools import zip_longest
from torch import Tensor
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from mmengine.runner import load_checkpoint
from typing import List, Optional, Sequence, Union
from mmengine.model import ImgDataPreprocessor
from mmpose.models.data_preprocessors import PoseDataPreprocessor
from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead
from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmpose.models.backbones.hrnet import HRNet
from mmengine.config import Config


@TRANSFORMS.register_module()
class LoadImagePair(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            results_ll, results_wl = results[0], results[1]
            if 'img' not in results_ll:
                 # Load image from file by :meth:`LoadImageFromFile.transform`
                results_ll = super().transform(results_ll)
            else:
                img = results_ll['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results_ll:
                    results_ll['img_path'] = None
                results_ll['img_shape'] = img.shape[:2]
                results_ll['ori_shape'] = img.shape[:2]
            
            if 'img' not in results_wl:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results_wl = super().transform(results_wl)
            else:
                img = results_wl['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results_wl:
                    results_wl['img_path'] = None
                results_wl['img_shape'] = img.shape[:2]
                results_wl['ori_shape'] = img.shape[:2]
                
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return [results_ll, results_wl]

@TRANSFORMS.register_module()
class GetBBoxCenterScalePair(BaseTransform):
    """Convert bboxes from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()

        self.padding = padding

    def transform_one(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if 'bbox_center' in results and 'bbox_scale' in results:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('Use the existing "bbox_center" and "bbox_scale"'
                              '. The padding will still be applied.')
            results['bbox_scale'] = results['bbox_scale'] * self.padding

        else:
            bbox = results['bbox']
            center, scale = bbox_xyxy2cs(bbox, padding=self.padding)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results
    
    def transform(self, results):
        results_ll, results_wl = results[0], results[1]
        return [self.transform_one(results_ll), self.transform_one(results_wl)]

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str
    
@TRANSFORMS.register_module()
class RandomFlipPair(BaseTransform):
    """Randomly flip the image, bbox and keypoints.

    Required Keys:

        - img
        - img_shape
        - flip_indices
        - input_size (optional)
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - img_mask (optional)

    Modified Keys:

        - img
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - img_mask (optional)

    Added Keys:

        - flip
        - flip_direction

    Args:
        prob (float | list[float]): The flipping probability. If a list is
            given, the argument `direction` should be a list with the same
            length. And each element in `prob` indicates the flipping
            probability of the corresponding one in ``direction``. Defaults
            to 0.5
        direction (str | list[str]): The flipping direction. Options are
            ``'horizontal'``, ``'vertical'`` and ``'diagonal'``. If a list is
            is given, each data sample's flipping direction will be sampled
            from a distribution determined by the argument ``prob``. Defaults
            to ``'horizontal'``.
    """

    def __init__(self,
                 prob: Union[float, List[float]] = 0.5,
                 direction: Union[str, List[str]] = 'horizontal') -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      List) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def transform_one(self, results: dict, flip_dir) -> dict:
        """The transform function of :class:`RandomFlip`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        if flip_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = flip_dir

            h, w = results.get('input_size', results['img_shape'])
            # flip image and mask
            if isinstance(results['img'], list):
                results['img'] = [
                    imflip(img, direction=flip_dir) for img in results['img']
                ]
            else:
                results['img'] = imflip(results['img'], direction=flip_dir)

            if 'img_mask' in results:
                results['img_mask'] = imflip(
                    results['img_mask'], direction=flip_dir)

            # flip bboxes
            if results.get('bbox', None) is not None:
                results['bbox'] = flip_bbox(
                    results['bbox'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)

            if results.get('bbox_center', None) is not None:
                results['bbox_center'] = flip_bbox(
                    results['bbox_center'],
                    image_size=(w, h),
                    bbox_format='center',
                    direction=flip_dir)

            # flip keypoints
            if results.get('keypoints', None) is not None:
                keypoints, keypoints_visible = flip_keypoints(
                    results['keypoints'],
                    results.get('keypoints_visible', None),
                    image_size=(w, h),
                    flip_indices=results['flip_indices'],
                    direction=flip_dir)

                results['keypoints'] = keypoints
                results['keypoints_visible'] = keypoints_visible

        return results
    
    def transform(self, results):
        results_ll, results_wl = results[0], results[1]
        flip_dir = self._choose_direction()
        return [self.transform_one(results_ll, flip_dir), self.transform_one(results_wl, flip_dir)]

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'
        return repr_str
    
@TRANSFORMS.register_module()
class RandomHalfBodyPair(BaseTransform):
    """Data augmentation with half-body transform that keeps only the upper or
    lower body at random.

    Required Keys:

        - keypoints
        - keypoints_visible
        - upper_body_ids
        - lower_body_ids

    Modified Keys:

        - bbox
        - bbox_center
        - bbox_scale

    Args:
        min_total_keypoints (int): The minimum required number of total valid
            keypoints of a person to apply half-body transform. Defaults to 8
        min_half_keypoints (int): The minimum required number of valid
            half-body keypoints of a person to apply half-body transform.
            Defaults to 2
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 min_total_keypoints: int = 9,
                 min_upper_keypoints: int = 2,
                 min_lower_keypoints: int = 3,
                 padding: float = 1.5,
                 prob: float = 0.3,
                 upper_prioritized_prob: float = 0.7) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.padding = padding
        self.prob = prob
        self.upper_prioritized_prob = upper_prioritized_prob

    def _get_half_body_bbox(self, keypoints: np.ndarray,
                            half_body_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, D)
            upper_body_ids (list): The list of half-body keypont indices

        Returns:
            tuple: A tuple containing half-body bbox center and scale
            - center: Center (x, y) of the bbox
            - scale: Scale (w, h) of the bbox
        """

        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]

        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        scale = np.array([w, h], dtype=center.dtype) * self.padding

        return center, scale

    @cache_randomness
    def _random_select_half_body(self, keypoints_visible: np.ndarray,
                                 upper_body_ids: List[int],
                                 lower_body_ids: List[int]
                                 ) -> List[Optional[List[int]]]:
        """Randomly determine whether applying half-body transform and get the
        half-body keyponit indices of each instances.

        Args:
            keypoints_visible (np.ndarray, optional): The visibility of
                keypoints in shape (N, K, 1) or (N, K, 2).
            upper_body_ids (list): The list of upper body keypoint indices
            lower_body_ids (list): The list of lower body keypoint indices

        Returns:
            list[list[int] | None]: The selected half-body keypoint indices
            of each instance. ``None`` means not applying half-body transform.
        """

        if keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]

        half_body_ids = []

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                upper_valid_ids = [i for i in upper_body_ids if visible[i] > 0]
                lower_valid_ids = [i for i in lower_body_ids if visible[i] > 0]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                prefer_upper = np.random.rand() < self.upper_prioritized_prob
                if (num_upper < self.min_upper_keypoints
                        and num_lower < self.min_lower_keypoints):
                    indices = None
                elif num_lower < self.min_lower_keypoints:
                    indices = upper_valid_ids
                elif num_upper < self.min_upper_keypoints:
                    indices = lower_valid_ids
                else:
                    indices = (
                        upper_valid_ids if prefer_upper else lower_valid_ids)

            half_body_ids.append(indices)

        return half_body_ids

    def transform_one(self, results: Dict, half_body_ids) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        bbox_center = []
        bbox_scale = []

        for i, indices in enumerate(half_body_ids):
            if indices is None:
                bbox_center.append(results['bbox_center'][i])
                bbox_scale.append(results['bbox_scale'][i])
            else:
                _center, _scale = self._get_half_body_bbox(
                    results['keypoints'][i], indices)
                bbox_center.append(_center)
                bbox_scale.append(_scale)

        results['bbox_center'] = np.stack(bbox_center)
        results['bbox_scale'] = np.stack(bbox_scale)
        return results
    
    def transform(self, results):
        results_ll, results_wl = results[0], results[1]
        half_body_ids = self._random_select_half_body(
            results_ll['keypoints_visible'],
            results_ll['upper_body_ids'],
            results_ll['lower_body_ids'])
        return [self.transform_one(results_ll, half_body_ids), self.transform_one(results_wl, half_body_ids)]

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(min_total_keypoints={self.min_total_keypoints}, '
        repr_str += f'min_upper_keypoints={self.min_upper_keypoints}, '
        repr_str += f'min_lower_keypoints={self.min_lower_keypoints}, '
        repr_str += f'padding={self.padding}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'upper_prioritized_prob={self.upper_prioritized_prob})'
        return repr_str
    
@TRANSFORMS.register_module()
class RandomBBoxTransformPair(BaseTransform):
    r"""Rnadomly shift, resize and rotate the bounding boxes.

    Required Keys:

        - bbox_center
        - bbox_scale

    Modified Keys:

        - bbox_center
        - bbox_scale

    Added Keys:
        - bbox_rotation

    Args:
        shift_factor (float): Randomly shift the bbox in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = x(y)_scale \cdot shift_factor` in pixels.
            Defaults to 0.16
        shift_prob (float): Probability of applying random shift. Defaults to
            0.3
        scale_factor (Tuple[float, float]): Randomly resize the bbox in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to (0.5, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-rotate_factor, rotate_factor]` in degrees. Defaults
            to 80.0
        rotate_prob (float): Probability of applying random rotation. Defaults
            to 0.6
    """

    def __init__(self,
                 shift_factor: float = 0.16,
                 shift_prob: float = 0.3,
                 scale_factor: Tuple[float, float] = (0.5, 1.5),
                 scale_prob: float = 1.0,
                 rotate_factor: float = 80.0,
                 rotate_prob: float = 0.6) -> None:
        super().__init__()

        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

    @staticmethod
    def _truncnorm(low: float = -1.,
                   high: float = 1.,
                   size: tuple = ()) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    @cache_randomness
    def _get_transform_params(self, num_bboxes: int) -> Tuple:
        """Get random transform parameters.

        Args:
            num_bboxes (int): The number of bboxes

        Returns:
            tuple:
            - offset (np.ndarray): Offset factor of each bbox in shape (n, 2)
            - scale (np.ndarray): Scaling factor of each bbox in shape (n, 1)
            - rotate (np.ndarray): Rotation degree of each bbox in shape (n,)
        """
        random_v = self._truncnorm(size=(num_bboxes, 4))
        offset_v = random_v[:, :2]
        scale_v = random_v[:, 2:3]
        rotate_v = random_v[:, 3]

        # Get shift parameters
        offset = offset_v * self.shift_factor
        offset = np.where(
            np.random.rand(num_bboxes, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = scale_v * sigma + mu
        scale = np.where(
            np.random.rand(num_bboxes, 1) < self.scale_prob, scale, 1.)

        # Get rotation parameters
        rotate = rotate_v * self.rotate_factor
        rotate = np.where(
            np.random.rand(num_bboxes) < self.rotate_prob, rotate, 0.)

        return offset, scale, rotate

    def transform_one(self, results: Dict, offset, scale, rotate) -> Optional[dict]:
        """The transform function of :class:`RandomBboxTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        bbox_scale = results['bbox_scale']
        num_bboxes = bbox_scale.shape[0]

        results['bbox_center'] = results['bbox_center'] + offset * bbox_scale
        results['bbox_scale'] = results['bbox_scale'] * scale
        results['bbox_rotation'] = rotate

        return results
    
    def transform(self, results):
        results_ll, results_wl = results[0], results[1]
        offset, scale, rotate = self._get_transform_params(
            results_ll['bbox_scale'].shape[0])
        return [self.transform_one(results_ll, offset, scale, rotate), self.transform_one(results_wl, offset, scale, rotate)]

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(shift_prob={self.shift_prob}, '
        repr_str += f'shift_factor={self.shift_factor}, '
        repr_str += f'scale_prob={self.scale_prob}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'rotate_prob={self.rotate_prob}, '
        repr_str += f'rotate_factor={self.rotate_factor})'
        return repr_str
    
@TRANSFORMS.register_module()
class GenerateTargetPair(BaseTransform):
    """Encode keypoints into Target.

    The generated target is usually the supervision signal of the model
    learning, e.g. heatmaps or regression labels.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys:

        - The keys of the encoded items from the codec will be updated into
            the results, e.g. ``'heatmaps'`` or ``'keypoint_weights'``. See
            the specific codec for more details.

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding.
            Both single encoder and multiple encoders (given as a list) are
            supported
        multilevel (bool): Determine the method to handle multiple encoders.
            If ``multilevel==True``, generate multilevel targets from a group
            of encoders of the same type (e.g. multiple :class:`MSRAHeatmap`
            encoders with different sigma values); If ``multilevel==False``,
            generate combined targets from a group of different encoders. This
            argument will have no effect in case of single encoder. Defaults
            to ``False``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
        target_type (str, deprecated): This argument is deprecated and has no
            effect. Defaults to ``None``
    """

    def __init__(self,
                 encoder: MultiConfig,
                 encoder_heatmap = None,
                 target_type: Optional[str] = None,
                 multilevel: bool = False,
                 use_dataset_keypoint_weights: bool = False) -> None:
        super().__init__()

        if target_type is not None:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn(
                    'The argument `target_type` is deprecated in'
                    ' GenerateTarget. The target type and encoded '
                    'keys will be determined by encoder(s).',
                    DeprecationWarning)

        self.encoder_cfg = deepcopy(encoder)
        self.multilevel = multilevel
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

        if isinstance(self.encoder_cfg, list):
            self.encoder = [
                KEYPOINT_CODECS.build(cfg) for cfg in self.encoder_cfg
            ]
        else:
            assert not self.multilevel, (
                'Need multiple encoder configs if ``multilevel==True``')
            self.encoder = KEYPOINT_CODECS.build(self.encoder_cfg)
            
        if encoder_heatmap is not None:
            self.encoder_heatmap = KEYPOINT_CODECS.build(encoder_heatmap)
        else :
            self.encoder_heatmap = None

    def transform_one(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GenerateTarget`.

        See ``transform()`` method of :class:`BaseTransform` for details.
        """

        if results.get('transformed_keypoints', None) is not None:
            # use keypoints transformed by TopdownAffine
            keypoints = results['transformed_keypoints']
        elif results.get('keypoints', None) is not None:
            # use original keypoints
            keypoints = results['keypoints']
        else:
            raise ValueError(
                'GenerateTarget requires \'transformed_keypoints\' or'
                ' \'keypoints\' in the results.')

        keypoints_visible = results['keypoints_visible']
        if keypoints_visible.ndim == 3 and keypoints_visible.shape[2] == 2:
            keypoints_visible, keypoints_visible_weights = \
                keypoints_visible[..., 0], keypoints_visible[..., 1]
            results['keypoints_visible'] = keypoints_visible
            results['keypoints_visible_weights'] = keypoints_visible_weights

        # Encoded items from the encoder(s) will be updated into the results.
        # Please refer to the document of the specific codec for details about
        # encoded items.
        if not isinstance(self.encoder, list):
            # For single encoding, the encoded items will be directly added
            # into results.
            auxiliary_encode_kwargs = {
                key: results.get(key, None)
                for key in self.encoder.auxiliary_encode_keys
            }
            encoded = self.encoder.encode(
                keypoints=keypoints,
                keypoints_visible=keypoints_visible,
                **auxiliary_encode_kwargs)
            
            if self.encoder_heatmap is not None:
                encoded_heatmap = self.encoder_heatmap.encode(
                    keypoints=keypoints,
                    keypoints_visible=keypoints_visible,
                    **auxiliary_encode_kwargs)
                encoded['heatmaps_pixel_weights'] = encoded_heatmap['heatmaps']
            else:
                encoded['heatmaps_pixel_weights'] = None

            if self.encoder.field_mapping_table:
                encoded[
                    'field_mapping_table'] = self.encoder.field_mapping_table
            if self.encoder.instance_mapping_table:
                encoded['instance_mapping_table'] = \
                    self.encoder.instance_mapping_table
            if self.encoder.label_mapping_table:
                encoded[
                    'label_mapping_table'] = self.encoder.label_mapping_table

        else:
            encoded_list = []
            _field_mapping_table = dict()
            _instance_mapping_table = dict()
            _label_mapping_table = dict()
            for _encoder in self.encoder:
                auxiliary_encode_kwargs = {
                    key: results[key]
                    for key in _encoder.auxiliary_encode_keys
                }
                encoded_list.append(
                    _encoder.encode(
                        keypoints=keypoints,
                        keypoints_visible=keypoints_visible,
                        **auxiliary_encode_kwargs))

                _field_mapping_table.update(_encoder.field_mapping_table)
                _instance_mapping_table.update(_encoder.instance_mapping_table)
                _label_mapping_table.update(_encoder.label_mapping_table)

            if self.multilevel:
                # For multilevel encoding, the encoded items from each encoder
                # should have the same keys.

                keys = encoded_list[0].keys()
                if not all(_encoded.keys() == keys
                           for _encoded in encoded_list):
                    raise ValueError(
                        'Encoded items from all encoders must have the same '
                        'keys if ``multilevel==True``.')

                encoded = {
                    k: [_encoded[k] for _encoded in encoded_list]
                    for k in keys
                }

            else:
                # For combined encoding, the encoded items from different
                # encoders should have no overlapping items, except for
                # `keypoint_weights`. If multiple `keypoint_weights` are given,
                # they will be multiplied as the final `keypoint_weights`.

                encoded = dict()
                keypoint_weights = []

                for _encoded in encoded_list:
                    for key, value in _encoded.items():
                        if key == 'keypoint_weights':
                            keypoint_weights.append(value)
                        elif key not in encoded:
                            encoded[key] = value
                        else:
                            raise ValueError(
                                f'Overlapping item "{key}" from multiple '
                                'encoders, which is not supported when '
                                '``multilevel==False``')

                if keypoint_weights:
                    encoded['keypoint_weights'] = keypoint_weights

            if _field_mapping_table:
                encoded['field_mapping_table'] = _field_mapping_table
            if _instance_mapping_table:
                encoded['instance_mapping_table'] = _instance_mapping_table
            if _label_mapping_table:
                encoded['label_mapping_table'] = _label_mapping_table

        if self.use_dataset_keypoint_weights and 'keypoint_weights' in encoded:
            if isinstance(encoded['keypoint_weights'], list):
                for w in encoded['keypoint_weights']:
                    w = w * results['dataset_keypoint_weights']
            else:
                encoded['keypoint_weights'] = encoded[
                    'keypoint_weights'] * results['dataset_keypoint_weights']

        if self.encoder_heatmap is not None:
            encoded['field_mapping_table'].update({"heatmaps_pixel_weights":"heatmaps_pixel_weights"})
        results.update(encoded)
        return results
    
    def transform(self, results):
        results_ll, results_wl = results[0], results[1]
        return [self.transform_one(results_ll), self.transform_one(results_wl)]

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(encoder={str(self.encoder_cfg)}, ')
        repr_str += ('use_dataset_keypoint_weights='
                     f'{self.use_dataset_keypoint_weights})')
        return repr_str
    
@TRANSFORMS.register_module()
class ScalingLLPair(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform_one(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        image_LL = results['img']
        rgb_mean_LL = np.mean(image_LL, axis=(0, 1))
        scaling_LL = 255 * 0.4 / rgb_mean_LL
        results['img'] = image_LL * scaling_LL
        return results
    
    def transform(self, results):
        results_ll, results_wl = results[0], results[1]
        return [self.transform_one(results_ll), results_wl]

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str

@TRANSFORMS.register_module()
class TopdownAffinePair(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform_one(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            if results.get('transformed_keypoints', None) is not None:
                transformed_keypoints = results['transformed_keypoints'].copy()
            else:
                transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints
        else:
            results['transformed_keypoints'] = np.zeros([])
            results['keypoints_visible'] = np.ones((1, 1, 1))

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale

        return results

    def transform(self, results):
        results_ll, results_wl = results
        results_ll = self.transform_one(results_ll)
        results_wl = self.transform_one(results_wl)

        return [results_ll, results_wl]
    
    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str

@TRANSFORMS.register_module()
class PackPoseInputsPair(BaseTransform):
    """Pack the inputs data for pose estimation.

    The ``img_meta`` item is always populated. The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default it includes:

        - ``id``: id of the data sample

        - ``img_id``: id of the image

        - ``'category_id'``: the id of the instance category

        - ``img_path``: path to the image file

        - ``crowd_index`` (optional): measure the crowding level of an image,
            defined in CrowdPose dataset

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``input_size``: the input size to the network

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

        - ``flip_indices``: the indices of each keypoint's symmetric keypoint

        - ``raw_ann_info`` (optional): raw annotation of the instance(s)

    Args:
        meta_keys (Sequence[str], optional): Meta keys which will be stored in
            :obj: `PoseDataSample` as meta info. Defaults to ``('id',
            'img_id', 'img_path', 'category_id', 'crowd_index, 'ori_shape',
            'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
            'flip_direction', 'flip_indices', 'raw_ann_info')``
    """

    # items in `instance_mapping_table` will be directly packed into
    # PoseDataSample.gt_instances without converting to Tensor
    instance_mapping_table = dict(
        bbox='bboxes',
        bbox_score='bbox_scores',
        keypoints='keypoints',
        keypoints_cam='keypoints_cam',
        keypoints_visible='keypoints_visible',
        # In CocoMetric, the area of predicted instances will be calculated
        # using gt_instances.bbox_scales. To unsure correspondence with
        # previous version, this key is preserved here.
        bbox_scale='bbox_scales',
        # `head_size` is used for computing MpiiPCKAccuracy metric,
        # namely, PCKh
        head_size='head_size',
    )

    # items in `field_mapping_table` will be packed into
    # PoseDataSample.gt_fields and converted to Tensor. These items will be
    # used for computing losses
    field_mapping_table = dict(
        heatmaps='heatmaps',
        instance_heatmaps='instance_heatmaps',
        heatmap_mask='heatmap_mask',
        heatmap_weights='heatmap_weights',
        displacements='displacements',
        displacement_weights='displacement_weights',)


    label_mapping_table = dict(
        keypoint_labels='keypoint_labels',
        keypoint_weights='keypoint_weights',
        keypoints_visible_weights='keypoints_visible_weights')

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'category_id',
                            'crowd_index', 'ori_shape', 'img_shape',
                            'input_size', 'input_center', 'input_scale',
                            'flip', 'flip_direction', 'flip_indices',
                            'raw_ann_info', 'dataset_name'),
                 pack_transformed=False):
        self.meta_keys = meta_keys
        self.pack_transformed = pack_transformed

    def transform_one(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`PoseDataSample`): The annotation info of the
                sample.
        """
        # Pack image(s) for 2d pose estimation
        if 'img' in results:
            img = results['img']
            inputs_tensor = image_to_tensor(img)
        # Pack keypoints for 3d pose-lifting
        elif 'lifting_target' in results and 'keypoints' in results:
            if 'keypoint_labels' in results:
                keypoints = results['keypoint_labels']
            else:
                keypoints = results['keypoints']
            inputs_tensor = keypoints_to_tensor(keypoints)

        data_sample = PoseDataSample()

        # pack instance data
        gt_instances = InstanceData()
        _instance_mapping_table = results.get('instance_mapping_table',
                                              self.instance_mapping_table)
        for key, packed_key in _instance_mapping_table.items():
            if key in results:
                gt_instances.set_field(results[key], packed_key)

        # pack `transformed_keypoints` for visualizing data transform
        # and augmentation results
        if self.pack_transformed and 'transformed_keypoints' in results:
            gt_instances.set_field(results['transformed_keypoints'],
                                   'transformed_keypoints')

        data_sample.gt_instances = gt_instances

        # pack instance labels
        gt_instance_labels = InstanceData()
        _label_mapping_table = results.get('label_mapping_table',
                                           self.label_mapping_table)
        for key, packed_key in _label_mapping_table.items():
            if key in results:
                if isinstance(results[key], list):
                    # A list of labels is usually generated by combined
                    # multiple encoders (See ``GenerateTarget`` in
                    # mmpose/datasets/transforms/common_transforms.py)
                    # In this case, labels in list should have the same
                    # shape and will be stacked.
                    _labels = np.stack(results[key])
                    gt_instance_labels.set_field(_labels, packed_key)
                else:
                    gt_instance_labels.set_field(results[key], packed_key)
        data_sample.gt_instance_labels = gt_instance_labels.to_tensor()

        # pack fields
        gt_fields = None
        _field_mapping_table = results.get('field_mapping_table',
                                           self.field_mapping_table)
        for key, packed_key in _field_mapping_table.items():
            # print(key)
            if key in results:
                if isinstance(results[key], list):
                    if gt_fields is None:
                        gt_fields = MultilevelPixelData()
                    else:
                        assert isinstance(
                            gt_fields, MultilevelPixelData
                        ), 'Got mixed single-level and multi-level pixel data.'
                else:
                    if gt_fields is None:
                        gt_fields = PixelData()
                    else:
                        assert isinstance(
                            gt_fields, PixelData
                        ), 'Got mixed single-level and multi-level pixel data.'

                gt_fields.set_field(results[key], packed_key)

        if gt_fields:
            data_sample.gt_fields = gt_fields.to_tensor()

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)

        packed_results = dict()
        packed_results['inputs'] = inputs_tensor
        packed_results['data_samples'] = data_sample

        return packed_results
    
    def transform(self, results):
        results_ll, results_wl = results[0], results[1]
        return [self.transform_one(results_ll), self.transform_one(results_wl)]

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys}, '
        repr_str += f'pack_transformed={self.pack_transformed})'
        return repr_str

OptIntSeq = Optional[Sequence[int]]

def add_prefix(d, p):
    return {k + p: v for k, v in d.items()}

@DATASETS.register_module()
class ExlposeDataset1(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/exlpose.py')
    
    def wl_ll_path_pairs(self):
        pairs_file_path = "/workspace/data/ExLPose/annotations/ExLPose/brightPath2darkPath.txt"
        wl_ll_pairs = {}
        with open(pairs_file_path) as f:
            x = f.readlines()
            for i in x:
                t = i.split(" ")
                # t[1].strip("\n")
                wl_ll_pairs[t[1].rstrip("\n")] = t[0]
        return wl_ll_pairs
    
    @force_full_init
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        :class:`BaseCocoStyleDataset` overrides this method from
        :class:`mmengine.dataset.BaseDataset` to add the metainfo into
        the ``data_info`` before it is passed to the pipeline.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        file_pairs = self.wl_ll_path_pairs()

        # Mixed image transformations require multiple source images for
        # effective blending. Therefore, we assign the 'dataset' field in
        # `data_info` to provide these auxiliary images.
        # Note: The 'dataset' assignment should not occur within the
        # `get_data_info` function, as doing so may cause the mixed image
        # transformations to stall or hang.
        data_info['dataset'] = self
        ll_img_path = data_info['img_path']
        idx = ll_img_path.rfind('/')
        idx = ll_img_path.rfind('/', 0 , idx)
        if ll_img_path[idx+1:] not in file_pairs:
            return self.pipeline([data_info, copy.deepcopy(data_info)])
        wl_img_path = file_pairs[ll_img_path[idx+1:]]
        wl_data_info = copy.deepcopy(data_info)
        wl_data_info['img_path'] = ll_img_path[:idx+1] + wl_img_path
        return self.pipeline([data_info, wl_data_info])

@MODELS.register_module()
class ExlPoseDataPreprocessor(PoseDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:

        if not training:
            data = data[0]
            batch_pad_shape = self._get_pad_shape(data)
            data = super().forward(data=data, training=training)
            inputs, data_samples = data['inputs'], data['data_samples']


            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })
            return {'inputs': inputs, 'data_samples': data_samples}

        ll, wl = data
        data = ll 
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        # update metainfo since the image shape might change
        batch_input_shape = tuple(inputs[0].size()[-2:])
        for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
            data_sample.set_metainfo({
                'batch_input_shape': batch_input_shape,
                'pad_shape': pad_shape
            })


        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)
                
        inputs_ll = copy.deepcopy(inputs)
        data_samples_ll = copy.deepcopy(data_samples)
        
        data = wl
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']


        batch_input_shape = tuple(inputs[0].size()[-2:])
        for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
            data_sample.set_metainfo({
                'batch_input_shape': batch_input_shape,
                'pad_shape': pad_shape
            })


        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': [inputs_ll, inputs], 'data_samples': data_samples_ll}

@MODELS.register_module()
class TeacherHeatmapHead(HeatmapHead):
    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 loss_feat: OptConfigType = None, # Feature distillation config
                 
                 # Configs cho các loại output distillation loss tùy chọn
                 loss_output_kl_cfg: OptConfigType = None,  # Cấu hình cho KL Divergence output loss
                 loss_output_mse_cfg: OptConfigType = None, # Cấu hình cho MSE output loss (nếu muốn riêng)
                                                            # (Trước đây dùng out_w * self.loss_module)

                 out_w: int = 1,
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        BaseHead.__init__(self,init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)
        # self.loss_feat_module = MODELS.build(loss_feat)
        current_in_channels = self.in_channels
        self.loss_feat_module = None
        if loss_feat is not None:
            self.loss_feat_module = MODELS.build(loss_feat)

        self.loss_output_kl_module = None
        if loss_output_kl_cfg is not None:
            self.loss_output_kl_module = MODELS.build(loss_output_kl_cfg)

        self.loss_output_mse_module = None
        if loss_output_mse_cfg is not None:
            self.loss_output_mse_module = MODELS.build(loss_output_mse_cfg)
            # Ví dụ loss_output_mse_cfg: dict(type='KeypointMSELoss', use_target_weight=True, loss_weight=LAMBDA_MSE_OUT)


        if isinstance(current_in_channels, Sequence): # Nếu in_channels là list/tuple (ví dụ từ FPN)
             current_in_channels = current_in_channels[0] # Lấy phần tử đầu tiên hoặc theo logic của bạn
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None
        self.out_w = out_w
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def loss(self,
             feats: Tuple[Tensor], 
             batch_data_samples: OptSampleList,
             layer_feats: Optional[List[Tensor]] = None, 
             teacher_layer_feats: Optional[List[Tensor]] = None, 
             teacher_out: Optional[Tensor] = None, 
             train_cfg: ConfigType = {}) -> dict:
        
        pred_fields_student = self.forward(feats) # Heatmap student S(I_L)

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        losses = dict()


        loss_pose = self.loss_module(pred_fields_student, gt_heatmaps, keypoint_weights)
        losses.update(loss_kpt=loss_pose)

        if self.loss_feat_module is not None and teacher_layer_feats is not None and layer_feats is not None:
            loss_sim = self.loss_feat_module(layer_feats, teacher_layer_feats) 
            losses.update(loss_sim_feat=loss_sim) 


        if teacher_out is not None:
            # KL Divergence Output Distillation
            if self.loss_output_kl_module is not None:
                loss_kl = self.loss_output_kl_module(pred_fields_student, teacher_out.detach(), keypoint_weights)
                losses.update(loss_kl_output=loss_kl)
            

            if self.loss_output_mse_module is not None:
                # KeypointMSELoss thường nhận keypoint_weights trực tiếp
                loss_mse_out = self.loss_output_mse_module(pred_fields_student, teacher_out.detach(), keypoint_weights)
                losses.update(loss_mse_output=loss_mse_out)



        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields_student),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

@MODELS.register_module()
class SCAMLoss(nn.Module):
    def __init__(self,
                 dist_type='L2',  
                 loss_weight=1.0,
                 feature_norm=True,
                 avg_pool=False,    
                 pool_cfg=None):    
        super().__init__()
        self.dist_type = dist_type.lower()
        self.loss_weight = loss_weight
        self.feature_norm = feature_norm
        self.avg_pool = avg_pool

        if self.dist_type == 'l2':
            self.criterion = nn.MSELoss(reduction='mean')
        elif self.dist_type == 'cosine':
            self.criterion = nn.CosineSimilarity(dim=1) 
        else:
            raise ValueError(f"Unsupported dist_type: {dist_type}. Choose 'L2' or 'cosine'.")

        self.pool = None
        if self.avg_pool:
            if pool_cfg:
                self.pool = MODELS.build(pool_cfg) 
            else:

                self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)


    def _calculate_loss_for_pair(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:

        if self.pool:
            s_feat_pooled = self.pool(s_feat)
            t_feat_pooled = self.pool(t_feat)
        else:
            s_feat_pooled = s_feat
            t_feat_pooled = t_feat
        
 
        if s_feat_pooled.shape[2:] != t_feat_pooled.shape[2:]:
            s_feat_pooled = F.interpolate(s_feat_pooled, size=t_feat_pooled.shape[2:], mode='bilinear', align_corners=False)

        if self.dist_type == 'l2':
            return self.criterion(s_feat_pooled, t_feat_pooled)
        
        elif self.dist_type == 'cosine':

            if self.feature_norm:
                s_feat_norm = F.normalize(s_feat_pooled, p=2, dim=1)
                t_feat_norm = F.normalize(t_feat_pooled, p=2, dim=1)
            else:
                s_feat_norm = s_feat_pooled
                t_feat_norm = t_feat_pooled
            
            similarity = self.criterion(s_feat_norm, t_feat_norm) 

            if similarity.ndim > 1 and similarity.shape != s_feat_norm.shape[0:1]: 
                similarity = similarity.mean(dim=list(range(1, similarity.ndim)))

            loss_per_instance = 1.0 - similarity
            return loss_per_instance.mean() 

    def forward(self, token_s: Union[torch.Tensor, List[torch.Tensor]], 
                token_t: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(token_s, list) and isinstance(token_t, list):
            if not token_s: # Handle empty list case
                return torch.tensor(0.0, device=self.loss_weight.device if isinstance(self.loss_weight, torch.Tensor) else 'cpu')

            assert len(token_s) == len(token_t), "Student and teacher must have the same number of feature layers"
            if not token_s: return torch.tensor(0.0, device=token_t[0].device if token_t else 'cpu')

            num_valid_pairs = 0
            total_loss = 0.0
            for s_feat, t_feat in zip(token_s, token_t):
                if s_feat is not None and t_feat is not None:
                    total_loss += self._calculate_loss_for_pair(s_feat, t_feat.detach())
                    num_valid_pairs +=1
            loss = total_loss / num_valid_pairs if num_valid_pairs > 0 else torch.tensor(0.0, device=total_loss.device)

        elif isinstance(token_s, torch.Tensor) and isinstance(token_t, torch.Tensor):
            loss = self._calculate_loss_for_pair(token_s, token_t.detach())
        else:
            raise TypeError(f"Inputs token_s and token_t must be Tensors or lists of Tensors, got {type(token_s)} and {type(token_t)}")
            
        return loss * self.loss_weight

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dist_type={self.dist_type}, '
                f'loss_weight={self.loss_weight}, '
                f'feature_norm={self.feature_norm if self.dist_type == "cosine" else "N/A"}, '
                f'avg_pool={self.avg_pool})')

@MODELS.register_module()
class HRNet1(HRNet):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
        frozen_stages=-1,
        init_cfg=[
            dict(type='Normal', std=0.001, layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ],
        conv_trans=None,
        conv_teacher_trans=None,

        dropblock_size = 0,
    ):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        BaseBackbone.__init__(self, init_cfg=init_cfg)
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages
        self.dropblock_size = dropblock_size
        self.iter = 0
        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        self.upsample_cfg = self.extra.get('upsample', {
            'mode': 'nearest',
            'align_corners': None
        })

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * get_expansion(block)
        self.layer1 = self._make_layer(block, 64, stage1_out_channels,
                                       num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=self.stage4_cfg.get('multiscale_output', False))

        self.conv_trans = conv_trans
        if conv_trans is not None and conv_teacher_trans is not None:
            self.trans = nn.ModuleList()
            for idx, num_channel in enumerate(conv_trans):
                self.trans.append(nn.Sequential(nn.Conv2d(num_channel, conv_teacher_trans[idx], kernel_size = 1), nn.ReLU(), nn.Conv2d(conv_teacher_trans[idx], conv_teacher_trans[idx], kernel_size = 1)).to('cuda'))
        self._freeze_stages()


    def forward(self, x, isTeacher=False):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        layer_feats = [x]
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)  
        y_list = self.stage2(x_list)
        layer_feats.append(y_list[0])

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i]) 
        y_list = self.stage3(x_list)
        layer_feats.append(y_list[0])
        
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        layer_feats.append(y_list[0])
        
            
        if not isTeacher and self.conv_trans is not None:
            trans_layer_feats = []
            for idx, layer_feat in enumerate(layer_feats):
                # print(self.trans[idx], layer_feat.shape)
                trans_layer_feats.append(self.trans[idx](layer_feat))
            layer_feats = trans_layer_feats

        return tuple([y_list[0]]), layer_feats

    def train(self, mode=True):
        """Convert the model into training mode."""
        BaseBackbone.train(self,mode)

        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

@MODELS.register_module()
class TeacherTopdownPoseEstimator(TopdownPoseEstimator):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 teacher_ckpt = None,
                 teacher_config = None,                 
                 teacher_dropout_epoch: Optional[int] = None,
                 teacher_dropout_iter: Optional[int] = None,

                ):
        BasePoseEstimator.__init__(self,
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
        
        if isinstance(teacher_config, str):
            teacher_config = mmengine.Config.fromfile(teacher_config)
        if teacher_config is not None:
            self.teacher_model = MODELS.build(teacher_config['model'])
        self.teacher_config = teacher_config
        self.teacher_ckpt = teacher_ckpt
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')


        self.teacher_dropout_epoch = teacher_dropout_epoch
        self.teacher_dropout_iter = teacher_dropout_iter
        self.current_epoch = 0 
    def set_epoch(self, epoch: int): #
        self.current_epoch = epoch+1


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:

        self.backbone.iter += 1
        if self.teacher_config is not None:
            if self.teacher_ckpt is not None:
                with torch.no_grad():
                    teacher_feats, teacher_layer_feats = self.teacher_model.backbone(inputs[1], isTeacher=True)
                    teacher_out = self.teacher_model.head(teacher_feats)
            else:
                teacher_feats, teacher_layer_feats = self.teacher_model.backbone(inputs[1],isTeacher=True)
                teacher_out = self.teacher_model.head(teacher_feats)
        else:
            teacher_feats = None
            teacher_layer_feats = None
            teacher_out = None
        
        feats, layer_feats = self.extract_feat(inputs[0])

        losses = dict()

        if self.with_head:
            if self.teacher_ckpt is None or self.teacher_config is not None:
                if self.teacher_dropout_iter is not None:
                    if self.backbone.iter % self.teacher_dropout_iter == 0:
                        losses.update(self.head.loss(feats, data_samples, layer_feats, teacher_layer_feats, None, train_cfg=self.train_cfg))
                    else:
                        losses.update(self.head.loss(feats, data_samples, layer_feats, teacher_layer_feats, teacher_out, train_cfg=self.train_cfg))
                else:
                    if self.teacher_dropout_epoch is None or self.current_epoch < self.teacher_dropout_epoch:
                        losses.update(self.head.loss(feats, data_samples, layer_feats, teacher_layer_feats, teacher_out, train_cfg=self.train_cfg))
                    else:
                        losses.update(self.head.loss(feats, data_samples, None, None, None, train_cfg=self.train_cfg))
                losses.update(add_prefix(self.teacher_model.head.loss(teacher_feats, data_samples, None, None, None, train_cfg=self.train_cfg), "_teacher"))
            else :
                losses.update(self.head.loss(feats, data_samples, layer_feats, teacher_layer_feats, teacher_out, train_cfg=self.train_cfg))
        # print(data_samples[0].shape)
        return losses


    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:


        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats, _ = self.extract_feat(inputs)
            _feats_flip, _ = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats, _ = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None
                 ) -> Union[Tensor, Tuple[Tensor]]:

        x = self.extract_feat(inputs)
        if len(x) == 2:
            x = x[0]
        if self.with_head:
            x = self.head.forward(x)

        return x

@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.set_epoch(epoch)

@MODELS.register_module()
class AdaptiveKLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for heatmaps, with adaptive weighting based on student's entropy.
    Implements L_KD_adaptive = (1 - H(S(x))/log C) * L_KD
    """
    def __init__(self, temperature=1.0, reduction='batchmean', loss_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction) 
        self.loss_weight = loss_weight # Đây là lambda_KD trong công thức mới

    def forward(self, student_heatmaps, teacher_heatmaps, keypoint_weights=None):
        """
        Args:
            student_heatmaps (Tensor): Predicted heatmaps from student (logits). Shape (N, K, H, W).
            teacher_heatmaps (Tensor): Target heatmaps from teacher (detached). Shape (N, K, H, W).
            keypoint_weights (Tensor, optional): Not directly used in this implementation.
        """
        if student_heatmaps is None or teacher_heatmaps is None:

            return student_heatmaps.new_tensor(0.) if student_heatmaps is not None else torch.tensor(0.)
        N, K, H, W = student_heatmaps.shape
        

        student_log_probs = F.log_softmax(student_heatmaps.view(N, K, -1) / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_heatmaps.view(N, K, -1) / self.temperature, dim=-1).detach() # Detach teacher
        

        student_input_flat = student_log_probs.view(N * K, -1)
        teacher_target_flat = teacher_probs.view(N * K, -1)
        

        base_kl_loss = self.kl_div_loss(student_input_flat, teacher_target_flat)

        base_kl_loss = base_kl_loss * (self.temperature**2)


        student_heatmaps_flat_for_entropy = student_heatmaps.view(N * K, -1)
        student_probs_for_entropy = F.softmax(student_heatmaps_flat_for_entropy, dim=-1) + 1e-9


        student_entropy = -torch.sum(
            student_probs_for_entropy * torch.log2(student_probs_for_entropy), 
            dim=-1
        ) 
        num_classes = H * W
        max_entropy = torch.log2(torch.tensor(num_classes, device=student_heatmaps.device))


        normalized_entropy = torch.clamp(student_entropy / max_entropy, 0, 1)
        adaptive_weight = (1.0 - normalized_entropy).mean() 


        adaptive_kl_loss = adaptive_weight * base_kl_loss


        final_loss = adaptive_kl_loss * self.loss_weight

        return final_loss