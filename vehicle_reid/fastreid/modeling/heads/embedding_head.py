# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from vehicle_reid.fastreid.config import configurable
from vehicle_reid.fastreid.layers import *
from vehicle_reid.fastreid.layers import pooling, any_softmax
from vehicle_reid.fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY

class SpTeVi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spte = nn.Linear(9, 128)
        self.vi = nn.Linear(4096, 128)
        self.fusion = nn.Linear(256, 128)
        self.predict = nn.Linear(128, 1)
        # self.spte = nn.Linear(9, 256)
        # self.vi = nn.Linear(4096, 256)
        # self.fusion = nn.Linear(512, 256)
        # self.predict = nn.Linear(256, 1)
        
    def forward(self, spatialtemporal, visual):
        spatial_temporal_feature = torch.relu(self.spte(spatialtemporal))
        visual_feature = torch.relu(self.vi(visual))
        cat_feature = torch.cat((spatial_temporal_feature, visual_feature), dim=-1)
        fusion_feature = torch.relu(self.fusion(cat_feature))
        pred_match = self.predict(fusion_feature)
        
        return pred_match

@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        
        self.match_cls_layer = SpTeVi()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)
        self.match_cls_layer.apply(weights_init_kaiming)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = neck_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
        }
        # New matching features strategy
        # spatial_temporals, visuals = [], []
        # bs = temporals.shape[0]
        # for i in range(bs):
        #     spatial_temporal, visual = [], []
        #     for j in range(bs):
        #         if temporals[i] < temporals[j]:
        #             spatial_vector = torch.cat([spatials[i], spatials[j], ((temporals[j] - temporals[i] - 51) / (1189 - 51)).unsqueeze(0)], dim=0)
        #             visual_vector = torch.cat([feat[i], feat[j]], dim=0)
        #         else:
        #             spatial_vector = torch.cat([spatials[j], spatials[i], ((temporals[i] - temporals[j] - 51) / (1189 - 51)).unsqueeze(0)], dim=0)
        #             visual_vector = torch.cat([feat[j], feat[i]], dim=0)
        #         spatial_temporal.append(spatial_vector)
        #         visual.append(visual_vector)
        #     spatial_temporal = torch.stack(spatial_temporal)
        #     visual = torch.stack(visual)
        #     spatial_temporals.append(spatial_temporal)
        #     visuals.append(visual)
        # spatial_temporals = torch.stack(spatial_temporals)
        # visuals = torch.stack(visuals)
            
        # pred_matches = self.match_cls_layer(spatial_temporals, visuals)

        # return {
        #     "cls_outputs": cls_outputs,
        #     "pred_class_logits": logits.mul(self.cls_layer.s),
        #     "features": feat,
        #     "pred_match_logits": pred_matches
        # }
