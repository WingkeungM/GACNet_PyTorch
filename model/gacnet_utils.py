# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
import torch.nn.functional as F

import pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils


gac_par = [
    [32, 16], #MLP for xyz
    [16, 16], #MLP for feature
    [64]      #hidden node of MLP for mergering
]


class MLP1D(nn.Module):
    def __init__(self, inchannel, mlp_1d):
        super(MLP1D, self).__init__()
        self.mlp_conv1ds = nn.ModuleList()
        self.mlp_bn1ds = nn.ModuleList()
        self.mlp_1d = mlp_1d
        last_channel = inchannel
        for i, outchannel in enumerate(self.mlp_1d):
            self.mlp_conv1ds.append(nn.Conv1d(last_channel, outchannel, 1))
            self.mlp_bn1ds.append(nn.BatchNorm1d(outchannel))
            last_channel = outchannel

    def forward(self, features):
        features = features.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_conv1ds):
            bn = self.mlp_bn1ds[i]
            features = F.relu(bn(conv(features)))
        features = features.permute(0, 2, 1)
        return features

class MLP2D(nn.Module):
    def __init__(self, inchannel, mlp_2d):
        super(MLP2D, self).__init__()
        self.mlp_conv2ds = nn.ModuleList()
        self.mlp_bn2ds = nn.ModuleList()
        self.mlp_2d = mlp_2d
        last_channel = inchannel
        for i, outchannel in enumerate(self.mlp_2d):
            self.mlp_conv2ds.append(nn.Conv2d(last_channel, outchannel, 1))
            self.mlp_bn2ds.append(nn.BatchNorm2d(outchannel))
            last_channel = outchannel

    def forward(self, features):
        features = features.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_conv2ds):
            bn = self.mlp_bn2ds[i]
            features = F.relu(bn(conv(features)))
        features = features.permute(0, 3, 2, 1)
        return features



class CoeffGeneration(nn.Module):
    def __init__(self, inchannel):
        super(CoeffGeneration, self).__init__()
        self.inchannel = inchannel
        self.MlP2D = MLP2D(self.inchannel, gac_par[1])
        self.MlP2D_2 = MLP2D(32, gac_par[2])
        self.conv1 = nn.Conv2d(64, 16 + self.inchannel, 1)
        self.bn1 = nn.BatchNorm2d(16 + self.inchannel)

    def forward(self, grouped_features, features, grouped_xyz, mode='with_feature'):
        if mode == 'with_feature':
            coeff = grouped_features - features.unsqueeze(dim=2)
            coeff = self.MlP2D(coeff)
            coeff = torch.cat((grouped_xyz, coeff), dim=-1)
        if mode == 'edge_only':
            coeff = grouped_xyz
        if mode == 'feature_only':
            coeff = grouped_features - features.unsqueeze(dim=2)
            coeff = self.MlP2D(coeff)

        grouped_features = torch.cat((grouped_xyz, grouped_features), dim=-1)
        coeff = self.MlP2D_2(coeff)
        coeff = coeff.permute(0, 3, 2, 1)
        coeff = self.bn1(self.conv1(coeff))
        coeff = coeff.permute(0, 3, 2, 1)
        coeff = F.softmax(coeff, dim=2)

        grouped_features = coeff * grouped_features
        grouped_features = torch.sum(grouped_features, dim=2)
        return grouped_features

class GraphAttentionConvLayer(nn.Module):
    def __init__(self, feature_inchannel, mlp1, mlp2):
        super(GraphAttentionConvLayer, self).__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.feature_inchannel = feature_inchannel
        self.edge_mapping = MLP2D(3, gac_par[0])
        self.MLP1D = MLP1D(self.feature_inchannel, self.mlp1)
        self.MLP1D_2 = MLP1D(2 * self.mlp1[-1] + 16, self.mlp2)
        self.coeff_generation = CoeffGeneration(self.mlp1[-1])

    def forward(self, graph, features):
        xyz, ids = graph['vertex'], graph['adjids']
        grouped_xyz = pointnet2_utils.grouping_operation(xyz.permute(0, 2, 1).contiguous(), ids).permute(0, 2, 3, 1).contiguous()
        grouped_xyz -= xyz.unsqueeze(dim=2)
        grouped_xyz = self.edge_mapping(grouped_xyz)

        features = self.MLP1D(features)
        grouped_features = pointnet2_utils.grouping_operation(features.permute(0, 2, 1).contiguous(), ids).permute(0, 2, 3, 1).contiguous()

        new_features = self.coeff_generation(grouped_features, features, grouped_xyz)
        if self.mlp2 is not None and features is not None:
            new_features = torch.cat((features, new_features), dim=-1)
            new_features = self.MLP1D_2(new_features)
        return new_features


class GraphPoolingLayer(nn.Module):
    def __init__(self, pooling='max'):
        super(GraphPoolingLayer, self).__init__()
        self.pooling = pooling

    def forward(self, features, coarse_map):
        grouped_features = pointnet2_utils.grouping_operation(features.permute(0, 2, 1).contiguous(), coarse_map).permute(0, 2, 3, 1).contiguous()
        if self.pooling == 'max':
            new_features = torch.max(grouped_features, dim=2)[0]
        return new_features

class PointUpsampleLayer(nn.Module):
    def __init__(self, inchannel, upsample_parm):
        super(PointUpsampleLayer, self).__init__()
        self.inchannel = inchannel
        self.upsample_list = upsample_parm
        self.MLP1D = MLP1D(self.inchannel, self.upsample_list)


    def forward(self, xyz1, xyz2, features1, features2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        features1 = features1.permute(0, 2, 1)
        features2 = features2.permute(0, 2, 1)
        assert xyz1.is_contiguous()
        assert xyz2.is_contiguous()
        dist, idx = pointnet2_utils.three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        assert features2.is_contiguous()
        interpolated_features = pointnet2_utils.three_interpolate(features2, idx, weight)

        new_features = torch.cat((interpolated_features, features1), dim=1).permute(0, 2, 1)

        if self.upsample_list is not None:
            new_features = self.MLP1D(new_features)
        return new_features

class GraphAttentionConvLayerforFeatureRefine(nn.Module):
    def __init__(self, inchannel):
        super(GraphAttentionConvLayerforFeatureRefine, self).__init__()
        self.inchannel = inchannel
        self.edge_mapping = MLP2D(6, gac_par[0])
        self.coeff_generation = CoeffGeneration(self.inchannel)
        self.conv1 = nn.Conv1d(16 + 2 * self.inchannel, self.inchannel, 1)

    def forward(self, initf, features, ids):
        initf = initf.permute(0, 2, 1).contiguous()
        grouped_initf = pointnet2_utils.grouping_operation(initf, ids).permute(0, 2, 3, 1).contiguous()
        grouped_initf -= initf.permute(0, 2, 1).unsqueeze(dim=2)
        grouped_initf = self.edge_mapping(grouped_initf)
        features = features.permute(0, 2, 1).contiguous()
        grouped_features = pointnet2_utils.grouping_operation(features, ids).permute(0, 2, 3, 1).contiguous()
        features = features.permute(0, 2, 1)

        new_features = self.coeff_generation(grouped_features, features, grouped_initf)

        new_features = torch.cat((features, new_features), dim=-1)
        new_features = new_features.permute(0, 2, 1)
        new_features = self.conv1(new_features)
        new_features = new_features.permute(0, 2, 1)

        return new_features



def graph_coarse(xyz_org, ids_full, stride):
    """ Coarse graph with down sampling, and find their corresponding vertexes at previous (or father) level. """
    if stride > 1:
        sub_pts_ids = pointnet2_utils.furthest_point_sample(xyz_org, stride)
        sub_xyz = pointnet2_utils.gather_operation(xyz_org.permute(0, 2, 1).contiguous(), sub_pts_ids).permute(0, 2, 1).contiguous()

        ids = pointnet2_utils.grouping_operation(ids_full.permute(0, 2, 1).float().contiguous(),
                                    sub_pts_ids.unsqueeze(dim=-1).contiguous()).long().squeeze(-1).permute(0, 2, 1).contiguous()  # (batchsize, num_point, maxsample)

        return sub_xyz, ids
    else:
        return xyz_org, ids_full




