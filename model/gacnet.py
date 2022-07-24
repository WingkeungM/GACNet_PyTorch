import torch.nn as nn
import torch.nn.functional as F
import pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils

from model.gacnet_utils import *


graph_inf = {'stride_list': [1024, 256, 64, 32], #can be seen as the downsampling rate
             'radius_list': [0.1, 0.2, 0.4, 0.8, 1.6], # radius for neighbor points searching
             'maxsample_list': [12, 21, 21, 21, 12] #number of neighbor points for each layer
}

# number of units for each mlp layer
forward_parm = [
                [ [32,32,64], [64] ],
                [ [64,64,128], [128] ],
                [ [128,128,256], [256] ],
                [ [256,256,512], [512] ],
                [ [256,256], [256] ]
]

# for feature interpolation stage
upsample_parm = [
                  [128, 128],
                  [128, 128],
                  [256, 256],
                  [256, 256]
]

# parameters for fully connection layer
fullconect_parm = 128

net_inf = {'forward_parm': forward_parm,
           'upsample_parm': upsample_parm,
           'fullconect_parm': fullconect_parm
}




class GACNet(nn.Module):
    def __init__(self, num_classes, graph_inf, net_inf):
        super(GACNet, self).__init__()
        self.num_classes = num_classes
        self.forward_parm, self.upsample_parm, self.fullconect_parm = \
            net_inf['forward_parm'], net_inf['upsample_parm'], net_inf['fullconect_parm']

        self.stride_inf = graph_inf['stride_list']

        self.graph_attention_layer1 = GraphAttentionConvLayer(4, self.forward_parm[0][0], self.forward_parm[0][1])
        self.graph_attention_layer2 = GraphAttentionConvLayer(64, self.forward_parm[1][0], self.forward_parm[1][1])
        self.graph_attention_layer3 = GraphAttentionConvLayer(128, self.forward_parm[2][0], self.forward_parm[2][1])
        self.graph_attention_layer4 = GraphAttentionConvLayer(256, self.forward_parm[3][0], self.forward_parm[3][1])

        self.gragh_pooling_layer = GraphPoolingLayer()
        self.mid_graph_attention_layers = GraphAttentionConvLayer(512, self.forward_parm[-1][0], self.forward_parm[-1][1])

        self.point_upsample_layer1 = PointUpsampleLayer(512 + 256, self.upsample_parm[3])
        self.point_upsample_layer2 = PointUpsampleLayer(256 + 256, self.upsample_parm[2])
        self.point_upsample_layer3 = PointUpsampleLayer(128 + 256, self.upsample_parm[1])
        self.point_upsample_layer4 = PointUpsampleLayer(64 + 128, self.upsample_parm[0])

        self.graph_attention_layer_for_featurerefine = GraphAttentionConvLayerforFeatureRefine(self.num_classes)

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_classes, 1)

    def forward(self, features, graph_prd, coarse_map):
        inif = features[:, :, 0:6]  # (x,y,z,r,g,b)
        features = features[:, :, 2:]  # (z, r, g, b, and (initial geofeatures if possible))

        feature_prd = []

        features = self.graph_attention_layer1(graph_prd[0], features)
        feature_prd.append(features)
        features = self.gragh_pooling_layer(features, coarse_map[0])

        features = self.graph_attention_layer2(graph_prd[1], features)
        feature_prd.append(features)
        features = self.gragh_pooling_layer(features, coarse_map[1])

        features = self.graph_attention_layer3(graph_prd[2], features)
        feature_prd.append(features)
        features = self.gragh_pooling_layer(features, coarse_map[2])

        features = self.graph_attention_layer4(graph_prd[3], features)
        feature_prd.append(features)
        features = self.gragh_pooling_layer(features, coarse_map[3])

        features = self.mid_graph_attention_layers(graph_prd[-1], features)

        features = self.point_upsample_layer1(graph_prd[3]['vertex'], graph_prd[3 + 1]['vertex'], feature_prd[3], features)
        features = self.point_upsample_layer2(graph_prd[2]['vertex'], graph_prd[2 + 1]['vertex'], feature_prd[2], features)
        features = self.point_upsample_layer3(graph_prd[1]['vertex'], graph_prd[1 + 1]['vertex'], feature_prd[1], features)
        features = self.point_upsample_layer4(graph_prd[0]['vertex'], graph_prd[0 + 1]['vertex'], feature_prd[0], features)

        features = features.permute(0, 2, 1)
        features = F.relu(self.bn1(self.conv1(features)))
        features = self.drop(features)
        features = self.conv2(features)
        features = features.permute(0, 2, 1)

        features = self.graph_attention_layer_for_featurerefine(inif, features, graph_prd[0]['adjids'])
        features = F.log_softmax(features, dim=2)
        return features


def build_graph_pyramid(xyz, graph_inf):
    """ Builds a pyramid of graphs and pooling operations corresponding to progressively coarsened point cloud.
    Inputs:
        xyz: (batchsize, num_point, nfeature)
        graph_inf: parameters for graph building (see run.py)
    Outputs:
        graph_prd: graph pyramid contains the vertices and their edges at each layer
        coarse_map: record the corresponding relation between two close graph layers (for graph coarseing/pooling)
    """
    stride_list, radius_list, maxsample_list = graph_inf['stride_list'], graph_inf['radius_list'], graph_inf['maxsample_list']

    graph_prd = []
    graph = {}
    coarse_map = []

    xyz = xyz.contiguous()
    ids = pointnet2_utils.ball_query(radius_list[0], maxsample_list[0], xyz, xyz)
    graph['vertex'], graph['adjids'] = xyz, ids
    graph_prd.append(graph.copy())

    for stride, radius, maxsample in zip(stride_list, radius_list[1:], maxsample_list[1:]):
        xyz, coarse_map_ids = graph_coarse(xyz, ids, stride)
        coarse_map.append(coarse_map_ids.int())
        ids = pointnet2_utils.ball_query(radius, maxsample, xyz, xyz)
        graph['vertex'], graph['adjids'] = xyz, ids
        graph_prd.append(graph.copy())

    return graph_prd, coarse_map

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    input = torch.randn((8,6,4096)).permute(0, 2, 1).cuda()
    graph_prd, coarse_map = build_graph_pyramid(input[:, :, :3], graph_inf)
    model = GACNet(50, graph_inf, net_inf).cuda()
    logits = model(input, graph_prd, coarse_map)

