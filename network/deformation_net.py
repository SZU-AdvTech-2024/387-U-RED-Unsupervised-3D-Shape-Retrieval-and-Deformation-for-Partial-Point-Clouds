import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_graph.attention_gnn import GraphAttentionNet
from attention_graph.attention_utils import FeedForwardNet_norm


# for this net, the input
class NodeDecoder(nn.Module):
    def __init__(self, input_dim, intermediate_layer, embedding_size, use_norm='use_bn'):
        super(NodeDecoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, intermediate_layer)
        self.fc2 = nn.Linear(intermediate_layer, embedding_size)
        self.use_norm = use_norm
        if use_norm == 'use_bn':
            self.bn1 = nn.BatchNorm1d(intermediate_layer)
        if use_norm == 'use_ln':
            self.ln1 = nn.LayerNorm(intermediate_layer, elementwise_affine=True)
        if use_norm == 'use_in':
            self.in1 = nn.InstanceNorm1d(intermediate_layer)

    def forward(self, x):
        if self.use_norm == 'use_bn':
            x = self.fc1(x)
            x = x.permute(0, 2, 1)
            x = self.bn1(x)
            x = F.relu(x)
            x = x.permute(0, 2, 1)
        elif self.use_norm == 'use_ln':
            x = F.relu(self.ln1(self.fc1(x)))
        elif self.use_norm == 'use_in':
            x = F.relu(self.in1(self.fc1(x)))

        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DeformNet_MatchingNet(nn.Module):
    def __init__(self, input_dim, num_stages=2, num_heads=4, part_latent_dim=256,
                 graph_dim=128, output_dim=6, use_offset=False, point_f_dim=256,
                 points_num=2048, max_num_parts=12, matching=True):
        super(DeformNet_MatchingNet, self).__init__()
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.num_heads = num_heads
        self.use_offset = use_offset
        self.output_dim = output_dim
        self.graph_dim = graph_dim
        self.point_f_dim = point_f_dim
        self.points_num = points_num
        self.max_num_parts = max_num_parts
        # network modules
        # 1 encode the part code into target dimension
        # for deformation
        self.part_encoding = FeedForwardNet_norm([part_latent_dim, 128, self.graph_dim], use_norm='None')
        self.param_decoder = FeedForwardNet_norm([self.input_dim, 256, self.output_dim], use_norm='None')
        # self.param_decoder = FeedForwardNet_norm([self.input_dim + 32, 256, self.output_dim], use_norm='None')  # add noise
        self.graph_attention_net = GraphAttentionNet(self.num_stages,
                                                     self.graph_dim, self.num_heads, use_offset=use_offset)

        # for matching
        self.matching = matching
        if matching:
            self.matching_net = FeedForwardNet_norm([self.point_f_dim + self.graph_dim * 2, 512,
                                                     1024, self.points_num], use_norm='use_bn')
        else:
            self.matching_net = None

    def forward(self, target_f, src_part_f, per_point_f):
        # node that for the batch, we process each element individually
        # global_f : bs x 256   full or partial global source feature
        # target_f : bs x 256
        # part_f : bs x {32 x nofobj} list
        # per_point_f: bs x 256 x 1024   # target occlusion per point, its matching matrix to source
        # src_part_f: [bsx16, 128], target_f : bs x 256, per_point_f: bs x 2048 x 256
        bs = target_f.shape[0]
        max_num_parts = src_part_f.shape[1]
        src_part_f = src_part_f.view(bs, max_num_parts, -1).permute(0, 2, 1)  # bs x c x max_num_parts
        global_src_f = src_part_f.mean(dim=-1)  # bs x c
        global_node = torch.cat([global_src_f.unsqueeze(-1), target_f.unsqueeze(-1)], dim=-1)  # bs x 256 x 2
        global_node_a, part_node_a = self.graph_attention_net(global_node, src_part_f)
        global_node_r = torch.cat([global_node_a[:, :, 0],
                                   global_node_a[:, :, 1]], dim=1).view(bs, -1, 1).repeat(1, 1, max_num_parts)
        full_r = torch.cat([global_node_r, part_node_a], dim=1)  # bs x 256+512 x 8
        params = self.param_decoder(full_r)  # bs x 6 x max_num_parts
        params = params.permute(0, 2, 1).contiguous() # bs x max_num_parts x 6

        return params


class re_residual_net(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(re_residual_net, self).__init__()
        self.input_dim = input_dim
        self.residual_net = FeedForwardNet_norm([self.input_dim, 256, 256, 32, output_dim], use_norm='use_bn')

    def forward(self, concat_feature):
        # concat feature: bs x num_points x feature dim
        assert self.input_dim == concat_feature.shape[-1]
        concat_feature = concat_feature.permute(0, 2, 1)
        residual_value = self.residual_net(concat_feature)
        return residual_value.permute(0, 2, 1) # bs x num_points x 3




if __name__ == '__main__':
    import numpy as np

    global_f = np.random.random(size=(8, 256))
    target_f = np.random.random(size=(8, 256))
    part_f = np.random.random(size=(8, 32, 12))
    per_point_f = np.random.random(size=(8, 256, 512))
    global_f = torch.from_numpy(global_f.astype(np.float32)).cuda()
    target_f = torch.from_numpy(target_f.astype(np.float32)).cuda()
    part_f = torch.from_numpy(part_f.astype(np.float32)).cuda()
    per_point_f = torch.from_numpy(per_point_f.astype(np.float32)).cuda()
    network = DeformNet_MatchingNet(256 * 3, 3, 4, graph_dim=256)
    network = network.cuda()
    result_list, matching_m = network(global_f, target_f, part_f, per_point_f)
    s = 1
