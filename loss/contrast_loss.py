'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def compute_contrast_loss_loss(tgt_part_f, src_f, src_labels):
    # B x C
    bs = len(tgt_part_f)
    num_part = src_f.size(1)
    tgt_part_f = tgt_part_f.reshape(bs*num_part, -1)
    src_f = src_f.reshape(bs*num_part, -1)
    src_labels = src_labels.reshape(bs*num_part)

    logit_scale = nn.Parameter(torch.ones([], device=src_f.device) * np.log(1 / 0.07)).exp()# outputs['logit_scale']
    local_batch_size = len(tgt_part_f)
    last_local_batch_size = None

    if local_batch_size != last_local_batch_size:
        labels = local_batch_size * get_rank() + torch.arange(
            local_batch_size, device=tgt_part_f.device
        )
        last_local_batch_size = local_batch_size
    labels[src_labels.squeeze()==-1]=-1

    # normalized features
    tgt_part_f_embed = F.normalize(tgt_part_f, dim=-1, p=2)
    src_f_embed = F.normalize(src_f, dim=-1, p=2)

    # gather features from all GPUs
    tgt_part_f_embed_all, src_f_embed_all = \
        all_gather_batch([tgt_part_f_embed, src_f_embed])

    # cosine similarity as logits
    logits_per_tgt_src = logit_scale * tgt_part_f_embed @ src_f_embed_all.t()
    
    loss = F.cross_entropy(logits_per_tgt_src, labels, ignore_index=-1)

    # compute accuracy
    # with torch.no_grad():
    #     pred = torch.argmax(logits_per_pc_text, dim=-1)
    #     correct = pred.eq(labels).sum()
    #     pc_text_acc = 100 * correct / local_batch_size

    #     pred = torch.argmax(logits_per_pc_image, dim=-1)
    #     correct = pred.eq(labels).sum()
    #     pc_image_acc = 100 * correct / local_batch_size
    return loss


if __name__ == "__main__":
    # loss = ULIPWithImageLoss()
    outputs = {'pc_embed':torch.rand(32,128), 'text_embed':torch.rand(32,128), 'image_embed':torch.rand(32,128), 'logit_scale':torch.rand(1)}
    # loss(outputs)
    
    
    pass