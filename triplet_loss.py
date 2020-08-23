import numpy as np
import torch
import os


def triplet_loss(opt, anchor, pos_map, neg_map, margin):
    '''anchor, pos_mapping, neg_mapping all should be TENSOR.'''
    # anchor: 1 X C X H X W    where 1 is Batch size

    anchor = anchor.view(anchor.size(1), -1)        # anchor: C X HW    where c is latent dimension.

    c = anchor.size(0)
    hw = anchor.size(1)

    total_loss = torch.zeros([hw, 1])
    for i in range(hw):

        # get embedding values
        v_a = anchor.narrow_copy(0, i, c)       # V_a : 1 x c

        vector_map_p = np.zeros((hw, 1))
        vector_map_n = np.zeros((hw, 1))
        p = pos_map[i]
        n = neg_map[i]
        vector_map_p[i][p] = 1.
        vector_map_n[i][n] = 1.

        vector_map_p = torch.from_numpy(vector_map_p)
        vector_map_n = torch.from_numpy(vector_map_n)

        v_p = torch.mm(vector_map_p, anchor)    # V_p : 1 x C
        v_n = torch.mm(vector_map_n, anchor)    # V_n : 1 x C

        # get L2 distance
        pos_loss = (v_a - v_p).pow(2).sum(1).pow(.5)
        neg_loss = (v_a - v_n).pow(2).sum(1).pow(.5)

        pos_loss = torch.mean(pos_loss, 1)
        neg_loss = torch.mean(neg_loss, 1)

        # remove the negative values
        clamp = torch.clamp(pos_loss + -1 * neg_loss + margin, min=0)
        # mean for batch
        trip = torch.mean(clamp, 0)

        total_loss[i] = trip

    return total_loss.mean()