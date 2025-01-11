from pytorch3d.loss import chamfer_distance
import torch
from Shape_Measure.distance import EMDLoss, ChamferLoss

def chamfer_distance2(p1, p2):
    dist = ChamferLoss()
    cost1, cost2 = dist(p1, p2)
    reducer = torch.mean
    loss = (reducer(cost1, dim=1)) + (reducer(cost2, dim=1))
    return loss

# TODO: refine function and get faster
def compute_cm_loss(source_p, target_p, target_part, mask=None, batch_reduction="mean"):
    if mask is not None:
        idx = mask.sum(1)*1024
        loss_all = []
        loss_part = []
        for bs in range(len(source_p)):
            # source_p has many points, target_p has 2048 points
            loss = chamfer_distance2(source_p[bs:bs+1, :int(idx[bs].item())], target_p[bs:bs+1,:])
            loss_part_now = []
            for i in range(len(target_part[bs])):
                loss_p = chamfer_distance2(source_p[bs:bs+1, i*1024:(i+1)*1024], target_part[bs][i].unsqueeze(0))
                loss_part_now.append(loss_p)
            loss_part.append(torch.stack(loss_part_now).mean())
            loss_all.append(loss)
        return torch.stack(loss_all).mean(), torch.stack(loss_part).mean()
        # return loss_all.mean()
    loss = chamfer_distance2(source_p, target_p)
    return loss
    
