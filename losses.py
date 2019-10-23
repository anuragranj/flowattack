import torch
import torch.nn as nn
from torch.autograd import Variable
epsilon = 1e-8
def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:,2,:,:]
        epe = epe * valid
        avg_epe = epe.sum()/(valid.sum() + epsilon)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)


    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()

def compute_cossim(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    #u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    #u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    #v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    similarity = nn.functional.cosine_similarity(gt[:,:2], pred)
    if nc == 3:
        valid = gt[:,2,:,:]
        similarity = similarity * valid
        avg_sim = similarity.sum()/(valid.sum() + epsilon)
    else:
        avg_sim = similarity.sum()/(bs*h_gt*w_gt)


    if type(avg_sim) == Variable: avg_sim = avg_sim.data

    return avg_sim.item()

def multiscale_cossim(gt, pred):
    assert(len(gt)==len(pred))
    loss = 0
    for (_gt, _pred) in zip(gt, pred):
        loss +=  - nn.functional.cosine_similarity(_gt, _pred).mean()

    return loss
