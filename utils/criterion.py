import torch.nn as nn
import torch
from torch.nn import functional as F

class Criterion(nn.Module):
    def __init__(self, batchsize=2, ignore_index=255):
        super(Criterion, self).__init__()
        self.ignore_index = ignore_index
        self.batchsize = batchsize
        self.criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def loss(self, preds, target, writer, i_iter, total_iters):
        # pred= [[seg1, seg2], [edge], [heatmap1, pafs1, ..., ...]]
        # [1, 15, 48, 48], [1, 26, 48, 48]......
        # target = [labels, edges, heatmap, pafs, heatmap_mask, pafs_mask]
        # [1, 15, 48, 48], [1, 26, 48, 48]
        h, w = target[0].size(1), target[0].size(2)
        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)
        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.Tensor([weight_neg, weight_pos])
        # loss for parsing
        loss_parsing = 0
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss_parsing += self.criterion_seg(scale_pred, target[0])  # CEloss
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss_parsing += self.criterion_seg(scale_pred, target[0])
        # loss for edge
        loss_edge = 0
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss_edge += F.cross_entropy(scale_pred, target[1],
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss_edge += F.cross_entropy(scale_pred, target[1],
                                    weights.cuda(), ignore_index=self.ignore_index)
        #loss for pose
        loss_pose = 0
        preds_pose = preds[2]
        num_pix1 = torch.sum(target[4] > 0.5, dtype=torch.float)
        num_pix2 = torch.sum(target[5] > 0.5, dtype=torch.float)
        if isinstance(preds_pose, list):
            for i, pred_pose in enumerate(preds_pose):
                if i % 2 == 0:  # heatmap
                    loss_pose += self.l2_loss(pred_pose, target[2], target[4], num_pix1)
                elif i % 2 == 1:  # pafs
                    loss_pose += self.l2_loss(pred_pose, target[3], target[5], num_pix2)
        else:
            print("error")

        print('iter:{}/{},parsing_loss:[{}],edge_loss:[{}],seg_loss:[{}],pose_loss:[{}]'.format(i_iter, total_iters,
                                                     loss_parsing.item(),
                                                     loss_edge.item(),
                                                    loss_parsing.item()+loss_edge.item(),
                                                    loss_pose.item()))
        if i_iter % 100 == 0:
            writer.add_scalar('loss_parsing', loss_parsing.item(), i_iter)
            writer.add_scalar('loss_edge', loss_edge.item(), i_iter)
            writer.add_scalar('loss_seg', loss_parsing.item()+loss_edge.item(), i_iter)
            writer.add_scalar('loss_pose', loss_pose.item(), i_iter)

        return loss_parsing + loss_edge + loss_pose

    def l2_loss(self, input, target, mask, num_pix):
        loss = (input - target) * mask
        loss = (loss * loss)/2
        return loss.sum()/num_pix

    def forward(self, preds, target, writer, i_iter, total_iters):
        loss = self.loss(preds, target, writer, i_iter, total_iters)
        return loss