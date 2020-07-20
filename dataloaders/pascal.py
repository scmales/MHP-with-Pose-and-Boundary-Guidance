from __future__ import print_function, division
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch
from utils.transforms import *
import os
from utils.keypoints import *
from scipy.io import loadmat

# conns = ((1, 5), (5, 6), (6, 7), (1, 11), (11, 12), (12, 13),
#               (1, 2), (2, 3), (3, 4), (1, 8), (8, 9), (9, 10), (1, 0))  # pafs: 有26对
conns = ((5, 6), (6, 7), (11, 12), (12, 13),(5,11),(2,5), (8,11),
              (1, 2), (2, 3), (3, 4), (1, 8), (8, 9), (9, 10), (1, 0))  # pafs: 有28对
class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    def __init__(self, root, dataset, crop_size=[384, 384], scale_factor=0.25,
                 rotation_factor=30, transform=None,
                 sigma=7, stride=8, num_heatmaps=15, num_pafs=28):
        self.root = root
        self.crop_size = np.asarray(crop_size)
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]  # 需要裁剪的长宽比
        # self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        # self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]] 竟然是保留用于LIP骨架交换的参数
        self.transform = transform
        self.dataset = dataset
        self.num_heatmaps = num_heatmaps
        self.num_pafs = num_pafs
        self.sigma = sigma
        self.stride = stride
        list_path = os.path.join(self.root, self.dataset + '_id.txt')
        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.number_samples = len(self.im_list)
    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:   #按长度长的一边缩放比例
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)  #scale为缩放后的结果

        return center, scale
    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]
        im_path = os.path.join(self.root, 'JPEGImages', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root,  'pascal_person_part_gt', im_name + '.png')
        if self.dataset != 'val':
            pose_anno_path = os.path.join(self.root,  'pose_annotations', im_name + '.mat')
            pose_anno = loadmat(pose_anno_path)
            kpts_persons = pose_anno['joints'][0]  # shape:[num,] num是人数,其中每个人=> shape:[14, 3]] array中的对象是array
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])  # center为原图缩放中心, s是缩放后的宽,长(输入长宽比得到)
        r = 0
        flag_flip = False

        parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
        if self.dataset == 'train':
            sf = self.scale_factor                                      #缩放因子 0.25
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf) #缩放因子在0.75 -1.25之间 乘上s
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0  #有随机数小于0.6的概率旋转因子在-60到60之间,否则不旋转

            if random.random() <= self.flip_prob:
                flag_flip = True
                im = im[:, ::-1, :]
                parsing_anno = parsing_anno[:, ::-1]

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        if self.transform:
            input = self.transform(input)

        #center为原图的缩放中心
        #图片先按crop缩放比例按长边缩放,再按缩放因子概率缩放得到s为最终缩放长和宽
        #r为旋转角度
        #做完仿射变换后再裁剪成输入大小
        meta = {
            'name': im_name,
            'center': center,  #缩放中心
            'height': h,       #原长
            'width': w,        #原宽
            'scale': s,        #缩放因子
            'rotation': r,     #旋转因子
        }

        if self.dataset == 'val':
            return input, meta
        else:
            ################################## 计算heatmap并叠加成多个通道 ############################################
            t_h = int(self.crop_size[0]/self.stride,)
            t_w = int(self.crop_size[1]/self.stride)
            heatmap = np.zeros([t_h, t_w, self.num_heatmaps], np.float32)  # 这个num是关节点+1的情况,+1是背景
            pafs = np.zeros([t_h, t_w, self.num_pafs], np.float32)
            person_num = pose_anno['boxes'].shape[1]
            for i in range(0, person_num):
                kpts = kpts_persons[i]
                kpts = self.transform_kpt(kpts, trans, w, flag_flip)    # shape : [14, 3]
                kpts = np.expand_dims(kpts, axis=0)                     # shape : [1, 14, 3]
                heatmap[:, :, :-1] = np.maximum(heatmap[:, :, :-1],
                                             genHeatmaps(kpts, self.crop_size[0], self.crop_size[1], self.sigma, self.stride))
                pafs += genPafs(kpts, conns, self.crop_size[0], self.crop_size[1], self.stride)  # 本来重叠的pafs部分应该取多人平均,但这个数据集情况比较少,直接累加好了
            heatmap[:, :, -1] = 1 - heatmap.max(axis=2)                 # 前面的仿射变换填充了0,最后一层为了不填充0
            ####################################################################################################

            label_edge = generate_edge(parsing_anno)
            label_edge = cv2.warpAffine(
                label_edge,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255))  # when cal loss, ignore 255
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255))

            #生成关节相关的mask
            mask = label_parsing.copy()
            mask[mask != 255] = 1
            mask[mask == 255] = 0
            mask = cv2.resize(mask, dsize=None, fx=1 / self.stride, fy=1 / self.stride, interpolation=cv2.INTER_AREA)
            heatmap_mask = np.ones(shape=heatmap.shape, dtype=np.float32)
            pafs_mask = np.ones(shape=pafs.shape, dtype=np.float32)
            for i in range(heatmap_mask.shape[2]):
                heatmap_mask[:, :, i] = mask
            for i in range(pafs_mask.shape[2]):
                pafs_mask[:, :, i] = mask

            label_parsing = torch.from_numpy(label_parsing)
            label_edge = torch.from_numpy(label_edge)
            heatmap = torch.from_numpy(heatmap.transpose([2, 0, 1]))
            pafs = torch.from_numpy(pafs.transpose([2, 0, 1]))
            heatmap_mask = torch.from_numpy(heatmap_mask.transpose([2, 0, 1]))
            pafs_mask = torch.from_numpy(pafs_mask.transpose([2, 0, 1]))

            return input, label_parsing, label_edge, heatmap, pafs, heatmap_mask, pafs_mask, meta


    def transform_kpt(self, kpts, M, w, flag_flip=False):
        if(flag_flip): # 若水平翻转 和宽做差
            kpts[:, 0] = w-kpts[:, 0]
            flip_pair_heatmap = [(2, 8), (3, 9), (4, 10), (5, 11), (6, 12), (7, 13)]
            for i, j in flip_pair_heatmap:
               kpts[[i, j], :] = kpts[[j, i], :]
        pos_kpts = np.ones((14, 3))
        pos_kpts[:, :-1] = kpts[:, :-1]
        pos_kpts = pos_kpts.transpose([1, 0])

        res_kpts = np.dot(M, pos_kpts).transpose((1, 0))  # (2,3)(3, 14) = (2 ,14)

        kpts[:, :-1] = res_kpts
        return kpts


    def data_vision(self, images, parsing_label, edge_label, kpts_persons):
        # (384, 384, 3) (384, 384) (384,384) [n, 14, 3]
        COLORS = [(0, 0, 0)
            , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
            , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
            , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
            , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
        cv2.imshow("img", images)
        cv2.waitKey(0)
        parsing_color = np.zeros(images.shape, dtype=np.uint8)
        for i, c in enumerate(COLORS):
            c0 = parsing_color[:, :, 0]
            c1 = parsing_color[:, :, 1]
            c2 = parsing_color[:, :, 2]

            c0[parsing_label == i] = c[0]
            c1[parsing_label == i] = c[1]
            c2[parsing_label == i] = c[2]
        parsing_color = parsing_color[:, :, ::-1]
        cv2.imshow("parsing_label", parsing_color)
        cv2.waitKey(0)

        cv2.imshow("edge_label", edge_label.astype(np.float32))
        cv2.waitKey(0)

        r = 4
        color = [(255, 0, 0), (0, 255, 0)]  # 红色代表可见,绿色代表被遮挡
        for kpts in kpts_persons:
            for i in range(len(conns)):
                kpt1 = kpts[conns[i][0]] # [33, 44, 1]
                kpt2 = kpts[conns[i][1]]
                x1 = kpt1[0]
                y1 = kpt1[1]
                v1 = kpt1[2]

                x2 = kpt2[0]
                y2 = kpt2[1]
                v2 = kpt2[2]
                # if v1>0.5 and v1<1.5:
                #     cv2.circle(parsing_color, (int(x1), int(y1)), r, color[0], -1)
                # elif v1>1.5 and v1<2.5:
                #     cv2.circle(parsing_color, (int(x1), int(y1)), r, color[1], -1)
                # if v2>0.5 and v2<1.5:
                #     cv2.circle(parsing_color, (int(x2), int(y2)), r, color[0], -1)
                # elif v2>1.5 and v2<2.5:
                #     cv2.circle(parsing_color, (int(x2), int(y2)), r, color[1], -1)

                if v1 > 0.5 and v1<1.5 and v2 > 0.5 and v2<1.5:  # nan表示图中没有
                    cv2.line(images, (int(x1), int(y1)), (int(x2), int(y2)), color[0], 5)
        # cv2.imshow("joint", parsing_color)
        # cv2.waitKey(0)
        cv2.imshow("img", images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from utils.keypoints import *
    from networks.model import PEPNet
    import matplotlib.pyplot as plt
    import collections
    root = "/home/dsc/scmales_git/PPBG/dataloaders/pascal_person_pose_and_part"
    restore_from = "/home/dsc/scmales_git/PPBG/snapshots/PASCAL_parsing_104.pth"
    model = PEPNet(num_classes=7, num_heatmaps=15, num_pafs=26)
    source_state = {k.replace('module.', ''): v for k, v in torch.load(restore_from)['state_dict'].items()}
    target_state = model.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
    model.load_state_dict(new_target_state)

    model = model.eval().cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # testloader = DataLoader(VOCSegmentation(root, "test", transform=transform, sigma=7, stride=8), batch_size=1, shuffle=False, num_workers=0)

    trainloader = DataLoader(VOCSegmentation(root, "train", transform=transform, sigma=7, stride=8), batch_size=1, shuffle=True, num_workers=0)
    for ii, sample in enumerate(trainloader):
        images, labels, edges, heatmap,  pafs, heatmap_mask, pafs_mask,  meta = sample

        # print(images.dtype)
        # print(images.min())
        # print(labels.dtype)
        # print(edges.dtype)
        # print(heatmap.dtype)
        # print(pafs.dtype)
        # print(heatmap_mask.dtype)
        # print(pafs_mask.dtype)

        # index = np.where(heatmap == 255)
        # heatmap[index[0], index[1], index[2], index[3]] = 0.

        images = np.array(images).squeeze(0).transpose((1, 2, 0))
        print(labels.shape)
        labels = np.array(labels).squeeze(0)

        plt.imshow(images)
        plt.show()
        plt.imshow(labels)
        plt.show()

        # heatmap = np.array(heatmap).squeeze(0)
        # heatmap_mask = np.array(heatmap_mask).squeeze(0)
        #
        # pafs = np.array(pafs).squeeze(0)
        # pafs_mask = np.array(pafs_mask).squeeze(0)
        #
        #
        #
        # for i in range(14, 15):
        #     pafs_v = heatmap[i, :, :]
        #     plt.imshow(pafs_v)
        #     plt.show()
        #
        #     pafs_mask_v = heatmap_mask[i, :, :]
        #     plt.imshow(pafs_mask_v)
        #     plt.show()
        #
        # pafs = np.abs(heatmap).max(axis=0)
        # plt.imshow(pafs)
        # plt.show()

        if ii == 1:
            break

    # for ii, sample in enumerate(testloader):
    #     if ii == 10:  # 2有3个人, 3有2个人
    #         images, meta = sample
    #         plt.imshow(np.array(images).squeeze(0).transpose([1, 2, 0]))
    #         plt.show()
    #
    #         im_name = meta['name']
    #         c = meta['center'].numpy()
    #         s = meta['scale'].numpy()
    #         w = meta['width']
    #         h = meta['height']
    #
    #         ############## 通过文件名生成原图的 heatmap 和 pafs ##############
    #         for i in range(len(im_name)):
    #             pose_anno_path = os.path.join(root, 'pose_annotations', im_name[i] + '.mat')
    #             pose_anno = loadmat(pose_anno_path)
    #             kpts_persons = pose_anno['joints'][0]  # shape:[num, 14, 3]  num是人数
    #             person_num = len(kpts_persons)
    #             gt_heatmap = np.zeros([h[i], w[i], 15], np.float32)  # 这个num是关节点+1的情况,+1是背景
    #             gt_pafs = np.zeros([h[i], w[i], 26], np.float32)
    #             for j in range(0, person_num):
    #                 kpts = kpts_persons[j]  # [1, 14, 3]
    #                 kpts = np.expand_dims(kpts, axis=0)
    #                 gt_heatmap[:, :, :-1] = np.maximum(gt_heatmap[:, :, :-1],
    #                                                    genHeatmaps(kpts, h[i], w[i], sigma=7, stride=1))
    #                 gt_pafs += genPafs(kpts, conns, h[i], w[i], stride=1)  # 本来重叠的pafs部分应该取多人平均,但这个数据集情况比较少,直接累加好了
    #             gt_heatmap[:, :, -1] = 1 - gt_heatmap.max(axis=2)  # 前面的仿射变换填充了0,最后一层为了不填充0
    #             plt.imshow(gt_heatmap[:, :, :-1].max(axis=2))
    #             plt.show()
    #             plt.imshow(gt_pafs.sum(axis=2))
    #             plt.show()
    #             ###################### 提取骨架
    #             # total_keypoints_num = 0
    #             # all_keypoints_by_type = []
    #             # for kpt_idx in range(14):
    #             #     total_keypoints_num += extract_keypoints(gt_heatmap[:, :, kpt_idx], all_keypoints_by_type,
    #             #                                              total_keypoints_num)
    #             # pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, gt_pafs)
    #             # # print(all_keypoints)  # [num_valid_kpts, 4] 4个维度为[w, h, score, index]
    #             # # print(pose_entries)   # [num_person, 14+2] 14为关节点, 2　[total_score, sum_num_keypoints]
    #             # kpts_persons = kpts_trans_format(pose_entries, all_keypoints)
    #             # print(kpts_persons)
    #         ###d = model(images.cuda())
    #     #         heatmap = pred[0][-2].cpu().detach()
    #     #         # heatmap = F.interpolate(heatmap, size=(384, 384), mode='bilinear', align_corners=True).cpu().detach().numpy()
    #     #         heatmap = np.array(heatmap).transpose([0, 2, 3, 1])
    #     #         pafs = pred[-1][-1].cpu().detach()
    #     #         # pafs = F.interpolate(pafs, size=(384, 384), mode='bilinear', align_corners=True).cpu().detach().numpy()
    #     #         pafs = np.array(pafs).transpose([0, 2, 3, 1])
    #     #         plt.imshow(heatmap.squeeze(0)[:, :, :-1].max(axis=2))
    #     #         plt.show()
    #     #         plt.imshow(pafs.squeeze(0).sum(axis=2))
    #     #         plt.show()
    #     #         input_size = [384, 384]
    #     #         for i, (t_heatmap, t_pafs) in enumerate(zip(heatmap, pafs)):  # 按batch展开分别求关节点和骨架
    #     #             t_heatmap = cv2.resize(t_heatmap, (384, 384), interpolation=cv2.INTER_CUBIC)
    #     #             t_pafs = cv2.resize(t_pafs, (384, 384), interpolation=cv2.INTER_LINEAR)
    #     #             trans = get_affine_transform(c[i], s[i], 0, input_size, inv=1)
    #     #             t_heatmap = cv2.warpAffine(
    #     #                 t_heatmap,
    #     #                 trans,
    #     #                 (int(w[i]), int(h[i])),
    #     #                 flags=cv2.INTER_NEAREST,
    #     #                 borderMode=cv2.BORDER_CONSTANT,
    #     #                 borderValue=(0))
    #     #             t_pafs = cv2.warpAffine(
    #     #                 t_pafs,
    #     #                 trans,
    #     #                 (int(w[i]), int(h[i])),
    #     #                 flags=cv2.INTER_NEAREST,
    #     #                 borderMode=cv2.BORDER_CONSTANT,
    #     #                 borderValue=(0))
    #     #             plt.imshow(t_heatmap[:, :, :-1].max(axis=2))
    #     #             plt.show()
    #     #             plt.imshow(t_pafs.sum(axis=2))
    #     #             plt.show()
    #     #             # total_keypoints_num = 0
    #     #             # all_keypoints_by_type = []
    #     #             # for kpt_idx in range(14):
    #     #             #     total_keypoints_num += extract_keypoints(t_heatmap[:, :, kpt_idx], all_keypoints_by_type,
    #     #             #                                              total_keypoints_num)
    #     #             # pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, t_pafs)
    #     #             # # print(all_keypoints)  # [num_valid_kpts, 4] 4个维度为[w, h, score, index]
    #     #             # # print(pose_entries)   # [num_person, 14+2] 14为关节点, 2　[total_score, sum_num_keypoints]
    #     #             # kpts_persons = kpts_trans_format(pose_entries, all_keypoints)
    #     #             # print(kpts_persons)
    #     #         break
    #         # np.set_printoptions(threshold=np.inf)######################把预测结果heatmap恢复成原图尺寸#####
    #         pre


