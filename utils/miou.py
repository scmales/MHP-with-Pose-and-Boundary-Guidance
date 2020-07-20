import numpy as np
import cv2
import os
from collections import OrderedDict
from utils.transforms import transform_parsing

LABELS_PASCAL = ['Background', 'head', 'torso', 'upper-arm', 'lower-arm', 'upper-leg', 'lower-leg']
def get_palette(num_cls):

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(gt_label, pred_label, num_classes):

    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size, dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    labels_dataset = LABELS_PASCAL
    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, 'pascal_person_part_gt', im_name + '.png')
        labels_dataset = LABELS_PASCAL

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)  # 列之和是每类类预测的数
    res = confusion_matrix.sum(0)  # 行之和是每类实际上的数
    tp = np.diag(confusion_matrix) # 每类预测正确的数

    pixel_accuracy = (tp.sum() / pos.sum()) * 100 #预测正确的除以总数
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    name_value = []

    for i, (label, iou) in enumerate(zip(labels_dataset, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IoU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value
