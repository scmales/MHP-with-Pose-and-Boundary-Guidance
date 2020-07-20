import numpy as np
import torchvision
import torch
from torch.nn import functional as F

# colour map
COLORS = [(0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]


def decode_parsing(labels, num_images=1, is_pred=False):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    pred_labels = labels[:num_images].clone().cpu().data

    if is_pred:
        pred_labels = torch.argmax(pred_labels, dim=1)
    n, h, w = pred_labels.size()

    labels_color = torch.zeros([n, 3, h, w], dtype=torch.uint8)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]

    return labels_color

def decode_heatmap(heatmap, num_images=1):
    heatmap_t = heatmap[:num_images].clone().cpu().data
    # n, c, h, w = heatmap_t.size()
    heatmap_t = np.array(heatmap_t)
    heatmap_t = heatmap_t[:, :-1, :, :].max(axis=1)
    heatmap_t = torch.from_numpy(heatmap_t)
    heatmap_t = heatmap_t.unsqueeze(dim=1)
    return heatmap_t
def decode_pafs(pafs, num_images=1):
    pafs_t = pafs[:num_images].clone().cpu().data
    # n, c, h, w = heatmap_t.size()
    pafs_t = np.array(pafs_t)
    pafs_t = np.abs(pafs_t).sum(axis=1)
    pafs_t = torch.from_numpy(pafs_t)
    pafs_t = pafs_t.unsqueeze(dim=1)
    return pafs_t
def inv_preprocess(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    rev_imgs = imgs[:num_images].clone().cpu().data
    rev_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(num_images):
        rev_imgs[i] = rev_normalize(rev_imgs[i])

    return rev_imgs

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)
