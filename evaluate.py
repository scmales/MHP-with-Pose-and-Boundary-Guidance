import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.model import ParsingNet
from dataloaders.pascal import VOCSegmentation
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from torch.nn.functional import interpolate
from utils.keypoints import *
from utils.get_parameters import *
DATA_DIRECTORY = './dataloaders/pascal_person_pose_and_part'
IGNORE_LABEL = 255
INPUT_SIZE = '384, 384'
BATCH_SIZE = 2


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Arguments")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--data-name", type=str, default='pascal',
                        help="choose different dataset for training.")
    return parser.parse_args()

def valid(parsingnet, valloader, input_size, num_samples, gpus):
    parsingnet.eval()
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    # interp = interpolate(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            image = image.cuda()
            num_images, _, h, w = image.size()
            # if index % 100 == 0:
            #     print('%d  processd' % (index * num_images))
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs = parsingnet(image)
            if len(gpus) > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interpolate(parsing, size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                    idx += nums
            else:
                parsing = outputs[0][-1]
                parsing = interpolate(parsing, size=(input_size[0], input_size[1]), mode='bilinear',
                                      align_corners=True).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]
    return parsing_preds, scales, centers

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    NUM_CLASSES = 7  # parsing
    NUM_HEATMAP = 15  # pose
    NUM_PAFS = 28  # pafs

    model = ParsingNet(num_classes=NUM_CLASSES, num_heatmaps=NUM_HEATMAP, num_pafs=NUM_PAFS)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    valdataset = VOCSegmentation(DATA_DIRECTORY, args.dataset, crop_size=input_size, transform=transform)
    num_samples = len(valdataset)
    num_classes = NUM_CLASSES
    valloader = data.DataLoader(valdataset, batch_size=args.batch_size * len(gpus),  #batchsize
                                shuffle=False, pin_memory=True)
    restore_from = args.restore_from
    try:
        state_dict = torch.load(restore_from)['state_dict']
        load_state(model, state_dict)

        model.eval()
        model.cuda()

        parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, gpus)

        mIoU = compute_mean_ioU(parsing_preds, scales, centers, num_classes, args.data_dir, input_size)
        print(str(mIoU))
    except:
        print("load model error")
if __name__ == '__main__':
    main()
