import argparse
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from networks.model import ParsingNet
from dataloaders.pascal import VOCSegmentation
import torchvision.transforms as transforms
import timeit
from tensorboardX import SummaryWriter
from utils.utils import decode_parsing, inv_preprocess, decode_heatmap, decode_pafs
from utils.criterion import Criterion
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.miou import compute_mean_ioU
from evaluate import valid
from utils.get_parameters import *

start = timeit.default_timer()

BATCH_SIZE = 2
DATA_DIRECTORY = './dataloaders/pascal_person_pose_and_part'
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
INPUT_SIZE = [384, 384]
GPU_IDS = "0"
SNAPSHOT_DIR = './snapshots/scalar_n'
EPOCHS = 300

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
START_EPOCH = 0
MOMENTUM = 0.9
POWER = 0.9
IGNORE_LABEL = 255

RESNET_IMAGENET = "./networks/resnet101-imagenet.pth"
RESTORE_FROM_PARSING='./snapshots/PASCAL_parsing_104.pth'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train Arguments")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--stride", type=int, default=8,
                        help="Scale heatmap and pafs for calculating loss")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from-parsing", type=str, default=RESTORE_FROM_PARSING,
                        help="Where restore parsingnet parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPU_IDS,
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="choose the number of recurrence.")
    parser.add_argument("--train-continue", type=int, default=0,
                        help="Whether to train continue.")
    parser.add_argument("--print-val", type=int, default=0,
                        help="Whether to print val.")
    parser.add_argument("--data-name", type=str, default='pascal',
                        help="choose different dataset for training.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_parsing_lr(optimizer, i_iter, total_iters):
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def main():
    """Create the model and start the training."""
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer = SummaryWriter(args.snapshot_dir)
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    h, w = [int(i) for i in args.input_size.split(',')]
    input_size = [h, w]
    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  ##为何使用了非确定性的卷积
    torch.backends.cudnn.enabled = True
    NUM_CLASSES = 7  # parsing
    NUM_HEATMAP = 15  # pose
    NUM_PAFS = 28  # pafs
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    # load dataset
    num_samples = 0
    trainloader = data.DataLoader(
        VOCSegmentation(args.data_dir, args.dataset, crop_size=input_size, stride=args.stride, transform=transform),
        batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=2,
        pin_memory=True)

    valloader = None
    if args.print_val != 0:
        valdataset = VOCSegmentation(args.data_dir, 'val', crop_size=input_size, transform=transform)
        num_samples = len(valdataset)
        valloader = data.DataLoader(valdataset, batch_size=8 * len(gpus),  # batchsize
                                     shuffle=False, pin_memory=True)

    parsingnet = ParsingNet(num_classes=NUM_CLASSES, num_heatmaps=NUM_HEATMAP, num_pafs=NUM_PAFS)
    criterion_parsing = Criterion()
    criterion_parsing = DataParallelCriterion(criterion_parsing)
    criterion_parsing.cuda()

    optimizer_parsing = optim.SGD(
        parsingnet.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    optimizer_parsing.zero_grad()
    # 加载预训练参数
    print(args.train_continue)
    if not args.train_continue:
        checkpoint = torch.load(RESNET_IMAGENET)
        load_state(parsingnet, checkpoint)
    else:
        checkpoint = torch.load(args.restore_from_parsing)
        if 'current_epoch' in checkpoint:
            current_epoch = checkpoint['current_epoch']
            args.start_epoch = current_epoch

        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        load_state(parsingnet, checkpoint)



    parsingnet = DataParallelModel(parsingnet).cuda()
    total_iters = args.epochs * len(trainloader)
    for epoch in range(args.start_epoch, args.epochs):
        parsingnet.train()
        for i_iter, batch in enumerate(trainloader):
            i_iter += len(trainloader) * epoch
            lr = adjust_parsing_lr(optimizer_parsing, i_iter, total_iters)

            images, labels, edges, heatmap, pafs, heatmap_mask, pafs_mask, _ = batch
            images = images.cuda()
            labels = labels.long().cuda(non_blocking=True)
            edges = edges.long().cuda(non_blocking=True)
            heatmap = heatmap.cuda()
            pafs = pafs.cuda()
            heatmap_mask = heatmap_mask.cuda()
            pafs_mask = pafs_mask.cuda()


            preds = parsingnet(images)
            loss_parsing = criterion_parsing(preds, [labels, edges, heatmap, pafs, heatmap_mask, pafs_mask], writer,i_iter,total_iters)
            optimizer_parsing.zero_grad()
            loss_parsing.backward()
            optimizer_parsing.step()
            if i_iter % 100 == 0:
                writer.add_scalar('parsing_lr', lr, i_iter)
                writer.add_scalar('loss_total', loss_parsing.item(), i_iter)
            if i_iter % 500 == 0:

                if len(gpus) > 1:
                    preds = preds[0]

                images_inv = inv_preprocess(images, args.save_num_images)
                parsing_labels_c = decode_parsing(labels, args.save_num_images, is_pred=False)
                preds_colors = decode_parsing(preds[0][-1], args.save_num_images, is_pred=True)
                edges_colors = decode_parsing(edges, args.save_num_images, is_pred=False)
                pred_edges = decode_parsing(preds[1][-1], args.save_num_images, is_pred=True)

                img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
                parsing_lab = vutils.make_grid(parsing_labels_c, normalize=False, scale_each=True)
                pred_v = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
                edge = vutils.make_grid(edges_colors, normalize=False, scale_each=True)
                pred_edges = vutils.make_grid(pred_edges, normalize=False, scale_each=True)

                writer.add_image('Images/', img, i_iter)
                writer.add_image('Parsing_labels/', parsing_lab, i_iter)
                writer.add_image('Parsing_Preds/', pred_v, i_iter)

                writer.add_image('Edges/', edge, i_iter)
                writer.add_image('Edges_preds/', pred_edges, i_iter)

        if (epoch + 1) % 15 == 0:
            if args.print_val != 0:
                parsing_preds, scales, centers = valid(parsingnet, valloader, input_size, num_samples, gpus)
                mIoU = compute_mean_ioU(parsing_preds, scales, centers, NUM_CLASSES, args.data_dir, input_size)
                f = open(os.path.join(args.snapshot_dir, "val_res.txt"), "a+")
                f.write(str(epoch)+str(mIoU) + '\n')
                f.close()
            snapshot_name_parsing = osp.join(args.snapshot_dir, 'PASCAL_parsing_' + str(epoch) + '' + '.pth')
            torch.save({'state_dict': parsingnet.state_dict(),
                        'optimizer': optimizer_parsing.state_dict(),
                        'current_epoch': epoch},
                       snapshot_name_parsing)

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()