import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from losses import get_losses_unlabeled, BCE_soft_labels, sigmoid_rampup
# from model.resnet import resnet34
from utils.get_data import get_data
from model.basenet import Predictor
from utils.utils import get_classlist, lr_scheduler

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--train_steps', type=int, default=50000)
parser.add_argument('--rampup_coeff', type=float, default=30.0)
parser.add_argument('--rampup_length', type=int, default=20000)
parser.add_argument('--threshold', default=0.95, type=float)
parser.add_argument('--lr_multiplier', default=0.1, type=float)
logs_file = '' # ??
checkpath = '' # ??


args = parser.parse_args()

torch.manual_seed(1)
device = torch.device("mps")
resnet_output_vector_size = 1000 # ??

source_annotation_path = 'data/annotations/labeled_source_images_webcam.txt'
class_list = get_classlist(source_annotation_path)
num_class = len(class_list)

feature_extractor = resnet34(pretrained=True)
# feature_extractor = resnet34()
feature_extractor.fc = nn.Identity()
predictor = Predictor(num_class=num_class, input_vector_size=512, norm_factor=args.temperature)
nn.init.xavier_normal_(predictor.fc.weight)
# nn.init.zeros_(predictor.fc.bias)

feature_extractor_params = []
for key, value in dict(feature_extractor.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            feature_extractor_params += [{'params': [value], 'lr': args.lr_multiplier, 'weight_decay': 0.0005}]
        else:
            feature_extractor_params += [{'params': [value], 'lr': args.lr_multiplier * 10, 'weight_decay': 0.0005}]

feature_extractor = nn.DataParallel(feature_extractor)
predictor = nn.DataParallel(predictor)
feature_extractor = feature_extractor.to(device)
predictor = predictor.to(device)

opt = {}
opt['logs_file'] = logs_file
opt['checkpath'] = checkpath
opt['class_list'] = class_list

labeled_datasets, unlabled_target_dataset, target_dataset_val = get_data(args)

labeled_dataloader = DataLoader(labeled_datasets, batch_size=args.batch_size, num_workers=0, shuffle=True,
                                drop_last=True)
unlabled_target_data_loader = DataLoader(unlabled_target_dataset, batch_size=args.batch_size, num_workers=0,
                                         shuffle=True, drop_last=True)

target_val_dataloader = DataLoader(target_dataset_val, batch_size=min(args.batch_size, len(target_dataset_val)),
                                   num_workers=0, shuffle=True, drop_last=True)

labeled_len = len(labeled_dataloader)
unlabeled_len = len(unlabled_target_data_loader)
val_dataset_len = len(target_dataset_val)

print("hi")
writer = SummaryWriter()


def train():
    labeled_data_iter = iter(labeled_dataloader)
    unlabled_data_iter = iter(unlabled_target_data_loader)

    feature_extractor.train()
    predictor.train()

    optimizer_feature_extractor = optim.SGD(feature_extractor_params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_predictor = optim.SGD(list(predictor.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005,
                                    nesterov=True)

    params_lr_feat_extractor = []
    for param_group in optimizer_feature_extractor.param_groups:
        params_lr_feat_extractor.append(param_group["lr"])
    params_lr_predictor = []
    for param_group in optimizer_predictor.param_groups:
        params_lr_predictor.append(param_group["lr"])

    BCE = BCE_soft_labels().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    for step in range(args.train_steps):

        optimizer_feature_extractor = lr_scheduler(params_lr_feat_extractor, optimizer_feature_extractor, step, init_lr=args.learning_rate)
        optimizer_predictor = lr_scheduler(params_lr_predictor, optimizer_predictor, step, init_lr=args.learning_rate)

        lr_g = optimizer_feature_extractor.param_groups[0]['lr']
        lr_f = optimizer_predictor.param_groups[0]['lr']

        if step % 100 == 0:
            log_train = 'Train Ep: {} lr_f{:.6f} lr_g{:.6f}\n'.format(step, lr_f, lr_g)
            print(log_train)

        if step % labeled_len:
            labeled_data_iter = iter(labeled_dataloader)
        if step % unlabeled_len:
            unlabled_data_iter = iter(unlabled_target_data_loader)

        labeled_data_iter_next = next(labeled_data_iter)
        labeled_data_images = labeled_data_iter_next[0].to(device)
        labeled_data_labels = labeled_data_iter_next[1].to(device)
        features = feature_extractor(labeled_data_images)
        predictions = predictor(features)

        cross_entropy_loss = criterion(predictions, labeled_data_labels)
        writer.add_scalar("Loss/train/cross entropy", cross_entropy_loss, step)

        cross_entropy_loss.backward(retain_graph=True)
        optimizer_feature_extractor.step()
        optimizer_predictor.step()
        optimizer_feature_extractor.zero_grad()
        optimizer_predictor.zero_grad()

        # calculate loss for unlabeled target data
        unlabeled_data_iter_next = next(unlabled_data_iter)
        unlabeled_data_images = unlabeled_data_iter_next[0].type(torch.FloatTensor).to(device)
        unlabeled_data_images_t = unlabeled_data_iter_next[2].type(torch.FloatTensor).to(device)
        unlabeled_data_images_t2 = unlabeled_data_iter_next[3].type(torch.FloatTensor).to(device)
        unlabeled_data_labels = unlabeled_data_iter_next[1].to(device)

        rampup = sigmoid_rampup(step, args.rampup_length)
        w_consistency = args.rampup_coeff * rampup

        # adversarial_adaptive_clustering_loss, pseudo_labels_loss, consistency_loss = get_losses_unlabeled(args,
        #                                     feature_extractor, predictor, unlabeled_data_images, unlabeled_data_images_t,
        #                                     unlabeled_data_images_t2, unlabeled_data_labels, BCE, w_consistency, device)

        # loss = adversarial_adaptive_clustering_loss + pseudo_labels_loss + consistency_loss
        loss = -1
        # writer.add_scalar("Loss/train/unlabeled", loss, step)
        # loss.backward()
        # optimizer_feature_extractor.step()
        # optimizer_predictor.step()
        # optimizer_feature_extractor.zero_grad()
        # optimizer_predictor.zero_grad()

        if step % 10 == 0:
            print("step: " + str(step) + ". ce loss: " + str(cross_entropy_loss) + ". unlabeled loss: " + str(loss))

        if step % 100 == 0:
            val_loss, val_accuracy = test(target_val_dataloader)


def test(dataloader):
    feature_extractor.eval()
    predictor.eval()
    test_loss = 0
    num_correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().to(device)
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(dataloader):
            im_data_t = data_t[0].to(device)
            gt_labels_t = data_t[1].to(device)
            features = feature_extractor(im_data_t)
            predictions = predictor(features)
            output_all = np.r_[output_all, predictions.data.cpu().numpy()]
            size += im_data_t.size(0)
            predictions1 = predictions.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), predictions1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            num_correct += predictions1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(predictions, gt_labels_t) / val_dataset_len
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)\n'.format(test_loss, num_correct, size,
                                                                                    100. * num_correct / size))
    feature_extractor.train()
    predictor.train()
    return test_loss.data, 100 * float(num_correct) / size

train()
writer.flush()