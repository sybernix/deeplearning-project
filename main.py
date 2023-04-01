import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from losses import get_losses_unlabeled, BCE_soft_labels, sigmoid_rampup
from utils.get_data import get_data
from model.basenet import Predictor
from utils.utils import get_classlist

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--train_steps', type=int, default=1000)
parser.add_argument('--rampup_coeff', type=float, default=30.0)
parser.add_argument('--rampup_length', type=int, default=20000)
parser.add_argument('--threshold', default=0.95, type=float)
logs_file = '' # ??
checkpath = '' # ??


args = parser.parse_args()

device = torch.device("cpu")
resnet_output_vector_size = 1000 # ??

source_annotation_path = 'data/annotations/labeled_source_images_webcam.txt'
class_list = get_classlist(source_annotation_path)
num_class = len(class_list)

feature_extractor = resnet34(pretrained=True)
predictor = Predictor(num_class=num_class, input_vector_size=resnet_output_vector_size, norm_factor=args.temperature)
nn.init.xavier_normal_(predictor.fc.weight)
# nn.init.zeros_(F.fc.bias)

feature_extractor = nn.DataParallel(feature_extractor)
predictor = nn.DataParallel(predictor)
feature_extractor = feature_extractor.to(device)
predictor = predictor.to(device)

opt = {}
opt['logs_file'] = logs_file
opt['checkpath'] = checkpath
opt['class_list'] = class_list

labeled_datasets, unlabled_target_dataset = get_data(args)

labeled_dataloader = DataLoader(labeled_datasets, batch_size=args.batch_size, num_workers=0, shuffle=True,
                                drop_last=True)
unlabled_target_data_loader = DataLoader(unlabled_target_dataset, batch_size=args.batch_size, num_workers=0,
                                         shuffle=True, drop_last=True)

labeled_len = len(labeled_dataloader)
unlabeled_len = len(unlabled_target_data_loader)

print("hi")
writer = SummaryWriter()


def train():
    labeled_data_iter = iter(labeled_dataloader)
    unlabled_data_iter = iter(unlabled_target_data_loader)

    feature_extractor.train()
    predictor.train()

    optimizer_feature_extractor = optim.SGD(feature_extractor.parameters(), lr=args.learning_rate, momentum=0.9,
                                            weight_decay=0.0005, nesterov=True)
    optimizer_predictor = optim.SGD(predictor.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005,
                                    nesterov=True)

    BCE = BCE_soft_labels().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    for step in range(args.train_steps):

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

        adversarial_adaptive_clustering_loss, pseudo_labels_loss, consistency_loss = get_losses_unlabeled(args,
                                            feature_extractor, predictor, unlabeled_data_images, unlabeled_data_images_t,
                                            unlabeled_data_images_t2, unlabeled_data_labels, BCE, w_consistency, device)

        loss = adversarial_adaptive_clustering_loss + pseudo_labels_loss + consistency_loss
        writer.add_scalar("Loss/train/unlabeled", loss, step)
        loss.backward()
        optimizer_feature_extractor.step()
        optimizer_predictor.step()
        optimizer_feature_extractor.zero_grad()
        optimizer_predictor.zero_grad()

        if step % 10 == 0:
            print("step: " + str(step) + ". ce loss: " + str(cross_entropy_loss) + ". unlabeled loss: " + str(loss))


train()
writer.flush()