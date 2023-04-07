import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_losses_unlabeled(args, feature_extractor, predictor, unlabeled_data_images, unlabeled_data_images_transformed1,
                         unlabeled_data_images_transformed2, target, binary_cross_entropy, w_consistency, device):
    features = feature_extractor(unlabeled_data_images)
    features_transformed1 = feature_extractor(unlabeled_data_images_transformed1)
    features_transformed2 = feature_extractor(unlabeled_data_images_transformed2)

    predictions = predictor(features, reversed=True, lambd=1.0)
    predictions_transformed1 = predictor(features_transformed1, reversed=True, lambd=1.0)

    probabilities = F.softmax(predictions, dim=1)
    probabilities_transformed1 = F.softmax(predictions_transformed1, dim=1)

    # calculate adversarial adaptive clustering loss
    adversarial_adaptive_clustering_loss = adversarial_adaptive_clustering_loss_unlabeled(args, features, target,
                                                                                          probabilities,
                                                                                          probabilities_transformed1,
                                                                                          binary_cross_entropy, device)
    predictions = predictor(features)
    predictions_transformed1 = predictor(features_transformed1)
    predictions_transformed2 = predictor(features_transformed2)

    probabilities = F.softmax(predictions, dim=1)
    probabilities_transformed1 = F.softmax(predictions_transformed1, dim=1)
    probabilities_transformed2 = F.softmax(predictions_transformed2, dim=1)

    max_probabilities, pseudo_labels = torch.max(probabilities.detach_(), dim=1)
    mask = max_probabilities.ge(args.threshold).float()

    pseudo_labels_loss = (F.cross_entropy(predictions_transformed2, pseudo_labels, reduction='none') * mask).mean()

    consistency_loss = w_consistency * F.mse_loss(probabilities_transformed1, probabilities_transformed2)

    return adversarial_adaptive_clustering_loss, pseudo_labels_loss, consistency_loss



def adversarial_adaptive_clustering_loss_unlabeled(args, features, target, probabilities, probabilities_transformed,
                                                   binary_cross_entropy, device):
    target_unlabeled = pairwise_target(args, features, target, device)
    probability_bottleneck_row, _ = pairwise_enumerate_2d(probabilities)
    _, probability_bottleneck_column = pairwise_enumerate_2d(probabilities_transformed)
    adversarial_binary_cross_entropy_loss = -binary_cross_entropy(probability_bottleneck_row,
                                                                  probability_bottleneck_column, target_unlabeled)
    return adversarial_binary_cross_entropy_loss


def pairwise_target(args, features, target, device):
    """ Generates similarity label pairwise """
    detached_features = features.detach()
    topK = 5
    # when data is unlabeled
    if target is None:
        rank_features = detached_features
        rank_index = torch.argsort(rank_features, dim=1, descending=True)
        rank_index1, rank_index2 = pairwise_enumerate_2d(rank_index)
        rank_index1, rank_index2 = rank_index1[:, :topK], rank_index2[:, :topK]
        rank_index1, _ = torch.sort(rank_index1, dim=1)
        rank_index2, _ = torch.sort(rank_index2, dim=1)
        rank_difference = rank_index1 - rank_index2
        rank_difference = torch.sum(torch.abs(rank_difference), dim=1)
        target_unlabeled = torch.ones_like(rank_difference).float().to(device)
        target_unlabeled[rank_difference > 0] = 0
    elif target is not None:
        target_row, target_column = pairwise_enumerate_1D(target)
        target_unlabeled = torch.zeros(target.size(0) * target.size(0)).float().to(device)
        target_unlabeled[target_row == target_column] = 1
    else:
        raise ValueError('Verify the target labels')
    return target_unlabeled


def pairwise_enumerate_1D(x):
    assert x.ndimension() == 1, 'Dimension of the input must be 1'
    x1 = x.repeat(x.size(0), )
    x2 = x.repeat(x.size(0)).view(-1, x.size(0)).transpose(1, 0).reshape(-1)
    return x1, x2


def pairwise_enumerate_2d(x):
    assert x.ndimension() == 2, 'Dimension of the input must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2


class BCE_soft_labels(nn.Module):
    eps = 1e-1
    def forward(self, probability1, probability2, similarity):
        P = probability1.mul_(probability2)
        P = P.sum(1)
        negativeLogP = -(similarity * torch.log(P + self.eps) + (1. - similarity) * torch.log(1. - P + self.eps))
        return negativeLogP.mean()

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
