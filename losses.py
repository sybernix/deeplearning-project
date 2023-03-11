import torch
import torch.nn.functional as F


def get_losses_unlabeled(feature_extractor, predictor, unlabeled_data_images, unlabeled_data_images_transformed1,
                         unlabeled_data_images_transformed2):
    features = feature_extractor(unlabeled_data_images)
    features_transformed1 = feature_extractor(unlabeled_data_images_transformed1)
    features_transformed2 = feature_extractor(unlabeled_data_images_transformed2)

    predictions = predictor(features)
    predictions_transformed1 = predictor(features_transformed1)

    probabilities = F.softmax(predictions, dim=1)
    probabilities_transformed1 = F.softmax(predictions_transformed1, dim=1)

    # calculate adversarial adaptive clustering loss

def adversarial_adaptive_clustering_loss_unlabeled():
    print()

def pairwise_target(args, features, target, device):
    """ Generates similarity label pairwise """
    detached_features = features.detach()
    # when data is unlabeled
    if target is None:
        rank_features = detached_features
        rank_index = torch.argsort(rank_features, dim=1, descending=True)
        rank_index1, rank_index2 = pairwise_enumerate_2d(rank_index)
        rank_index1, rank_index2 = rank_index1[:, :args.topK], rank_index2[:, :args.topK]
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
