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

def pairwise_target():
    print()

def pairwise_enumerate_1D(x):
    assert x.ndimension() == 1, 'Dimension of the input must be 1'
    x1 = x.repeat(x.size(0), )
    x2 = x.repeat(x.size(0)).view(-1, x.size(0)).transpose(1, 0).reshape(-1)
    return x1, x2
