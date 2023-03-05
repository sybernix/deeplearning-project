def get_losses_unlabeled(feature_extractor, unlabeled_data_images):
    features = feature_extractor(unlabeled_data_images)
