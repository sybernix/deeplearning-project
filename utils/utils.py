import os

import matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use('macosx')


def get_classlist(annotation_path):
    with open(annotation_path) as f:
        class_list = []
        for class_num, img_file_name in enumerate(f.readlines()):
            class_name = img_file_name.split(' ')[0].split('/')[-2]
            if class_name not in class_list:
                class_list.append(str(class_name))
        return class_list


def display_images_from_dataloader(data_loader):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(4):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()


def lr_scheduler(param_lr, optimizer, iter, gamma=0.0001, power=0.75, init_lr=0.001):
    coeff = init_lr * (1 + gamma * iter) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = coeff * param_lr[i]
        i += 1
    return optimizer


def prepare_folder():
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    path = 'save/%s' % now

    log_file = os.path.join(path, 'logs')

    checkpoint_path = os.path.join(path, 'checkpoint')

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return path, log_file, checkpoint_path

