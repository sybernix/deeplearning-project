def get_classlist(annotation_path):
    with open(annotation_path) as f:
        class_list = []
        for class_num, img_file_name in enumerate(f.readlines()):
            class_name = img_file_name.split(' ')[0].split('/')[-2]
            if class_name not in class_list:
                class_list.append(str(class_name))
        return class_list