import os
import numpy as np


def associate_data(image_dir, mask_dir):
    name_association = {}
    print("Associating image files with mask files...")
    for file in os.listdir(image_dir):
        if file.endswith('.png'):
            # Recover image name as key
            image_name = os.path.splitext(file)[0]
            split_name = image_name.split('_')
            mask_filename = f'{split_name[0]}_mask_{split_name[2]}.png'
            mask_path = os.path.join(mask_dir, mask_filename)
            name_association[image_name] = mask_path
    return name_association


def split_train_val(dataset_dicts, train_ratio, dataset_pct=1.0, seed=2013):
    np.random.seed(seed)

    assert 0.0 < dataset_pct <= 1.0, "Invalid 'dataset_pct': 0 < dataset_pct <= 1"

    num_images = len(dataset_dicts)
    num_to_use = int(num_images * dataset_pct)

    indices = np.random.permutation(num_images)[:num_to_use]
    train_indices = set(indices[:int(train_ratio * num_to_use)])
    val_indices = set(indices[int(train_ratio * num_to_use):])

    train_dicts = dataset_dicts[train_indices]
    val_dicts = dataset_dicts[val_indices]

    split_dicts = {'train': train_dicts, 'val': val_dicts}

    return train_dicts, val_dicts


def build_dataset_dicts(im_dir, mask_dir, train_ratio, dataset_pct=1.0,
                        seed=2013, size=512):
    name_association = associate_data(im_dir, mask_dir)

    dataset_dicts = []
    for idx, im_name in enumerate(name_association.keys()):
        info = {}
        im_path = os.path.join(im_dir, im_name)

        info['file_name'] = im_path
        info['image_id'] = idx
        info['height'] = size
        info['width'] = size
        info['sem_seg_file_name'] = name_association[im_name]

        dataset_dicts.append(info)

    split_dicts = split_train_val(dataset_dicts, train_ratio, dataset_pct, seed)
    return split_dicts

