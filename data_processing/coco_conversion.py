# import json
# import os
# import numpy as np
# import pycocotools.mask as mask_util
#
#
# def associate_data(image_dir, mask_dir):
#     name_association = {}
#     for file in os.listdir(image_dir):
#         if file.endswith('.png'):
#             # Recover image name as key
#             image_name = os.path.splitext(file)[0]
#             split_name = image_name.split('_')
#             mask_filename = f'{split_name[0]}_mask_{split_name[2]}.npy'
#             mask_path = os.path.join(mask_dir, mask_filename)
#             name_association[image_name] = mask_path
#     return name_association
#
#
# def to_coco(name_association, class_names, size=512):
#
#     categories = [{'id': i, 'name': name, 'supercategory': 'none'} for i, name in enumerate(class_names, 0)]
#
#     coco_images = []
#     coco_annotations = []
#     ann_id = 0
#
#     for file_id, (image_name, mask_path) in enumerate(name_association.items(), 0):
#
#         coco_images.append({
#             'id': file_id,
#             'file_name': image_name,
#             'height': size,
#             'width': size
#         })
#
#         mask = np.load(mask_path)
#
#         for class_id in np.unique(mask):
#             if class_id == 0:
#                 continue  # Skip background
#             class_mask = (mask == class_id).astype(np.uint8)
#             rle = mask_util.encode(np.asfortranarray(class_mask))
#             bbox = mask_util.toBbox(rle).tolist()
#             area = int(mask_util.area(rle))
#
#             coco_annotations.append({
#                 'id': ann_id,
#                 'image_id': image_name,
#                 'category_id': int(class_id),
#                 'area': area,
#                 'bbox': bbox,
#                 'segmentation': rle,
#                 'iscrowd': 0
#             })
#
#             ann_id += 1
#
#     coco_dict = {
#         'images': coco_images,
#         'categories': categories,
#         'annotations': coco_annotations
#     }
#
#     return coco_dict
#
#
# def split_train_val(coco_dict, train_ratio, dataset_pct=1.0, seed=2013):
#     np.random.seed(seed)
#
#     assert 0.0 < dataset_pct <= 1.0, "Invalid 'dataset_pct': 0 < dataset_pct <= 1"
#
#     num_images = len(coco_dict['images'])
#     num_to_use = int(num_images*dataset_pct)
#     indices = np.random.permutation(num_images)[:num_to_use]
#
#     train_indices = set(indices[:int(train_ratio * num_to_use)])
#     val_indices = set(indices[int(train_ratio * num_to_use):])
#
#     train_images = [coco_dict['images'][i] for i in train_indices]
#     val_images = [coco_dict['images'][i] for i in val_indices]
#
#     train_image_ids = {img['id'] for img in train_images}
#     val_image_ids = {img['id'] for img in val_images}
#
#     train_annotations = [ann for ann in coco_dict['annotations'] if ann['image_id'] in train_image_ids]
#     val_annotations = [ann for ann in coco_dict['annotations'] if ann['image_id'] in val_image_ids]
#
#     train_data = {
#         'images': train_images,
#         'annotations': train_annotations,
#         'categories': coco_dict['categories']
#     }
#
#     val_data = {
#         'images': val_images,
#         'annotations': val_annotations,
#         'categories': coco_dict['categories']
#     }
#
#     print("JSON Split into training/val Complete.\n")
#     return train_data, val_data
#
#
# def json_gen(json_dir, image_dir, mask_dir, class_names, train_ratio=0.8, dataset_pct=1.0, seed=2013, size=512):
#     train_json = os.path.join(json_dir, 'train.json')
#     val_json = os.path.join(json_dir, 'val.json')
#     os.makedirs(json_dir, exist_ok=True)
#
#     name_association = associate_data(image_dir, mask_dir)
#     coco_dict = to_coco(name_association, class_names, size)
#     train_data, val_data = split_train_val(coco_dict, train_ratio, dataset_pct, seed)
#
#     with open(train_json, 'w') as f:
#         json.dump(train_data, f)
#
#     with open(val_json, 'w') as f:
#         json.dump(val_data, f)


import json
import os
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm


def associate_data(image_dir, mask_dir):
    name_association = {}
    print("Associating image files with mask files...")
    for file in tqdm(os.listdir(image_dir), desc="Processing files"):
        if file.endswith('.png'):
            # Recover image name as key
            image_name = os.path.splitext(file)[0]
            split_name = image_name.split('_')
            mask_filename = f'{split_name[0]}_mask_{split_name[2]}.npy'
            mask_path = os.path.join(mask_dir, mask_filename)
            name_association[image_name] = mask_path
    return name_association


def to_coco(name_association, class_names, size=512):
    categories = [{'id': i, 'name': name, 'supercategory': 'none'} for i, name in enumerate(class_names, 0)]

    coco_images = []
    coco_annotations = []
    ann_id = 0

    print("Converting to COCO format...")
    for file_id, (image_name, mask_path) in tqdm(enumerate(name_association.items(), 0),
                                                 desc="Processing images", total=len(name_association)):

        coco_images.append({
            'id': file_id,
            'file_name': image_name,
            'height': size,
            'width': size
        })

        mask = np.load(mask_path)

        for class_id in np.unique(mask):
            if class_id == 0:
                continue  # Skip background
            class_mask = (mask == class_id).astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(class_mask))
            bbox = mask_util.toBbox(rle).tolist()
            area = int(mask_util.area(rle))

            coco_annotations.append({
                'id': ann_id,
                'image_id': file_id,
                'category_id': int(class_id),
                'area': area,
                'bbox': bbox,
                'segmentation': rle,
                'iscrowd': 0
            })

            ann_id += 1

    coco_dict = {
        'images': coco_images,
        'categories': categories,
        'annotations': coco_annotations
    }

    return coco_dict


def split_train_val(coco_dict, train_ratio, dataset_pct=1.0, seed=2013):
    np.random.seed(seed)

    assert 0.0 < dataset_pct <= 1.0, "Invalid 'dataset_pct': 0 < dataset_pct <= 1"

    num_images = len(coco_dict['images'])
    num_to_use = int(num_images * dataset_pct)
    indices = np.random.permutation(num_images)[:num_to_use]

    train_indices = set(indices[:int(train_ratio * num_to_use)])
    val_indices = set(indices[int(train_ratio * num_to_use):])

    train_images = [coco_dict['images'][i] for i in train_indices]
    val_images = [coco_dict['images'][i] for i in val_indices]

    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}

    train_annotations = [ann for ann in coco_dict['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in coco_dict['annotations'] if ann['image_id'] in val_image_ids]

    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_dict['categories']
    }

    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_dict['categories']
    }

    print("JSON Split into training/val Complete.\n")
    return train_data, val_data


def json_gen(json_dir, image_dir, mask_dir, class_names, train_ratio=0.8, dataset_pct=1.0, seed=2013, size=512):
    print("Starting JSON generation...")
    train_json = os.path.join(json_dir, 'train.json')
    val_json = os.path.join(json_dir, 'val.json')
    os.makedirs(json_dir, exist_ok=True)

    print("Associating data...")
    name_association = associate_data(image_dir, mask_dir)
    print("Creating COCO dictionary...")
    coco_dict = to_coco(name_association, class_names, size)
    print("Splitting data into training and validation sets...")
    train_data, val_data = split_train_val(coco_dict, train_ratio, dataset_pct, seed)

    print("Saving JSON files...")
    with open(train_json, 'w') as f:
        json.dump(train_data, f)

    with open(val_json, 'w') as f:
        json.dump(val_data, f)

    print("JSON generation complete.")