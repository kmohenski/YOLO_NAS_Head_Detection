from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
# from super_gradients.training.utils.distributed_training_utils import setup_device
from tqdm.auto import tqdm
from super_gradients.training.transforms.transforms import (
    DetectionRandomAffine,
    DetectionHSV,
    DetectionHorizontalFlip,
    DetectionPaddedRescale,
    DetectionStandardize,
    DetectionTargetsFormatTransform,
)

import os
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import random

# GLOBAL VARIABLES
## Training variables
EPOCHS = 30
BATCH_SIZE = 2
WORKERS = 4

# INPUT_DIM = [45*32, 26*32] # 3/4 size
INPUT_DIM = (60*32, 34*32) # full size

## Dataset variables
# ROOT_DIR = 'YN-HT'
ROOT_DIR = 'YN-HT-s'
CHECKPOINT_DIR = 'checkpoints'
train_images_dir = 'images/train'
train_labels_dir = 'labels/train'
val_images_dir = 'images/val'
val_labels_dir = 'labels/val'
test_images_dir = 'images/test'
test_labels_dir = 'labels/test'
classes = ['Head']
# classes = ['Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare']

models_to_train = [
    'yolo_nas_s'
    # 'yolo_nas_m'
    # 'yolo_nas_l'
]


def model_training():
    dataset_params = {
        'data_dir': ROOT_DIR,
        'train_images_dir': train_images_dir,
        'train_labels_dir': train_labels_dir,
        'val_images_dir': val_images_dir,
        'val_labels_dir': val_labels_dir,
        'test_images_dir': test_images_dir,
        'test_labels_dir': test_labels_dir,
        'classes': classes
    }

    # DATA
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes'],
            'input_dim': INPUT_DIM
        },
        dataloader_params={
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            'batch_size': BATCH_SIZE,
            'num_workers': WORKERS,
            "persistent_workers": True
        }
    )

    train_data.dataset.transforms.pop(2)

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes'],
            'input_dim': INPUT_DIM
        },
        dataloader_params={
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            'batch_size': BATCH_SIZE,
            'num_workers': WORKERS,
            "persistent_workers": True
        }
    )

    # TRAINING
    train_params = {
        'silent_mode': False,
        "average_best_models": True,
        "warmup_mode": "LinearEpochLRWarmup",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            ),
            DetectionMetrics_050_095(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50:0.95'
    }

    for model_to_train in models_to_train:
        trainer = Trainer(
            experiment_name=model_to_train,
            ckpt_root_dir=CHECKPOINT_DIR
        )
        model = models.get(
            model_to_train,
            num_classes=len(dataset_params['classes']),
            pretrained_weights='coco'
        )

        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data
        )


# VISUAL
## colors for bounding boxes
colors = np.random.uniform(0, 255, (len(classes), 3))


# Convert from yolo (center point, W, H) to bounding box (xmin, ymin, xmax, ymax)
# yolo is normalized so we will need to denormalize when plotting
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax


# Draws box and and labels on detected objects
def plot_box(image, bboxes, labels):
    # get image height and width
    height, width, _ = image.shape
    # draw line width
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    # font
    tf = max(lw - 1, 1)
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize
        xmin = int(x1 * width)
        ymin = int(y1 * height)
        xmax = int(x2 * width)
        ymax = int(y2 * height)

        # point 1 and 2 for cv2
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

        class_name = classes[int(labels[box_num])]
        color = colors[classes.index(class_name)]

        # draw rectangle
        cv2.rectangle(
            image,
            p1, p2,
            color=color,
            thickness=lw,
            lineType=cv2.LINE_AA
        )

        # draw text box
        w, h = cv2.getTextSize(
            class_name,
            0,
            fontScale=lw / 3,
            thickness=tf
        )[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(
            image,
            p1, p2,
            color=color,
            thickness=lw,
            lineType=cv2.LINE_AA
        )

        # write text
        cv2.putText(
            image,
            class_name,
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3.5,
            color=(255, 255, 255),
            thickness=lw,
            lineType=cv2.LINE_AA
        )

    return image


# Plot all images with bounding boxes
def plot(image_path, label_path, num_samples):
    all_training_images = glob.glob(image_path + '/*')
    all_training_images.sort()
    all_training_labels = glob.glob(label_path + '/*')
    all_training_labels.sort()

    temp = list(zip(all_training_images, all_training_labels))
    random.shuffle(temp)
    all_training_images, all_training_labels = zip(*temp)
    all_training_images, all_training_labels = list(all_training_images), list(all_training_labels)

    num_images = len(all_training_images)

    if num_samples == -1:
        num_samples = num_images

    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        image_name = all_training_images[i].split(os.path.sep)[-1]
        image = cv2.imread(all_training_images[i])
        with open(all_training_labels[i], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label, x_c, y_c, w, h = label_line.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)

        # Visualize 2x2 grid
        plt.subplot(2, 2, i + 1)
        plt.imshow(result_image[:, :, :])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

