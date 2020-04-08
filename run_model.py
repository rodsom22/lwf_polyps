#!/usr/bin/env python


"""
This code is based on Fizyr's keras RetinaNet with the following licence:

Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import numpy as np

import keras
import tensorflow as tf

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
import retinanet.models as models
from retinanet.preprocessing.csv_generator import CSVGenerator
from retinanet.preprocessing.pascal_voc import PascalVocGenerator
from retinanet.utils.config import read_config_file, parse_anchor_parameters
from retinanet.utils.keras_version import check_keras_version

from retinanet.utils.eval import evaluate
from retinanet.utils.eval_polyp import evaluate_polyp
from retinanet.utils.polyp_utils import recall, precision, f1, f2


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from retinanet.preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'ead' or args.dataset_type == 'polyp':
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    parser.add_argument('--model', help='Path to RetinaNet model.')
    parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for evaluation.')
    parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('--data-dir', help='Main data directory', dest='data_dir')
    parser.add_argument('--val-dir', help='Val data directory', dest='val_dir', default='val')
    parser.add_argument('--convert-model',
                        help='Convert the model to an inference model (ie. the input is a training model).',
                        action='store_true')
    parser.add_argument('--backbone', help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',
                        default=0.05, type=float)
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).',
                        default=0.5, type=float)
    parser.add_argument('--max-detections', help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path', help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1333)
    parser.add_argument('--config',
                        help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--dataset-type', help='for validation purposes', default='polyp')
    parser.add_argument('--im-threshold', help='threshold for drawing.', default=0.5, type=float)
    parser.add_argument('--mode', help='solely detection or scoring', default='scoring')
    parser.add_argument('--train-type', help='whether we doing multitask or not', default="single", type=str)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)
    print('Done')

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)

    # start evaluation
    if args.dataset_type == 'coco':
        from retinanet.utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)

    elif args.dataset_type == 'ead':

        false_positives, true_positives, iou, average_precisions, image_names, detection_list, scores_list, \
            labels_list, FP_list = evaluate(
                generator,
                model,
                iou_threshold=args.iou_threshold,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                save_path=args.save_path,
                im_threshold=args.im_threshold)

        #  calculate number of FP and TP:
        FP_sum = 0
        TP_sum = 0
        GT_sum = 0
        for i in range(len(false_positives)):
            FP_sum += int(false_positives[i][0])
            TP_sum += int(true_positives[i][0])
            GT_sum += int(true_positives[i][1])
        precision_self = TP_sum / GT_sum
        recall_self = TP_sum / (FP_sum + TP_sum)

        print("AP's: ", average_precisions)
        print("FP: ", false_positives)
        print("Total False Positives: ", FP_sum)
        print("TP: ", true_positives)
        print("Total True Positives: ", TP_sum)
        print("Overall precision: {:.2f}".format(precision_self))
        print("Overall recall: {:.2f}".format(recall_self))

        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        mIoU = np.mean(iou)
        print("len(iou): ", len(iou))
        print_mAP = sum(precisions) / sum(x > 0 for x in total_instances)

        print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
            sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        print('mAP: {:.4f}'.format(print_mAP))
        print('mIoU: {:.4f}'.format(mIoU))
        print('EAD Score (old): {:.4f}'.format(0.8 * print_mAP + 0.2 * mIoU))
        print('EAD Score (new): {:.4f}'.format(0.6 * print_mAP + 0.4 * mIoU))

        return precisions, total_instances, iou, image_names, detection_list, scores_list, labels_list, FP_list

    elif args.dataset_type == 'polyp' and args.mode == 'scoring':

        TP, FP, TN, FN, p_count = evaluate_polyp(
            generator,
            model,
            args.data_dir,
            args.val_dir,
            args.val_annotations,
            args.mode,
            args.classes,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path,
            im_threshold=args.im_threshold
        )

        prec = precision(TP, FP)
        rec = recall(TP, FN)
        f1_val = f1(prec, rec)
        f2_val = f2(prec, rec)

        print("\nTP: %d\nFP: %d\nFN: %d\nNumber of Polyps: %d\n" % (TP, FP, FN, p_count))
        print('precision: {:.4f}'.format(prec))
        print('recall: {:.4f}'.format(rec))
        print('f1: {:.4f}'.format(f1_val))
        print('f2: {:.4f}'.format(f2_val))

        return TP, FP, TN, FN, p_count

    elif args.dataset_type == 'polyp' and args.mode == 'detection':

        detections_df = evaluate_polyp(
            generator,
            model,
            args.data_dir,
            args.val_dir,
            args.val_annotations,
            args.mode,
            args.classes,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path,
            im_threshold=args.im_threshold,
            save_individual=False
        )

        return detections_df