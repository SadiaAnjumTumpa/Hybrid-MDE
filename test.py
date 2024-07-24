# Adapted from: https://github.com/uzh-rpg/rpg_ramnet/tree/master/RAM_Net

import os
import json
import logging
import argparse
import torch
import statistics
from models_for_test_snn import build_model # for event based ann model
from metrics import *
from data_loader.dataset import *
from torch.utils.data import DataLoader
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from dataloader import concatenate_subfolders
import matplotlib.pyplot as plt
import numpy as np
import time
from misc import *

logging.basicConfig(level=logging.INFO, format='')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval_metrics(output, target):
    metrics = [abs_rel_diff, scale_invariant_error, median_error, mean_error, rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics


def make_colormap(img, color_mapper):
    color_map_inv = np.ones_like(img[0]) * np.amax(img[0]) - img[0]
    color_map_inv = np.nan_to_num(color_map_inv, nan=1)
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    color_map_inv = color_mapper.to_rgba(color_map_inv)
    color_map_inv[:, :, 0:3] = color_map_inv[:, :, 0:3][..., ::-1]
    return color_map_inv

def main(args):
    train_logger = None
    calculate_scale = True
    L=1
    total_metrics = []
    if args.output_folder:
        ensure_dir(args.output_folder)
        depth_dir = join(args.output_folder, "depth")
        npy_dir = join(args.output_folder, "npy")
        color_map_dir = join(args.output_folder, "color_map")
        gt_dir_grey = join(args.output_folder, "ground_truth/grey")
        gt_dir_color_map = join(args.output_folder, "ground_truth/color_map")
        gt_dir_npy = join(args.output_folder, "ground_truth/npy")
        semantic_seg_dir_npy = join(args.output_folder, "semantic_seg/npy")
        semantic_seg_dir_frames = join(args.output_folder, "semantic_seg/frames")
        video_pred = join(args.output_folder, "video/predictions")
        video_gt = join(args.output_folder, "video/gt")
        video_inputs = join(args.output_folder, "video/inputs")
        masks = join(args.output_folder,"masks")
        ensure_dir(depth_dir)
        ensure_dir(npy_dir)
        ensure_dir(color_map_dir)
        ensure_dir(gt_dir_grey)
        ensure_dir(gt_dir_color_map)
        ensure_dir(gt_dir_npy)
        ensure_dir(semantic_seg_dir_npy)
        ensure_dir(semantic_seg_dir_frames)
        ensure_dir(video_pred)
        ensure_dir(video_gt)
        ensure_dir(video_inputs)
        ensure_dir(masks)
        print('Will write images to: {}'.format(depth_dir))

    event_path = "events/voxels" 
    rgb_path = "rgb/davis_left_sync"
    gt_path = "depth/data"
    test_dataset = concatenate_subfolders(join(args.data_path, args.data_folder),
                                           "SequenceSynchronizedFramesEventsDataset",
                                           event_path,
                                           gt_path,
                                           rgb_path,
                                           sequence_length=1,
                                           transform=CenterCrop(224),
                                           proba_pause_when_running=0.0,
                                           proba_pause_when_paused=0.0,
                                           step_size=args.step_size,
                                           clip_distance=args.clip_distance,
                                           every_x_rgb_frame=1,
                                           normalize='True',
                                           scale_factor=1,
                                           use_phased_arch="False",
                                           baseline="False",
                                           loss_composition = ['image','event0'],
                                           reg_factor=args.reg_factor,
                                           dataset_idx_flag=True,
                                           recurrency = "False")

    model = build_model(args)
    state = model.state_dict()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # Weights are  being loaded
    if args.initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(args.path_to_model)) 
        checkpoint = torch.load(args.path_to_model)
        model.load_state_dict(checkpoint['state_dict'])

    
    
    model.eval()
    model_size = sum(p.numel() for p in model.parameters()) / (1024**2)

    video_idx = 0
    N =len(test_dataset)
    if calculate_scale:
        scale = np.empty(N)

    # construct color mapper, such that same color map is used for all outputs.
    # get groundtruth that is not at the beginning
    item, dataset_idx= test_dataset[4]
    
    frame = item[0]['depth_image'].cpu().numpy()
    color_map_inv = np.ones_like(frame[0]) * np.amax(frame[0]) - frame[0]
    #color_map_inv = np.ones_like(frame) * np.amax(frame) - frame
    color_map_inv = np.nan_to_num(color_map_inv, nan=1)
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    
    vmax = np.percentile(color_map_inv, 95)
    normalizer = mpl.colors.Normalize(vmin=color_map_inv.min(), vmax=vmax)
    color_mapper_overall = cm.ScalarMappable(norm=normalizer, cmap='magma')
    color_map_inv = color_mapper_overall.to_rgba(color_map_inv)
    color_map = make_colormap(frame, color_mapper_overall)

    with torch.no_grad():
        idx = 0
        prev_dataset_idx = -1
        while idx < N:
            item, dataset_idx = test_dataset[idx]
            if dataset_idx > prev_dataset_idx:
                sequence_idx = 0
            
            input = {}
            for key, value in item[0].items():
                input[key] = value[None, :].to(device)
            new_predicted_targets, snn_output = model(input['image'], input['events'])
            if args.output_folder and sequence_idx > 1:
                groundtruth = input['depth_image']
                metrics = eval_metrics(new_predicted_targets, groundtruth)
                total_metrics.append(metrics)
                img = new_predicted_targets[0].cpu().numpy()


                # save SNN output numpy
                npy_dir_key_snn = join(npy_dir, 'snn_output')
                ensure_dir(npy_dir_key_snn)
                snn_data = snn_output[0].cpu().numpy()
                np.save(join(npy_dir_key_snn, 'snn_output_{:010d}.npy'.format(idx)), snn_data)

                # save depth image
                depth_dir_key = join(depth_dir,'depth') 
                ensure_dir(depth_dir_key)
                cv2.imwrite(join(depth_dir_key, 'frame_{:010d}.png'.format(idx)),img[0][:, :, None] * 255.0)
                # save numpy
                npy_dir_key = join(npy_dir, 'depth')
                ensure_dir(npy_dir_key)
                data = img
                np.save(join(npy_dir_key, 'depth_{:010d}.npy'.format(idx)), data)
                #save color map
                color_map_dir_key = join(color_map_dir, 'depth')
                ensure_dir(color_map_dir_key)
                color_map = make_colormap(img, color_mapper_overall)
                cv2.imwrite(join(color_map_dir_key, 'frame_{:010d}.png'.format(idx)), color_map * 255.0)
                for key, value in input.items():

                    if 'depth' in key:
                        # save events #SAT
                        event = input['events']
                        event_dir_key = join(args.output_folder, "events")
                        ensure_dir(event_dir_key)
                        input_data = event[0].cpu().numpy()
                        input_data = np.sum(input_data, axis=0)
                        negativ_input = np.where(input_data <= -0.5, 1.0, 0.0)
                        positiv_input = np.where(input_data > 0.9, 1.0, 0.0)
                        zeros_input = np.zeros_like(input_data)
                        total_image = np.concatenate((negativ_input[:, :, None], zeros_input[:, :, None], positiv_input[:, :, None]), axis=2)
                        cv2.imwrite(join(event_dir_key,'frame_{:010d}.png'.format(idx)),total_image * 255.0)
                    
                        # save GT images grey
                        gt_dir_grey_key = join(gt_dir_grey,'gt')
                        ensure_dir(gt_dir_grey_key)
                        img = value[0].cpu().numpy()
                        cv2.imwrite(join(gt_dir_grey_key, 'frame_{:010d}.png'.format(idx)), img[0][:, :, None] * 255.0)

                        # save GT images color map
                        gt_dir_cm_key = join(gt_dir_color_map, 'gt')
                        ensure_dir(gt_dir_cm_key)
                        color_map = make_colormap(img, color_mapper_overall)
                        cv2.imwrite(join(gt_dir_cm_key, 'frame_{:010d}.png'.format(idx)), color_map * 255.0)

                        # save GT to numpy array
                        gt_dir_npy_key = join(gt_dir_npy, 'gt')
                        ensure_dir(gt_dir_npy_key)
                        np.save(join(gt_dir_npy_key, 'frame_{:010d}.npy'.format(idx)), img)
                    

                if idx % 100 == 0:
                    print("saved image ", idx)
            
            sequence_idx += 1
            prev_dataset_idx = dataset_idx
            idx += 1
            
if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Hybrid MDE Testing')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='')
    parser.add_argument('--data_path', default = "/MVSEC_dataset/", type=str, help="data folder path")
    parser.add_argument('--output_folder', type=str,
                        help='path to folder for saving outputs',
                        default='')
    parser.add_argument('--data_folder', type=str,
                        help='path to folder of data to be tested',
                        default='')
    parser.add_argument('--clip_distance', default=80, type=int) # for mvsec
    parser.add_argument('--reg_factor', default=3.70378, type=float) # for mvsec
    parser.add_argument('--step_size', default=1, type=int) # for mvsec
    parser.add_argument('--num_enc_dec_layers', default=12, type=int,
                        help="Number of encoding and decoding layers in the transformer (depth)")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=12, type=int,
                        help ="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_res_blocks', default=1, type=int,
                        help="Number of residual blocks in RRDB")
    parser.add_argument('--initial_checkpoint', default=1, type=int)
    
    # SNN parameters
    parser.add_argument('--spike_threshold', default=1.0, type=float,
                        help="Spike threshold value")
    parser.add_argument('--leak_membrane', default=1.0, type=float,
                        help="Leak membrane value")
    parser.add_argument('--model_name', default="mm_hybrid_256", type=str, help ='Choose from ["mm_hybrid_224", "mm_hybrid_256"]')
                        
    args = parser.parse_args()

    main(args)
