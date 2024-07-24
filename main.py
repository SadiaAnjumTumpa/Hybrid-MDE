
import argparse
import numpy as np
import torch
import torch.nn
from utils.path_utils import ensure_dir
from torch.utils.data import DataLoader
from dataloader import build_dataset
from models import build_model 
from misc import *
from metrics import *
from torch.utils.tensorboard import SummaryWriter
import os
import math
import io
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from datetime import datetime

SEED = 42
set_seed(SEED) 

def get_model_size(model):
    # to calculate the model size in MB
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.getbuffer().nbytes
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def get_arguments():
  parser = argparse.ArgumentParser('Hybrid model training arguments', add_help=False)
  parser.add_argument('--data_path', default = "/MVSEC_dataset/mvsec_dataset_day2/", type=str, help="data folder path")
  parser.add_argument('--val_data_path', default = "/MVSEC_dataset/mvsec_dataset_day2/", type=str, help="data folder path")
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--num_worker', default=4, type=int)
  parser.add_argument('--device', default='cuda', help='device to use for train and test, cpu or cuda')
  parser.add_argument('--clip_distance', default=80, type=int) # for mvsec
  parser.add_argument('--reg_factor', default=3.7, type=int) # for mvsec
  parser.add_argument('--train_step_size', default=8, type=int) # for mvsec training
  parser.add_argument('--val_step_size', default=16, type=int) # for mvsec training
  parser.add_argument('--filters', default=64, type=int)
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
  parser.add_argument('--lr', default=3e-4, type=float)
  parser.add_argument('--weight_decay', default=1e-4, type=float)
  parser.add_argument('--lr_drop', default=200, type=int) 
  parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
  parser.add_argument('--start_epoch', default=0, type=int)
  parser.add_argument('--epochs', default=70, type=int)
  parser.add_argument('--log_step', default=50, type=int) 
  parser.add_argument('--initial_checkpoint', default=0, type=int)
  parser.add_argument('--save_freq', default=10, type=int)
  parser.add_argument('--optimizer', default="Adam", type=str, help= 'Choose from ["Adam", "SGD"]')
  parser.add_argument('--model_name', default="mm_hybrid", type=str, help ='Choose from ["events_hybrid", "mm_hybrid", "events_ann", "mm_ann"]')
  parser.add_argument('--expt_dir', default='default', help='path where to save experimental logs')
  parser.add_argument('--image_size', default= 256, type=int, help = "Input frame size")
  parser.add_argument('--spike_threshold', default=1.0, type=float,
                        help="Spike threshold value")
  parser.add_argument('--leak_membrane', default=1.0, type=float,
                        help="Leak membrane value")
  parser.add_argument('--shuffle', default='False')

  return parser.parse_args()

def main(args):

  monitor_mean_error = math.inf
  device = torch.device(args.device)

  #dataloader
  train_dataset = build_dataset(set="train", transform= Compose([RandomRotationFlip(0.0, 0.5, 0.0),RandomCrop(args.image_size)]),args=args)
  val_dataset = build_dataset(set="validation", transform =CenterCrop(args.image_size), args=args) 

  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_worker, shuffle=args.shuffle)
  val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers = args.num_worker)
  train_dataloader_len=len(train_dataloader.dataset)  
  val_dataloader_len=len(val_dataloader.dataset) 

  # model
  model = build_model(args)

  if args.initial_checkpoint:
    # Loading pretrained weights from vitbase 
    p=torch.load('pretrained_weights_updated_from_vitbase.pth') # Please set path to pretrained vitbase weights
    pretrained_model_weights = loading_weights_from_eventscape(model.state_dict(), p)
    model.load_state_dict(pretrained_model_weights)
    print('pretrained vitbase loaded successfully!')
  else: 
    print('Training from scratch...!!!')

  model = torch.nn.DataParallel(model)
  model=model.to(device)
  
  # Model summary
  n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('number of params:', n_parameters)
  size_mb = get_model_size(model)
  print(f"Model size: {size_mb:.2f} MB")
  
  # optimizer
  optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
  
  #loss
  criterion = torch.nn.L1Loss().to(device)
  criterion_normal = NormalLoss()
  grad_criterion = multi_scale_grad_loss
  
  # training
  print(f"Training started at {datetime.now()}\n===========================================")
  model.train()
  # randomly choosing few indices from both train and validation dataset for intermediate output visualization after each epoch
  val_preview_indices = np.random.choice(range(len(val_dataloader.dataset)), 2, replace=False)
  preview_indices = np.random.choice(range(len(train_dataloader.dataset)), 2, replace=False)
  
  for epoch in range(args.start_epoch, args.epochs):
    start_epoch = datetime.now()
    total_train_loss=0
    total_val_loss=0
    total_train_metrics = []
    total_val_metrics = []

    for batch_idx, sequence in enumerate(train_dataloader):
      rgb,events, gt= sequence[0]['image'].to(device), sequence[0]['events'].to(device), sequence[0]['depth_image'].to(device)
      optimizer.zero_grad()
      output= model(rgb, events) 
      
      #loss
      is_nan=torch.isnan(gt)
      l_loss = criterion(output[~is_nan], gt[~is_nan])
      grad_loss =grad_criterion(output, gt)
      gen_features = imgrad_yx(output)
      real_features = imgrad_yx(gt)
      n_loss = criterion_normal(gen_features, real_features)
      loss = 0.5*l_loss+ 0.25*grad_loss+n_loss 
      loss.backward()
      optimizer.step()
      total_train_loss+=loss.item()  
      
      #metrics 
      pred, gt = output.cpu().detach().clone(), gt.cpu().detach().clone()
      train_metrics = eval_metrics(pred,gt)
      total_train_metrics.append(train_metrics)

      if batch_idx % args.log_step == 0: 
        print(f"Train Epoch:{epoch} [{(batch_idx+1)*args.batch_size}/{train_dataloader_len} ({100*(batch_idx+1)*args.batch_size/train_dataloader_len:.0f}%)] | Batch Loss: {loss.item():.3f} | Running batch loss: {total_train_loss/(batch_idx+1):.3f} | MSE: {train_metrics[0]:.3f} | Abs ME: {train_metrics[1]:.3f}")   
    
    writer.add_scalar('train loss', total_train_loss/train_dataloader_len, epoch)   
    
    # Validation
    print('Validation...\n')
    model.eval()
    
    with torch.no_grad():
      
      for batch_idx, sequence in enumerate(val_dataloader):
        rgb,events, gt= sequence[0]['image'].to(device), sequence[0]['events'].to(device), sequence[0]['depth_image'].to(device)
        output= model(rgb, events)
        val_grad_loss = grad_criterion(output, gt)
        is_nan=torch.isnan(gt)
        val_l_loss = criterion(output[~is_nan], gt[~is_nan])
        gen_features = imgrad_yx(output)
        real_features = imgrad_yx(gt)
        val_n_loss = criterion_normal(gen_features, real_features)
        val_loss = 0.5*val_l_loss + 0.25*val_grad_loss+ val_n_loss # + val_sil_loss
        total_val_loss+=val_loss.item()
        
        # metrics
        pred, gt = output.cpu().detach().clone(),  gt.cpu().detach().clone()
        val_metrics = eval_metrics(pred,gt)
        total_val_metrics.append(val_metrics)

        if batch_idx % args.log_step == 0:
          print(f"Validation Epoch:{epoch} [{(batch_idx+1)*args.batch_size}/{val_dataloader_len} ({100*(batch_idx+1)*args.batch_size/val_dataloader_len:.0f}%)] | Batch Loss: {val_loss.item():.3f} | Running batch loss: {total_val_loss/(batch_idx+1):.3f} | MSE: {val_metrics[0]:.3f} | Abs ME: {val_metrics[1]:.3f}")   
        
    avg_loss_train = total_train_loss/train_dataloader_len
    avg_loss_val = total_val_loss/val_dataloader_len
    avg_train_metrics = np.sum(np.array(total_train_metrics),0)/len(total_train_metrics)
    avg_val_metrics = np.sum(np.array(total_val_metrics),0)/len(total_val_metrics)
    
    print(f"Epoch {epoch} done! Took {datetime.now()-start_epoch} time.\n\
          Train loss: {avg_loss_train:.3f}, Val loss: {avg_loss_val:.3f} | Train mse: {avg_train_metrics[0]:.3f}, Val mse: {avg_val_metrics[0]:.3f} | Train abs me: {avg_train_metrics[1]:.3f}, Val abs me: {avg_val_metrics[1]:.3f}")

    #saving checkpoints
    states={'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(),'val_loss':avg_loss_val} 
          
    # Updated to track mean square error 
    current_mean_sqr_error = avg_val_metrics[0] 
    if(current_mean_sqr_error < monitor_mean_error):
        monitor_mean_error = current_mean_sqr_error 
        filename = os.path.join(modellog_logdir,'model_best.pth.tar')
        torch.save(states, filename)
        print('Saving current best:{} at {} with validation absolute mean square error {}\n'.format(filename, epoch, monitor_mean_error))
    
    
  
if __name__=='__main__':
  args = get_arguments()

  # tensorboard and logging directories
  checkpoint_dir = "Experiments/"+ args.expt_dir # For experiments
  print(checkpoint_dir)
  ensure_dir(checkpoint_dir)
  int_path = os.path.join(checkpoint_dir, 'intermediate_results')
  print(int_path)
  ensure_dir(int_path)
  tensorboard_logdir = os.path.join(checkpoint_dir, 'tensorboard')
  ensure_dir(tensorboard_logdir)
  modellog_logdir = os.path.join(checkpoint_dir, 'checkpoints')
  ensure_dir(modellog_logdir)
  writer = SummaryWriter(log_dir=tensorboard_logdir)
 
  main(args)
  print(f'Training completed at {datetime.now()}')
