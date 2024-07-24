import torch
import torch.nn as nn
from submodule import * 
from snn_submodule import * 
import torch.nn.functional as F

#################################
### Input resolution 256x256 ####
#################################
class transformer_encoder_decoder_rgb_events_hybrid_256(nn.Module):
  
  def __init__(self, img_size=(256,256), patch_size=16, in_chans=1, d_model=768, dropout=0.1, depth=12, nhead=12, dim_feedforward= 2048, activation="relu", filters=64, out_chans=1, num_res_blocks=1, sp_th =1.0, leak_mem = 1.0, mlp_ratio=4., qkv_bias=True, norm_post=None):
    super().__init__()
    
    self.image_resize = img_size[0]
    self.spike_th = sp_th
    self.leak_mem = leak_mem
    self.patch_embed_rgb = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=d_model) # For MVSEC in_chans = 1, for DENSE in_chans = 3
    self.patch_embed_events = PatchEmbed(img_size = img_size, patch_size=patch_size, in_chans=1, embed_dim=d_model) 
    self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed_rgb.n_patches + self.patch_embed_events.n_patches,d_model))
    self.pos_drop = nn.Dropout(p=dropout)
    self.enc_blocks = nn.ModuleList(
            [
                Enc_Block(
                    dim=d_model,
                    n_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=dropout,
                    attn_p=dropout,
                )
                for _ in range(depth)
            ]
        )
   
    self.norm = nn.LayerNorm(d_model, eps=1e-6)
    self.token_fold = nn.Fold(output_size=img_size, kernel_size = patch_size, stride = patch_size)
    self.conv_skip_events = nn.Sequential(nn.Conv2d(1, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_skip_rgb = nn.Sequential(nn.Conv2d(1, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_fusion_events = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_fusion_rgb = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_fold_events = nn.Sequential(nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),nn.ReLU()) 
    self.conv_fold_rgb = nn.Sequential(nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),nn.ReLU())
    self.RRDB = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
    self.snn_events = SNNModel()
    self.conv = nn.Sequential(
    nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(filters, out_chans, kernel_size=3, stride=1, padding=1)
)
  
  def forward(self, rgb, events):
    n_samples = rgb.shape[0] 
    x_rgb = self.patch_embed_rgb(rgb)
    events = self.snn_events(events, self.image_resize, self.spike_th, self.leak_mem)
    x_events = self.patch_embed_events(events)
    x = torch.cat((x_rgb, x_events), dim=1)
    x=x+self.pos_embed
    x=self.pos_drop(x)
     
    for enc_block in self.enc_blocks:
      x=enc_block(x)
    x=self.norm(x)
    x_rgb = self.token_fold(x[:,:-256,:].transpose(1,2)) # For input size 256, need to update this two lines from 196 to 256
    x_events = self.token_fold(x[:,256:,:].transpose(1,2)) # for events only: self.token_fold(x.transpose(1,2))
    x_rgb =  self.conv_fold_rgb(x_rgb)+ self.conv_skip_rgb(rgb)
    x_events = self.conv_fold_events(x_events)+ self.conv_skip_events(events) 
    x = self.conv_fusion_events(x_events)+self.conv_fusion_rgb(x_rgb) 
    x = self.RRDB(x)
    x = self.conv(x)

    return x


#################################
### Input resolution 224x224 ####
#################################
class transformer_encoder_decoder_rgb_events_hybrid_224(nn.Module):
  
  def __init__(self, img_size=(224,224), patch_size=16, in_chans=1, d_model=768, dropout=0.1, depth=12, nhead=12, dim_feedforward= 2048, activation="relu", filters=64, out_chans=1, num_res_blocks=1, sp_th =1.0, leak_mem = 1.0, mlp_ratio=4., qkv_bias=True, norm_post=None):
    super().__init__()
    
    self.image_resize = img_size[0]
    self.spike_th = sp_th
    self.leak_mem = leak_mem
    self.patch_embed_rgb = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=d_model) # For MVSEC in_chans = 3
    self.patch_embed_events = PatchEmbed(img_size = img_size, patch_size=patch_size, in_chans=1, embed_dim=d_model) 
    self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed_rgb.n_patches + self.patch_embed_events.n_patches,d_model))
    self.pos_drop = nn.Dropout(p=dropout)
    self.enc_blocks = nn.ModuleList(
            [
                Enc_Block(
                    dim=d_model,
                    n_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=dropout,
                    attn_p=dropout,
                )
                for _ in range(depth)
            ]
        )
   
    self.norm = nn.LayerNorm(d_model, eps=1e-6)
    self.token_fold = nn.Fold(output_size=img_size, kernel_size = patch_size, stride = patch_size)
    self.conv_skip_events = nn.Sequential(nn.Conv2d(1, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_skip_rgb = nn.Sequential(nn.Conv2d(1, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_fusion_events = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_fusion_rgb = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),nn.LeakyReLU())
    self.conv_fold_events = nn.Sequential(nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),nn.ReLU()) 
    self.conv_fold_rgb = nn.Sequential(nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),nn.ReLU())
    self.RRDB = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
    self.snn_events = SNNModel()
    self.conv = nn.Sequential(
    nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(filters, out_chans, kernel_size=3, stride=1, padding=1)
)
  
  def forward(self, rgb, events):
    n_samples = rgb.shape[0] 
    x_rgb = self.patch_embed_rgb(rgb)
    events = self.snn_events(events, self.image_resize, self.spike_th, self.leak_mem)
    x_events = self.patch_embed_events(events)
    x = torch.cat((x_rgb, x_events), dim=1)
    x=x+self.pos_embed
    x=self.pos_drop(x)
     
    for enc_block in self.enc_blocks:
      x=enc_block(x)
    x=self.norm(x)
    x_rgb = self.token_fold(x[:,:-196,:].transpose(1,2))
    x_events = self.token_fold(x[:,196:,:].transpose(1,2)) # for events only: self.token_fold(x.transpose(1,2))
    x_rgb =  self.conv_fold_rgb(x_rgb)+ self.conv_skip_rgb(rgb)
    x_events = self.conv_fold_events(x_events)+ self.conv_skip_events(events) 
    x = self.conv_fusion_events(x_events)+self.conv_fusion_rgb(x_rgb) 
    x = self.RRDB(x)
    x = self.conv(x)

    return x


def build_model(args):
    if args.model_name =='mm_hybrid_224':
        print('----- Using 224x224 rgb+events SNN based hybrid model ------')
        return transformer_encoder_decoder_rgb_events_hybrid_224(
        d_model=args.hidden_dim,
              dropout=args.dropout,
              nhead=args.nheads,
              dim_feedforward=args.dim_feedforward,
              depth=args.num_enc_dec_layers,
              num_res_blocks=args.num_res_blocks,
              sp_th = args.spike_threshold,
              leak_mem = args.leak_membrane)
    
    elif args.model_name =='mm_hybrid_256':
        print('----- Using 256x256 rgb+events SNN based hybrid model ------')
        return transformer_encoder_decoder_rgb_events_hybrid_256(
        d_model=args.hidden_dim,
              dropout=args.dropout,
              nhead=args.nheads,
              dim_feedforward=args.dim_feedforward,
              depth=args.num_enc_dec_layers,
              num_res_blocks=args.num_res_blocks,
              sp_th = args.spike_threshold,
              leak_mem = args.leak_membrane)
    
    else:
       print('Not a valid model. Exiting...')
       exit(0)
