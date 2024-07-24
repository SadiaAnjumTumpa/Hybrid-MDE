# Hybrid-MDE
This repository is the Pytorch implementation of our ISVLSI'24 paper titled SNN-ANN Hybrid Networks for Embedded Multimodal Monocular Depth Estimation

## Training
Download the vit-base inital checkpoints to train the model [here](https://drive.google.com/file/d/18Azic_56AHn_ysWlmSKOmVM6sjXE_0UG/view?usp=drive_link).
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --epochs 70 --batch_size 8 --num_enc_dec_layers 12 --lr 0.0003
```
## Testing
Testing is done in two steps. First, is to run test.py script, which saves the prediction outputs in a folder. 
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --path_to_model experiments/exp_1/checkpoints/model_best.pth.tar --output_folder experiments/exp_1/test/ --data_folder test 
```
Later, we run evaluation.py script takes both the groundtruth and prediction output as inputs, and calculates the metric depth on logarithmic depth maps using both clip distance and reg_factor. 
```bash
python evaluation.py --target_dataset experiments/exp_1/test/ground_truth/npy/gt/ --predictions_dataset experiments/exp_1/test/npy/depth/ --clip_distance 80 --reg_factor 3.70378
```
