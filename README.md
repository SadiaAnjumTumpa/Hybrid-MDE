# Hybrid-MDE
This repository is the Pytorch implementation of our ISVLSI'24 paper titled SNN-ANN Hybrid Networks for Embedded Multimodal Monocular Depth Estimation.

# Abstract
Monocular depth estimation is a crucial task in many embedded vision systems with numerous applications in autonomous driving, robotics and augmented reality. Traditional methods often rely only on frame-based approaches, which struggle in dynamic scenes due to their limitations, while event-based cameras offer complementary high temporal resolution, though they lack spatial resolution and context. We propose a novel embedded multimodal monocular depth estimation framework using a hybrid spiking neural network (SNN) and artificial neural network (ANN) architecture. This framework leverages a custom accelerator, TransPIM for efficient transformer deployment, enabling real-time depth estimation on embedded systems. Our approach leverages the advantages of both frame-based and event-based cameras, where SNN extracts low-level features and generates sparse representations from events, which are then fed into an ANN with frame-based input for estimating depth. The SNN-ANN hybrid architecture allows for efficient processing of both RGB and event data showing competitive performance across different accuracy metrics in depth estimation with standard benchmark MVSEC and DENSE dataset. To make it accessible to embedded system we deploy it on TransPIM enabling 9× speedup and 183× lower energy consumption compared to standard GPUs opening up new possibilities for various embedded system applications.

# SNN-ANN Hybrid network architecture
![mm_model_D525_box](https://github.com/user-attachments/assets/c3fadaca-5b7c-4938-843a-d8739ec9b755)


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
## Acknowledgement
This work is supported in part by National Science Foundation (NSF) grant no. 1955815 and DoD grant no. N6833522C0487.
