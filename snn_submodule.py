import torch
import torch.nn as nn
import math
   
def conv_s(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        )

class SpikingNN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1e-5).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 1e-5] = 0
        return grad_input
        
        
def IF_Neuron(membrane_potential, threshold):
    global threshold_k
    threshold_k = threshold
    
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN.apply(ex_membrane)
    out = out.detach() + (1/threshold)*out - (1/threshold)*out.detach()

    return membrane_potential, out  
    
    
class SNNModel(nn.Module):
    
    def __init__(self,batchNorm=True, ):
        super(SNNModel,self).__init__()
        
        self.batchNorm =  batchNorm #(False for 2_noBN...)
        
        self.conv1   = conv_s(self.batchNorm,   1,  64, kernel_size=3, stride=1) 
        self.conv2   = conv_s(self.batchNorm,  64,  128, kernel_size=3, stride=1)
        self.conv3   = conv_s(self.batchNorm, 128,  256, kernel_size=3, stride=1)
        self.conv4   = conv_s(self.batchNorm, 256,  1, kernel_size=3, stride=1)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)



    def forward(self, input, image_resize, sp_threshold, leak_mem):
        threshold = sp_threshold
      
        mem_1 = torch.zeros(input.size(0), 64, int(image_resize), int(image_resize)).cuda()
        mem_2 = torch.zeros(input.size(0), 128, int(image_resize), int(image_resize)).cuda()
        mem_3 = torch.zeros(input.size(0), 256, int(image_resize), int(image_resize)).cuda()
        mem_4 = torch.zeros(input.size(0), 1, int(image_resize), int(image_resize)).cuda()

        mem_1_total = torch.zeros(input.size(0), 64, int(image_resize), int(image_resize)).cuda()
        mem_2_total = torch.zeros(input.size(0), 128, int(image_resize), int(image_resize)).cuda()
        mem_3_total = torch.zeros(input.size(0),256, int(image_resize), int(image_resize)).cuda()
        mem_4_total = torch.zeros(input.size(0), 1, int(image_resize), int(image_resize)).cuda()

        # 4D input voxel
        input4D = input
        
        for i in range(input4D.size(1)):

            input11 = input4D[:, i, :, :].unsqueeze(1) # Using unsqueeze rather than explicitly changing the dimension part
  
            current_1 = self.conv1(input11)
            mem_1 = leak_mem*mem_1 + current_1
            mem_1, out_conv1 = IF_Neuron(mem_1, threshold)
  
            current_2 = self.conv2(out_conv1)
            mem_2 = leak_mem*mem_2 + current_2
            mem_2, out_conv2 = IF_Neuron(mem_2, threshold)
        
            current_3 = self.conv3(out_conv2)
            mem_3 = leak_mem*mem_3 + current_3
            mem_3, out_conv3 = IF_Neuron(mem_3, threshold)
         
            current_4 = self.conv4(out_conv3)
            mem_4 = leak_mem*mem_4 + current_4
            mem_4_total = mem_4_total + current_4
            mem_4, out_conv4 = IF_Neuron(mem_4, threshold)

        return mem_4_total

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

