import torch
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

def get_conv_output(shape, inputs):
  bs = 1
  data = Variable(torch.rand(bs, *shape))
  output_feat = inputs(data)

  return output_feat.size(1)

def same_pad(h_in, kernal, stride, dilation):
  return (stride*(h_in-1)-h_in+(dilation*(kernal-1))+1) / 2.0