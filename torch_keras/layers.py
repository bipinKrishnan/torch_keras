from torch import nn

from .utils import get_conv_output, same_pad


def Input(shape):
  Input.shape = shape
  return Input.shape


class Dense(nn.Module):
  def __init__(self, outputs, activation):
    super().__init__()
    self.outputs = outputs
    self.activation = activation

  def __call__(self, inputs):
    self.inputs_size = 1
    
    if type(inputs) == tuple:
      for i in range(len(inputs)):
        self.inputs_size *= inputs[i]
      
      self.layers = nn.Sequential(
        nn.Linear(self.inputs_size, self.outputs),
        self.activation
    )

      return self.layers

    elif isinstance(inputs[-2], nn.Linear):
      self.inputs = inputs
      self.layers = list(self.inputs)
      self.layers.extend([nn.Linear(self.layers[-2].out_features, self.outputs), self.activation])

      self.layers = nn.Sequential(*self.layers)

      return self.layers

    else:
      self.inputs = inputs
      self.layers = list(self.inputs)
      self.layers.extend([nn.Linear(get_conv_output(Input.shape, self.inputs), self.outputs), self.activation])

      self.layers = nn.Sequential(*self.layers)

      return self.layers


class FlattenedLayer(nn.Module):
  def __init__(self):
    super().__init__()
    pass

  def forward(self, input):
      self.inputs = input.view(input.size(0), -1)
      return self.inputs


class Flatten():
  def __init__(self):
    pass

  def __call__(self, inputs):
    self.inputs = inputs
    self.layers = list(self.inputs)
    self.layers.extend([FlattenedLayer()])
    self.layers = nn.Sequential(*self.layers)

    return self.layers


class Conv2d(nn.Module):
  def __init__(self, filters, kernel_size, strides, padding, dilation, activation):
    super().__init__()
    self.filters = filters
    self.kernel = kernel_size
    self.strides = strides
    self.padding = padding
    self.dilation = dilation
    self.activation = activation

  def __call__(self, inputs):

    if type(inputs) == tuple:
      self.inputs_size = inputs

      if self.padding == 'same':
        self.padding = int(same_pad(self.inputs_size[-2], self.kernel, self.strides, self.dilation))
      else:
        self.padding = self.padding

      self.layers = nn.Sequential(
        nn.Conv2d(self.inputs_size[-3],
                  self.filters, 
                  self.kernel, 
                  self.strides, 
                  self.padding,
                  self.dilation),
        self.activation
    )

      return self.layers

    else:
      if self.padding == 'same':
        self.padding = int(same_pad(get_conv_output(Input.shape, inputs), self.kernel, self.strides, self.dilation))
      else:
        self.padding = self.padding

      self.inputs = inputs
      self.layers = list(self.inputs)
      self.layers.extend(
             [nn.Conv2d(self.layers[-2].out_channels, 
                    self.filters, 
                    self.kernel, 
                    self.strides, 
                    self.padding,
                    self.dilation),
             self.activation]
          )
      self.layers = nn.Sequential(*self.layers)

      return self.layers