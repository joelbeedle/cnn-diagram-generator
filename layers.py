"""
This module defines the Layer class and its subclasses, which are used to represent the different types of layers that
can be used in a convolutional neural network (CNN) model.

Classes:
    Layer: An abstract base class representing a layer in a CNN model. 
    ConvolutionLayer: A class representing a convolutional layer in a CNN model. 
    BatchNormalization: A class representing a batch normalization layer in a CNN model.
"""
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    An abstract base class representing a layer in a convolutional neural network (CNN) model. 

    Attributes:
        name (str): The name of the layer.

    Properties:
        input_channels (int): The number of input channels for the layer. 
        output_channels (int): The number of output channels for the layer.
        height (int): The height of the layer.
        width (int): The width of the layer.

    Methods:
        compute_shape: Abstract method that computes the output shape of the layer.

    Overrides:
        __eq__(self, __value: object) -> bool: Overrides the default __eq__ method to check if two layers are of the same
        class.
        __hash__(self) -> int: Overrides the default __hash__ method to allow for hashing of layers.

    Subclasses should implement the abstract method compute_shape, which should compute the output shape of the layer.

    Subclasses:
        ConvolutionLayer: A class representing a convolutional layer in a CNN model. 
        BatchNormalization: A class representing a batch normalization layer in a CNN model.
    """
    def __init__(self, input_channels, output_channels, height, width, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._height = height
        self._width = width
        self.name = str(self)

    @abstractmethod
    def compute_shape(self):
        pass

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def __eq__(self, __value: object) -> bool:
        return self.__class__ == __value

    def __hash__(self) -> int:
        return hash(repr(self))

class ConvolutionLayer(Layer):
    """A convolutional layer in a convolutional neural network.

    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        height (int): The height of the layer's input.
        width (int): The width of the layer's input.
        kernel_size (int): The size of the convolution kernel.
        dilation (int): The dilation rate of the convolution.
        stride (int): The stride of the convolution.
        padding (int): The padding of the convolution.

    Attributes:
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding of the convolution.
        dilation (int): The dilation rate of the convolution.

    Methods:
        compute_shape: Computes the shape of the layer's output.

    """
    def __init__(self, input_channels, output_channels, height, width, kernel_size, dilation=1, stride=1, padding=0, *args, **kwargs) -> None:
        super().__init__(input_channels, output_channels, height, width, *args, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def compute_shape(self):
        from math import floor
        new_dims = floor((self.height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)/self.stride + 1)
        return new_dims


class BatchNormalization(Layer):
    """A batch normalization layer in a convolutional neural network.

    Args:
        num_features (int): The number of features.
        height (int): The height of the layer's input.
        width (int): The width of the layer's input.
        momentum (float): The momentum for the moving averages.

    Methods:
        compute_shape: Computes the shape of the layer's output.

    """
    def __init__(self, num_features, height, width, momentum=0.1, *args, **kwargs):
        super().__init__(num_features, num_features, height, width, *args, **kwargs)
        self.momentum = momentum

    def compute_shape(self):
        return super().compute_shape()
