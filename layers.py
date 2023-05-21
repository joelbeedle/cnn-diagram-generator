"""
This module defines the Layer class and its subclasses, which are used to represent the
different types of layers that can be used in a convolutional neural network (CNN) model

Classes:
    Layer: An abstract base class representing a layer in a CNN model.
    ConvolutionLayer: A class representing a convolutional layer in a CNN model.
    BatchNormalization: A class representing a batch normalization layer in a CNN model.
"""
import tensorflow as tf


class Layer:
    def __init__(
        self, previous_layer, output_shape, name=None, *args, **kwargs
    ) -> None:
        self.previous_layer = previous_layer
        self.output_shape = output_shape
        self.height = self.output_shape[0]
        self.width = self.output_shape[0]
        self.channels = output_shape[-1]


class WrapperLayer(Layer):
    def __init__(
        self, previous_layer, original_layer, output_shape=None, *args, **kwargs
    ) -> None:
        self.previous_layer = previous_layer
        self._original_layer = original_layer
        self.output_shape = self._get_output_shape()
        self.name = self._original_layer.__class__.__name__
        super().__init__(self.previous_layer, self.output_shape, *args, **kwargs)

    def _get_output_shape(self):
        if isinstance(self._original_layer, tf.keras.layers.Layer):
            output_shape = self._original_layer.output_shape
            if isinstance(output_shape, tuple):
                print(output_shape, len(output_shape))
                if len(output_shape) == 2:
                    print(self)
                    shape = (
                        self.previous_layer.height,
                        self.previous_layer.width,
                        output_shape[1],
                    )
                else:
                    shape = output_shape
            elif isinstance(output_shape, list) and len(output_shape) == 1:
                print("example ", output_shape)
                shape = output_shape[0]

            shape = shape[1:]
            return shape


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

    def __init__(
        self,
        input_channels,
        output_channels,
        height,
        width,
        kernel_size,
        dilation=1,
        stride=1,
        padding=0,
    ) -> None:
        output_shape = (height, width, output_channels)
        super().__init__(output_shape, self.__class__.__name__)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def compute_shape(self):
        from math import floor

        new_height = floor(
            (
                self.height
                + 2 * self.padding
                - self.dilation * (self.kernel_size - 1)
                - 1
            )
            / self.stride
            + 1
        )
        new_width = floor(
            (self.width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )
        return self.output_channels, new_height, new_width


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

    def __init__(self, num_features, height, width, momentum=0.1):
        super().__init__(num_features, num_features, height, width)
        self.momentum = momentum

    def compute_shape(self):
        return self.num_features, self.height, self.width


class Dense(Layer):
    """A dense layer in a convolutional neural network."""

    def __init__(self, output_dim, *args, **kwargs):
        self.output_dim = output_dim
        super().__init__((output_dim,), *args, **kwargs)
