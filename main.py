"""
This script is used for visualizing a neural network architecture in 3D using NumPy and Matplotlib. It provides classes and functions to create and manipulate cubes representing network
layers and draw them along with their connections.
"""

import numpy as np
import matplotlib.pyplot as plt
from layers import Layer
from model import Model

def main():
    """Main entrypoint for testing"""
    # Example usage
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    layer1 = Layer('Convolution', 64, 64, 5, 'r', 0.5)
    layer2 = Layer('BatchNormalization', 64, 64, 32, 'b', 0.5)
    layer3 = Layer('Convolution', 32, 32, 5, 'b', 0.5)
    model = Model([layer1, layer2, layer3])
    model.draw(ax)

    # Set the aspect ratio of the plot to 'equal'
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))

    plt.show()


if __name__ == "__main__":
    main()
