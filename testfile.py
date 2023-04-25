import tensorflow as tf
from diagram import Diagram
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.Sequential([
    # Convolutional layers
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    # Flatten layer
    tf.keras.layers.Flatten(),
    # Dense layers
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

diagram = Diagram(model)
diagram.draw(ax)


limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
ax.set_box_aspect(np.ptp(limits, axis=1))

plt.show()