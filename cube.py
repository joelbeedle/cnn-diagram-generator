"""
Module for 3D Cube visualization for CNN network layers.

This module provides a class to represent a 3D cube and a class to manage and draw multiple cubes together.

Classes:
Cube: Represents a 3D cube with attributes such as width, height, depth, x, y, z, color, and alpha. It also contains methods to draw and translate vertices.
CubeCollection: A collection of Cube objects used to manage and draw multiple cubes together.

"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CubeCollection:
    """
    A collection of Cube objects, used to manage and draw multiple cubes together.

    Attributes:
        cubes (list): A list of Cube objects.
    """

    def __init__(self, cubes=[]):
        self.cubes = cubes

    def add(self, cube: "Cube"):
        """
        Add a Cube object to the CubeCollection.

        Args:
            cube (Cube): The Cube object to be added to the collection.
        """
        self.cubes.append(cube)

    def draw_cubes(self, ax):
        """
        Draw all the cubes in the collection on the provided 3D axis.

        Args:
            ax (Axes3D): The 3D axis on which to draw the cubes.
        """
        for cube in self.cubes:
            cube.draw(ax)


cube_collection = CubeCollection()


class Cube:
    """
    Represents a 3D cube, used to visualize network layers.

    Attributes:
        color (str): The color of the cube.
        alpha (float): The transparency of the cube.
        width (float): The width of the cube.
        height (float): The height of the cube.
        depth (float): The depth of the cube.
        x (float): The x-coordinate of the bottom left vertex of the cube.
        y (float): The y-coordinate of the bottom left vertex of the cube.
        z (float): The z-coordinate of the bottom left vertex of the cube.
        vertices (np.array): The vertices of the cube after scaling and translation.
    """

    def __init__(self, width, height, depth, x, y, z, colour, alpha) -> None:
        self.color = colour
        self.alpha = alpha
        self.__width = width
        self.__height = height
        self.__depth = depth
        self.x = x
        self.y = y
        self.z = z
        cube_collection.add(self)

    def draw(self, ax):
        """
        Draw the cube on the provided 3D axis.

        Args:
                ax (Axes3D): The 3D axis on which to draw the cube.
        """
        standard_vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        temp_vertices = []
        for x1, y1, z1 in standard_vertices:
            scaled = [x1 * self.width, y1 * self.depth, z1 * self.height]
            temp_vertices.append(scaled)
        pre_translated = np.array(temp_vertices, dtype=np.float32)
        translated = self._translate_vertices(pre_translated, (self.x, self.y, self.z))

        self.vertices = np.array(translated, dtype=np.float32)
        # Plot the edges of the cube
        for edge in edges:
            ax.plot(
                self.vertices[edge, 0],
                self.vertices[edge, 1],
                self.vertices[edge, 2],
                "black",
            )

        # Create the polygons for each face
        polygons = [
            Poly3DCollection(
                [self.vertices[list(face)]], alpha=self.alpha, facecolor=self.color
            )
            for face in faces
        ]

        # Add the polygons to the plot
        for poly in polygons:
            ax.add_collection3d(poly)

    def _standard_vertices_translations(self, x, y, z):
        """
        Translate the standard unit cube vertices by the given x, y, and z values.

        Args:
            x (float): The x-coordinate translation.
            y (float): The y-coordinate translation.
            z (float): The z-coordinate translation.

        Returns:
            np.array: A NumPy array containing the translated vertices.
        """
        standard_vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        new_vertices = []
        for x1, y1, z1 in standard_vertices:
            new_vertices.append([x1 + x, y1 + y, z1 + z])

        return np.array(new_vertices, dtype=np.float32)

    def _translate_vertices(self, old_vertices, new_coords):
        """
        Translate a set of vertices by the given new coordinates.

        Args:
            old_vertices (np.array): A NumPy array containing the original vertices.
            new_coords (tuple): A tuple containing the x, y, and z translation values.

        Returns:
            np.array: A NumPy array containing the translated vertices.
        """
        return old_vertices + np.array(new_coords, dtype=np.float32)

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def depth(self):
        return self.__depth
