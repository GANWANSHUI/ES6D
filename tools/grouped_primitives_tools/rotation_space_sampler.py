import numpy as np
import matplotlib.pyplot as plt
import math


def fibonacci_sphere(num_samples=1):
    """
    Sampling points in the surface of unit sphere

    Args:
        num_samples: the number of samples

    Returns:
        points: [(x1, y1, z1), (x2, y2, z2), ...]

    """
    points = []
    phi = 3. - math.sqrt(5.)
    phi = math.pi * phi  # golden angle in radians

    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def symmetric_axis_angle_sampler(num=5000, min_angle=10, max_angle=180):
    """
    Sampling symmetric axis-angles in rotation space

    Args:
        num: the number of unit axes in the surface of unit sphere
        min_angle: the minimum symmetric rotation angle of interest
        max_angle: the maximum symmetric rotation angle of interest

    Returns:
        aas: [(x1, y1, z1, theta1), (x2, y2, z2, theta2), ...]

    """
    print('start sampling.')
    pts = fibonacci_sphere(num)  # points
    aas = []  # axis-angles
    for i in range(len(pts)):
        for j in range(min_angle, max_angle+1):
            if 360 % j == 0:
                theta = float(j) * math.pi / 180.
                aas.append((pts[i][0], pts[i][1], pts[i][2], theta))
    print('sampling finished.')
    return aas


if __name__ == '__main__':
    n = 20000
    print('start sampling.')

    pts = fibonacci_sphere(n)
    aas = []
    for i in range(len(pts)):
        for j in range(360):
            theta = float(j) / math.pi
            aas.append([pts[i][0], pts[i][1], pts[i][2], theta])
    print('sampling finished.')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    pts = np.asarray(pts, dtype='float')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='.', s=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_box_aspect((1., 1., 1.))

    plt.show()
