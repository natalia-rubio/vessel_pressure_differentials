import numpy as np
#import matplotlib.pyplot as plt

def get_path(length, curvature, elem_length=1):
    """
    Returns a path of a given length and curvature
    curvature varies between -90 and 90 degrees
    """
    num_pts = int(length / elem_length)
    angles = np.linspace(0, curvature, num_pts-1)
    path = [[0.0, 0.0, 0.0],]

    for i in range(num_pts-1):
        x = float(path[-1][0] + elem_length * np.sin(angles[i]*np.pi/180))
        y = float(path[-1][1] + elem_length * np.cos(angles[i]*np.pi/180))
        z = float(0)
        path.append([x, y, z])
    path.reverse()
    return path

def plot_2D_path(path):
    """
    Plots a path
    """
    plt.clf()
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1])
    plt.show()
    return

if __name__ == '__main__':
    path = get_path(length=10, curvature=90)
    # plot_2D_path(path)