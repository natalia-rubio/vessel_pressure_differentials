import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_radii(length, inlet_radius, outlet_radius, stenosis_dict, elem_length=1):
    """
    Returns a list of radii
    """
    num_pts = int(length / elem_length)
    base_radii = np.linspace(inlet_radius, outlet_radius, num_pts).astype(float)

    pts = np.linspace(0, num_pts, num_pts)
    stenosis_pdf = norm.pdf(pts, loc = num_pts*stenosis_dict["location"], scale = num_pts*stenosis_dict["spread"])
    stenosis_multiplier = 1 + stenosis_dict["magnitude"] * stenosis_pdf/np.max(stenosis_pdf)
    print(stenosis_multiplier)
    return list(base_radii * stenosis_multiplier)

def plot_radii(radii):
    """
    Plots radius along straight path
    """
    plt.clf()
    plt.plot(np.array(radii), "b")
    plt.plot(-1*np.array(radii), "b")
    plt.show()
    return

if __name__ == '__main__':
    stenosis_dict = {"magnitude": 0.5, 
                    "spread": 0.3,
                    "location": 0.5}
    radii = get_radii(length=10, inlet_radius=1, outlet_radius=2, stenosis_dict=stenosis_dict)
    plot_radii(radii)