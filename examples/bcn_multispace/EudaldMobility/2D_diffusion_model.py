import numpy as np
from scipy import spatial
import os
import matplotlib.pyplot as plt
from matplotlib import  colors
import utm

class diffusion_pollutnat():
    def __init__(self, boundaries, stations):

        # Grid boundaries
        self.boundaries = boundaries
        self.corner_0_0 = utm.from_latlon(self.boundaries["w"],self.boundaries["s"])
        self.corner_1_1 = utm.from_latlon(self.boundaries["e"], self.boundaries["n"])
        #print(self.corner_0_0)
        #print(self.corner_1_1)

        #Save location
        root_path = os.getcwd()
        path_name = "/examples/bcn_multispace/EudaldMobility/diffusion_plots/"

        # Grid division 
        m = 500
        self.ratioMap(m)

        # Maximum numbter of iterations.
        nitmax = 100
        # Number of particles in the simulation.
        nparticles = 50000
        # Output a frame (plot image) every nevery iterations.
        nevery = 2
        # Constant maximum value of z-axis value for plots.
        zmax = 10

        # Create the 3D figure object.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, self.m_x, self.m_x, dtype=int)
        y = np.linspace(0, self.m_y, self.m_y, dtype=int)
        # We'll need a meshgrid to plot the surface: this is X, Y.
        X, Y = np.meshgrid(x, y)

        # vmin, vmax set the minimum and maximum values for the colormap. This is to
        # be fixed for all plots, so define a suitable norm.
        vmin, vmax = 0, zmax
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # Initialize the location of all the particles to the centre of the stations.
        first_element = True
        for s, item in stations.items():
            station = utm.from_latlon(item[0], item[1])
            station_norm = self.normalize(station[0:2])
            if first_element == True:
                locs = np.ones((nparticles, 2), dtype=int) + station_norm
                first_element = False
            else: 
                loc = np.ones((nparticles, 2), dtype=int) + station_norm
                locs = np.concatenate((locs, loc))

        # Iterate for nitmax cycles.
        for j in range(nitmax):
            # Update the particles' locations at random. Particles move at random to
            # an adjacent grid cell. We're going to be pretty relaxed about the ~11%
            # probability that a particle doesn't move at all (displacement of (0,0)).
            locs += np.random.randint(-20, 20, locs.shape)
            if not (j+1) % nevery:
                # Create an updated grid pand plot it.
                grid = np.zeros((self.m_y, self.m_x)) #NO ENTENC PERQUE ELS HE DE GIRAR?? ALOMILLOR TRESPOSAR?
                #grid = np.zeros((50, 50))
                for i in range(locs.shape[0]):
                    x, y = locs[i]
                    # Add a particle to the grid if it is actually on the grid!
                    if 0 <= x < self.m_y and 0 <= y < self.m_x: #NO ENTENC PERQUE ELS HE DE GIRAR?? ALOMILLOR TRESPOSAR?
                        grid[x, y] += 1
                print(j+1,'/',nitmax)
                # Now clear the Axes of any previous plot and make a new surface plot.
                ax.clear()
                ax.plot_surface(X, Y, grid, rstride=1, cstride=1, cmap=plt.cm.autumn,
                                linewidth=1, vmin=vmin, vmax=vmax, norm=norm)
                ax.set_zlim(0, zmax)
                # Save to 'diff-000.png', 'diff-001.png', ...
                plt.savefig(root_path + path_name + 'diff-{:03d}.png'.format(j//nevery))

        os.system("convert -delay 5 -loop 1 -dispose previous " + root_path + path_name + "*.png " + root_path + path_name + "anim.gif")
        os.system("rm " + root_path + path_name + "*.png ")

    def find_closest_point(self, X, Y, pt):
        A = []
        print(len(X))
        print(len(Y))
        for i in np.arange(len(X)):
            for j in np.arange(len(Y)):
                A.append((X[i][j],Y[i][j]))
        result = A[spatial.KDTree(A).query(pt)[1]]
        distance,index = spatial.KDTree(A).query(pt)
        return result

    #Given a coord normalizes between 0 and mX, mY
    def normalize(self, coord):
        # Norm X coordinates
        norm_x = int(((coord[0] - self.corner_0_0[0]) / (self.corner_1_1[0] - self.corner_0_0[0])) * self.m_x)
        #Norm Y coordiantes
        norm_y = int(((coord[1] - self.corner_0_0[1]) / (self.corner_1_1[1] - self.corner_0_0[1])) * self.m_y)
        return (norm_x, norm_y)

    #Returns the matrix size having applied the ratio
    def ratioMap(self, m):
        self.ratio = (self.corner_0_0[0] - self.corner_1_1[0])/(self.corner_0_0[1] - self.corner_1_1[1])
        self.m_x = int(m * self.ratio)
        self.m_y = int(m)

if __name__ == '__main__':
    stations = {'50':(2.1874, 41.3864), 
                '54':(2.1480, 41.4261), 
                '43':(2.1538, 41.3853), 
                '44':(2.1534, 41.3987),
                '57':(2.1151, 41.3875),
                '4':(2.2045, 41.4039),
                '42':(2.1331, 41.3788)}

    boundaries = {'n': 41.4676352, 'e': 2.2240543, 's': 41.3214761, 'w':  2.0711951}
    diffusion_pollutnat(boundaries, stations)

