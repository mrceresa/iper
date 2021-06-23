import numpy as np
from scipy import spatial
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt
from matplotlib import  colors
import utm
from statistics import mean
import pandas as pd
import math

from datetime import datetime, timedelta

class Interpolation_Diffusion_Model():
    def __init__(self, boundaries, init_time):
        # Grid boundaries
        self.boundaries = boundaries
        self.current_time = init_time
        self.corner_0_0 = utm.from_latlon(self.boundaries["w"],self.boundaries["s"])
        self.corner_1_1 = utm.from_latlon(self.boundaries["e"], self.boundaries["n"])

        # Read Data, Once every month
        root_path = os.getcwd()
        path_name = "/examples/bcn_multispace/EudaldMobility/Pollution/qualitat_estacions_NOx.csv"
        self.dataset = pd.read_csv(root_path + path_name)

        # Define the stations
        stations = {}
        for index, item in self.dataset.loc[self.dataset['Dia'] == 1].iterrows():
            stations[item['Estacio']] = (item['Longitud'], item['Latitud'])

        # Create Target Grid to interpolate to 
        self.create_grid()

        # Reference Stations and Corners to grid
        self.poi_x, self.poi_y = self.poi_to_grid(stations)

        # Initialize grid every hour
        self.update_pollution_next_hour(self.current_time)

        #Interpolate
        self.interpolate()
        
        # Difussion
        #Init particles every hour
        self.init_particles() #location of each particle

        #self.locs = self.diffusion(self.locs)
        
        self.update_particles()

        self.plot_2D()
        self.plot_3D()

    def create_grid(self):
        m = 100 #Divison grid number
        xi = np.linspace(self.corner_0_0[0],self.corner_1_1[0], m)
        yi = np.linspace(self.corner_0_0[1],self.corner_1_1[1], m)
        self.X, self.Y = np.meshgrid(xi, yi)

    def poi_to_grid(self, stations):
        # Returns the projected coordinates x,y for stations and corners
        # Add stations
        x = []
        y = []
        for index, coord in stations.items():
            coord_proj = utm.from_latlon(coord[0], coord[1])
            x.append(coord_proj[0])
            y.append(coord_proj[1])
        # Add corners
        x += [self.corner_0_0[0],self.corner_1_1[0], self.corner_0_0[0], self.corner_1_1[0]]
        y += [self.corner_0_0[1],self.corner_1_1[1], self.corner_1_1[1], self.corner_0_0[1]]
        return x,y

    def update_pollution_next_hour(self, current_time):
        self.current_time = current_time
        current_hour = self.current_time.hour + 1
        if self.current_time.hour + 1 < 10:
            hour_tag = 'H0' + str(current_hour)
        else:
            hour_tag = 'H' + str(current_hour)
        pollutants = {}
        dataset_day_hour = self.dataset.loc[(self.dataset['Dia'] == self.current_time.day)][[hour_tag,'Estacio']]
        for index, item in dataset_day_hour.iterrows():
            if math.isnan(item[hour_tag]):
                pollutants[item['Estacio']] = dataset_day_hour[hour_tag].mean(skipna = True)
            else:
                pollutants[item['Estacio']] = (item[hour_tag]) 
        
        self.z = list(pollutants.values()) + [pollutants[57],pollutants[4],pollutants[43],pollutants[57]]

    def init_particles(self):
        first_element = True
        for i in range(len(self.X)):
            for j in range(len(self.Y)):
                if first_element == True:
                    self.locs = np.zeros((int(self.zi[i,j]), 2), dtype=int) + (i,j)
                    first_element = False
                else: 
                    loc  = np.zeros((int(self.zi[i,j]), 2), dtype=int) + (i,j)
                    self.locs = np.concatenate((self.locs, loc))
    
    def update_particles(self):
        # Create an updated grid pand plot it.
        grid = np.zeros((len(self.X), len(self.Y)))
        for i in range(self.locs.shape[0]): # For each particle, add on grid
            xp, yp = self.locs[i]
            # Add a particle to the grid if it is actually on the grid!
            if 0 <= xp < len(self.X) and 0 <= yp < len(self.Y):
                grid[xp,yp] += 1
        self.z_values = grid

    def interpolate(self):
        self.zi = griddata((self.poi_x,self.poi_y),self.z,(self.X,self.Y),method='cubic')
        self.zi[self.zi < 0] = 0

    def diffusion(self):
        self.locs += np.random.randint(-1, 2, self.locs.shape)
        return self.locs
    
    def plot_2D(self):
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(self.X,self.Y,self.z_values,np.arange(min(map(min, self.z_values))-10, max(map(max, self.z_values))+10, 1), cmap=plt.cm.autumn)
        plt.plot(self.poi_x,self.poi_y,'k.')
        plt.xlabel('X',fontsize=12)
        plt.ylabel('Y',fontsize=12)
        plt.colorbar()
        plt.show()

    def plot_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.clear()
        ax.plot_surface(self.X, self.Y, self.z_values, rstride=1, cstride=1, cmap=plt.cm.autumn, linewidth=1)
        plt.show()

    def find_closest_point_in_grid(self, pt):
        A = []
        for i in np.arange(len(self.X)):
            for j in np.arange(len(self.Y)):
                A.append((self.X[i][j],self.Y[i][j]))
        result = A[spatial.KDTree(A).query(pt)[1]]
        #distance,index = spatial.KDTree(A).query(pt)
        return result

    def get_pollution_value(self, position):
        coord = self.find_closest_point_in_grid(position)
        coord_idx = np.argwhere((self.X==coord[0]) & (self.Y==coord[1]))[0]
        return self.z_values[coord_idx[0],coord_idx[1]]

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
    boundaries = {'n': 41.4676352, 'e': 2.2240543, 's': 41.3214761, 'w':  2.0711951}
    DateTime = datetime(year=2021, month=1, day=4, hour= 7, minute=0, second=0) 
    #diffusion_pollutnat(boundaries, stations)
    P = Interpolation_Diffusion_Model(boundaries, DateTime)
    position = [770000, 231000]
    print(P.get_pollution_value(position))
    

