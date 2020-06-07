import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import numpy as np
from imageio import imread

import scipy.ndimage as ndi

from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

# Utitily functions for interactive plots 
def addPoint(path_collection, new_point, c='b'):
    """
    Add a new point to a path collection as produced by `plt.scatter`
    """
    offsets = path_collection.get_offsets()
    facecls = path_collection.get_facecolors()
    if facecls.shape[0] == 1:
        facecls = np.repeat(facecls, offsets.shape[0], axis=0)
    offsets = np.concatenate([offsets, np.array(new_point, ndmin=2)])
    facecls = np.concatenate([facecls, np.array(plt.matplotlib.colors.to_rgba(c), ndmin=2)])
    path_collection.set_offsets(offsets)
    path_collection.set_facecolors(facecls)
    path_collection.axes.figure.canvas.draw_idle()
    
def rmPoint(path_collection, coordinates):
    """
    Remove a point from a path collection as produced by `plt.scatter`
    """
    offsets = path_collection.get_offsets()
    facecls = path_collection.get_facecolors()
    if facecls.shape[0] == 1:
        facecls = np.repeat(facecls, offsets.shape[0], axis=0)
    idx_closest=np.argmin(np.sum((offsets-np.array(coordinates, ndmin=2))**2., axis=1))
    offsets = np.delete(offsets, idx_closest, axis=0)
    facecls = np.delete(facecls, idx_closest, axis=0)
    path_collection.set_offsets(offsets)
    path_collection.set_facecolors(facecls)
    path_collection.axes.figure.canvas.draw_idle()

class Image():

    def __init__(
        self, 
        file_name=None, 
        pixel_size=1., 
        pixel_size_units='µm',
        autocrop=True, ):
        if file_name:
            self.load(file_name, True)
        else:
            self.data = np.zeros([32, 32])
        self.set_scale(pixel_size, pixel_size_units)
        if autocrop:
            # Remove the bottom info strip of the SEM image
            self.data = self.data[:750, :]

    def load(self, file_name, to_float=True):
        self.data = imread(file_name) 
        if to_float: 
            # Let's hope the image is a uint8 to float in range 0-1
            self.data = self.data / 2**8 

    def set_scale(self, size, units='µm'):
        self._ps = size
        self._pu = units

    def show(self, fig=None, ax=None, colorbar=False, scalebar_size=5):
        if fig is None:
            fig, ax = plt.subplots()
        elif (fig is not None) and (ax is None):
            ax = fig.add_subplot()
        im = ax.imshow(self.data, cmap=plt.cm.afmhot)
        if colorbar:
            plt.colorbar(im, None, ax)
        ax.axis('off')

        if scalebar_size:
            self._add_scalebar_artist(ax, scalebar_size)

        return fig, ax

    def _add_scalebar_artist(self, axis, size_scaled=5):
        """ 
        Cheap scalebar, from matplotlib cook-book.
        """
        scalebar_size_pix = size_scaled / self._ps
        scalebar = AnchoredSizeBar(
            transform = axis.transData,
            size = scalebar_size_pix, 
            label = '{0:d} {1}'.format(int(round(size_scaled)), self._pu),
            loc = 'lower left', 
            pad = 1,
            color = 'white',
            frameon = False,
            label_top = True,
            size_vertical = 8, )
        axis.add_artist(scalebar)
    
    def median_filter(self, *args, **kwargs):
        """
        wrapping `ndimage.median_filter`
        """
        self.data = ndi.median_filter(self.data, *args, **kwargs)
    
    def gaussian_filter(self, *args, **kwargs):
        """
        wrapping `ndimage.gaussian_filter`
        """
        self.data = ndi.gaussian_filter(self.data, *args, **kwargs)

class FindMaxima():

    def __init__(self, image):
        """
        image should be of Image type (see above).
        """
        if not isinstance(image, Image):
            raise AttributeError(
                'The provided image attribute must be of Image type')
        self.image = image
        self.update()

    def update(self):
        self.auto_foreground()
        self.auto_fill_basins() # see below
        
    def auto_foreground(self, threshold_factor=1., *args, **kwargs):
        """
        *args and **kwargs used to set binary remove small objects filter.
        """ 
        # Binary filter options
        if len(args) == 0:
            kwargs.setdefault('structure', disk(2))
            kwargs.setdefault('iterations', 5)

        # heuristic threshold
        self.foreground_threshold = \
            threshold_otsu(self.image.data) * threshold_factor
        
        mask = self.image.data > self.foreground_threshold
        mask = ndi.binary_erosion(mask, *args, **kwargs)
        mask[0,:] = 0
        mask[:,1] = 0
        self.foreground = ndi.binary_dilation(mask, *args, **kwargs)

    def auto_fill_basins(self, initial_peaks=25, *args, **kwargs):
        
        # Binary filter options
        if len(args) == 0:
            kwargs.setdefault('iterations', 4)
        
        # Find Local Max I: builds a mask array with the foreground maxima
        foreground_labels, Nlabels = ndi.label(self.foreground)
        maxima_mask = peak_local_max(
            image = self.image.data,
            labels = foreground_labels,
            num_peaks_per_label=initial_peaks,
            min_distance=1, 
            indices=False, )
        maxima_mask = ndi.binary_dilation(
            maxima_mask, *args, **kwargs)
        maxima_labels, Nlabels = ndi.label(maxima_mask)

        # Find Local Max II: run again find local max for cleanup
        maxima_mask = peak_local_max(
            image = self.image.data,
            labels = maxima_labels,
            num_peaks_per_label=1,
            min_distance=1, 
            indices=False, )
        self.markers = np.argwhere(maxima_mask)

        # Finally use the watershed algorithm to label the basins
        maxima_labels, self.Nlabels = ndi.label(maxima_mask)
        self.labels = watershed(-self.image.data, maxima_labels)

class PyramidTool():

    def __init__(self, image, *args, **kwargs):
        self.image = image
        self.find_maxima = FindMaxima(self.image)
        self.interactive_plot(*args, **kwargs)

    def interactive_plot(self, *args, **kwargs):
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, *args, **kwargs)
        Ny, Nx = self.image.data.shape
        self.x, self.y = np.arange(Nx), np.arange(Ny)

        # First panel: normal image, foreground and markers
        ax = axs[0]
        self.image.show(fig=fig, ax=ax)
        
        self.contours = ax.contour(
            self.x, 
            self.y, 
            self.find_maxima.foreground, 
            [0.5], 
            linewidths=0.5, 
            colors='g', )
        self.pc_markers = ax.scatter(
            self.find_maxima.markers[:, 1], 
            self.find_maxima.markers[:, 0], 
            s=1, 
            c='r', 
            marker='o', )

        # Second panel: the watershed basins
        ax = axs[1]
        self.image.show(fig=fig, ax=ax)
        ax.images[0].set_cmap(plt.cm.gray)
        self.watershed_image = ax.imshow(
            self.find_maxima.labels, cmap=plt.cm.flag, alpha=.2)

        # interactivity            
        def onClick(event):
            if not event.inaxes:
                return
            if event.button == 1:
                addPoint(self.pc_markers, (event.xdata, event.ydata))
            elif event.button == 3:
                rmPoint(self.pc_markers, (event.xdata, event.ydata))
            print(self.pc_markers)
            self.update_watershed()

        cid = fig.canvas.mpl_connect('button_press_event', onClick)
        fig.tight_layout()
        
        self.axes = axs

    def update_watershed(self):
        # fill the basins using the existing path collection markers
        offsets = self.pc_markers.get_offsets()
        new_markers = np.round(offsets[:, ::-1], decimals=0)
        new_markers = new_markers.astype('int')
        new_mask = np.zeros_like(self.image.data, dtype=bool)
        new_markers = np.ravel_multi_index(new_markers.T, new_mask.shape)
        np.ravel(new_mask)[new_markers] = True
        new_labels, new_Nlabels = ndi.label(new_mask)
        watershed_labels = watershed(-self.image.data, new_labels)

        # update the plot
        ax = self.axes[1]
        # clean old lines (from measurements)
        ax.lines = []
        ax.figure.canvas.draw_idle()
        self.watershed_image.set_data(watershed_labels)
        self.watershed_image.set_clim(0, new_Nlabels)

    def load_coordinates(self, file_name, color='y'):
        """
        Use the `color` attribute to set the color of the scatter plot.
        """
        offsets = np.load(file_name)
        facecls = np.repeat(
            np.array(plt.matplotlib.colors.to_rgba(color))[None, :], 
            offsets.shape[0], 
            axis=0, )
        self.pc_markers.set_offsets(offsets)
        self.pc_markers.set_facecolors(facecls)
        self.pc_markers.axes.figure.canvas.draw_idle()  
        self.update_watershed()

    def save_coordinates(self, file_name):
        coordinates = self.pc_markers.get_offsets()
        coordinates = np.asarray(coordinates, np.float32)
        np.save(file_name,coordinates) 

    def measure_pyramids(self, pyramid_angle=45.):

        watershed_labels = self.watershed_image.get_array()
        # mark the pixels at the border
        boundaries = find_boundaries(watershed_labels)
        # find maxima in the watershed as array
        watershed_max = peak_local_max(
            image = self.image.data,
            labels = watershed_labels,
            num_peaks_per_label=1,
            min_distance=1, 
            indices=False, )

        # pyramid angles
        pyangle = np.deg2rad(pyramid_angle)
        pyangle_n = np.arange(4) * np.pi / 2 + pyangle
        pyangle_n = pyangle_n - np.pi 

        labels = np.unique(watershed_labels)
        Nlabels = len(labels)
        self.pyramid_vertex_coordinates = np.zeros([Nlabels, 4, 2])
        self.pyramid_center_coordinates = np.zeros([Nlabels, 2])
        lines  = []
        for io, label in enumerate(labels):
            # find the border pixels
            label_mask = watershed_labels == label
            label_bounds = boundaries.copy()
            label_bounds[~label_mask] = 0
            bounds = np.argwhere(label_bounds)

            # find the center pixel
            label_max = watershed_max.copy()
            label_max[~label_mask] = 0
            center = np.argwhere(label_max)
            self.pyramid_center_coordinates[io, ...] = center 

            # get the angles
            distances = bounds - center
            angles = np.arctan2(distances[:, 1], distances[:, 0])
            dists = []
            for ai in pyangle_n:
                idx = np.argmin(np.abs(angles - ai))
                coord = bounds[idx]
                dist = distances[idx]
                dists.append(dist)
            dists = np.array(dists)
            self.pyramid_vertex_coordinates[io, ...] = dists
            
        self.update_plot_with_measurements()

    def update_plot_with_measurements(self):
        # update plot
        ax = self.axes[1]

        # clean old lines
        ax.lines = []
        ax.figure.canvas.draw_idle()

        # put new lines
        pccs = self.pyramid_center_coordinates
        pvcs = self.pyramid_vertex_coordinates 
        for center, vertices in zip(pccs, pvcs):
            for vertex in vertices:
                coord = vertex + center
                line = plt.Line2D(
                    xdata = [coord[1], center[1]], 
                    ydata = [coord[0], center[0]],
                    linestyle= '--',
                    linewidth= 0.5, )
                ax.add_line(line)  
    
    def save_pyramid_measurements(self, file_name):

        np.savez(
            file_name, 
            centers=self.pyramid_center_coordinates,
            vertices=self.pyramid_vertex_coordinates, )
    
    def load_pyramid_measurements(self, file_name):

        npzdata = np.load(file_name)
        self.pyramid_center_coordinates = npzdata['centers']
        self.pyramid_vertex_coordinates = npzdata['vertices']
        self.update_plot_with_measurements()

    