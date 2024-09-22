import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import scipy.ndimage
import scipy.signal
from sklearn.cluster import KMeans

d_display = np.load(r"C:\Users\Realme\Downloads\Copy of Kerry_full.npy")
a = d_display.shape
print(a)

def explore3d(data_cube):
    source = mlab.pipeline.scalar_field(data_cube)
    source.spacing = [1, 1, -1]

    nx, ny, nz = data_cube.shape
    mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes', 
                                     slice_index=nx//2, colormap='gray')
    mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes', 
                                     slice_index=ny//2, colormap='gray')
    mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes', 
                                     slice_index=nz//2, colormap='gray')
    mlab.show()

# explore3d(d_display)

def bahorich_coherence(data, zwin):
    ni, nj, nk = data.shape
    out = np.zeros_like(data)
    
    # Pad the input to make indexing simpler. We're not concerned about memory usage.
    # We'll handle the boundaries by "reflecting" the data at the edge.
    padded = np.pad(data, ((0, 1), (0, 1), (zwin//2, zwin//2)), mode='reflect')

    for i, j, k in np.ndindex(ni, nj, nk):
        # Extract the "full" center trace
        center_trace = data[i,j,:]
        
        # Use a "moving window" portion of the adjacent traces
        x_trace = padded[i+1, j, k:k+zwin]
        y_trace = padded[i, j+1, k:k+zwin]

        # Cross correlate. `xcor` & `ycor` will be 1d arrays of length
        # `center_trace.size - x_trace.size + 1`
        xcor = np.correlate(center_trace, x_trace)
        ycor = np.correlate(center_trace, y_trace)
        
        # The result is the maximum normalized cross correlation value
        center_std = center_trace.std()
        px = xcor.max() / (xcor.size * center_std * x_trace.std())
        py = ycor.max() / (ycor.size * center_std * y_trace.std())
        out[i,j,k] = np.sqrt(px * py)

    return out




# op = bahorich_coherence(d_small,3)

# plt.imshow(op[49,:,:].T,interpolation = 'nearest',cmap='gray', aspect = 'auto')
# explore3d(d_display)
# # explore3d(op)
# plt.show

'''GRADIENT STRUCTURE TENSOR'''

def gradients(seismic, sigma):
    """Builds a 4-d array of the gaussian gradient of *seismic*."""
    grads = []
    for axis in range(3):
        # Gaussian filter with order=1 is a gaussian gradient operator
        grad = scipy.ndimage.gaussian_filter1d(seismic, sigma, axis=axis, order=1)
        grads.append(grad[..., np.newaxis])
    return np.concatenate(grads, axis=3)

def moving_window4d(grad, window, func):
    """Applies the given function *func* over a moving *window*, reducing
    the input *grad* array from 4D to 3D."""
    # Pad in the spatial dimensions, but leave the gradient dimension unpadded.
    half_window = [(x // 2, x // 2) for x in window] + [(0, 0)]
    padded = np.pad(grad, half_window, mode='reflect')

    out = np.empty(grad.shape[:3], dtype=float)
    for i, j, k in np.ndindex(out.shape):
        region = padded[i:i+window[0], j:j+window[1], k:k+window[2], :]
        out[i,j,k] = func(region)
    return out

def gst_coherence_calc(region):
    
    region = region.reshape(-1, 3)
    gst = region.T.dot(region) # This is the 3x3 gradient structure tensor

    # Reverse sort of eigenvalues of the GST (largest first)
    eigs = np.sort(np.linalg.eigvalsh(gst))[::-1]

    return (eigs[0] - eigs[1]) / (eigs[0] + eigs[1])

def gst_coherence(seismic, window, sigma=1):
    
    
    grad = gradients(seismic, sigma)
    return moving_window4d(grad, window, gst_coherence_calc)

d_small = d_display[:, :, :100]
gst_coh = gst_coherence(d_small, (3, 3, 9), sigma=1)

# explore3d(gst_coh)
# explore3d(d_small)


# function to plot in separate windows
def Explore3d(data_cube, figure=None):
    if figure is None:
        figure = mlab.figure()
    else:
        mlab.figure(figure)

    source = mlab.pipeline.scalar_field(data_cube)
    source.spacing = [1, 1, -1]

    nx, ny, nz = data_cube.shape
    mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes',
                                     slice_index=nx//2, colormap='gray')
    mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes',
                                     slice_index=ny//2, colormap='gray')
    mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes',
                                     slice_index=nz//2, colormap='gray')

    return figure

# Explore3d(d_small,figure='cube1')
# Explore3d(gst_coh,figure='cube2')

'''K - MEANS'''
'''K - MEANS'''
'''K - MEANS'''

flat = gst_coh.flatten().reshape(-1, 1)



kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(flat)
faults = kmeans.labels_.reshape((d_small.shape[0],d_small.shape[1],d_small.shape[2]))
# explore3d(faults)
# function to plt saults on seismic section

def explore3d_overlap(data_cube1, data_cube2):
    # Create scalar field sources for both data cubes
    source1 = mlab.pipeline.scalar_field(data_cube1)
    source1.spacing = [1, 1, -1]

    source2 = mlab.pipeline.scalar_field(data_cube2)
    source2.spacing = [1, 1, -1]

    nx, ny, nz = data_cube1.shape

    # Add image plane widgets for both data cubes
    mlab.pipeline.image_plane_widget(source1, plane_orientation='x_axes', 
                                     slice_index=nx//2, colormap='gray')
    mlab.pipeline.image_plane_widget(source1, plane_orientation='y_axes', 
                                     slice_index=ny//2, colormap='gray')
    mlab.pipeline.image_plane_widget(source1, plane_orientation='z_axes', 
                                     slice_index=nz//2, colormap='gray')

    mlab.pipeline.image_plane_widget(source2, plane_orientation='x_axes', 
                                     slice_index=nx//2, colormap='red',opacity=0.2)
    mlab.pipeline.image_plane_widget(source2, plane_orientation='y_axes', 
                                     slice_index=ny//2, colormap='red',opacity=0.2)
    mlab.pipeline.image_plane_widget(source2, plane_orientation='z_axes', 
                                     slice_index=nz//2, colormap='red',opacity=0.2)

    mlab.show()
    
# explore3d_overlap(d_small,faults)  
    
Explore3d(d_small,figure='cube1')
Explore3d(faults,figure='cube2')