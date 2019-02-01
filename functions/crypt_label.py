"""
Small, single use functions for crypt labeling

"""

import csv


from skimage import img_as_uint
from skimage import io
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, watershed
from skimage.feature import peak_local_max
from skimage.color import label2rgb
from scipy.ndimage import distance_transform_edt



def crypt_thresh_label(filepath, threshold_method=threshold_otsu, gaussian_size=2, closing_size=15, local_min=18):
    """Threshold, watershed, and label crypts
    
    Parameters:
    filepath = string to file
    threshold_method = skimage threshold method
    
    """
    crypt_image = io.imread(filepath)
    crypt_image = gaussian(crypt_image, gaussian_size)
    binary = crypt_image > threshold_method(crypt_image)
    binary = closing(binary, square(closing_size))
    distance = distance_transform_edt(binary)
    local_max = peak_local_max(distance, indices=False, min_distance=local_min)
    markers = label(local_max)
    return watershed(-distance, markers, mask=binary)

def save_image_and_csv(labeled_image, path_to_files, save_image=True):
    """Output labeled image to csv with area and coordinate info
    Writes file in path to csv
    
    Parameters:
    labeled_image = skimage np.array that is labeled (i.e.
    the image returned from crypt_thresh_label)
    path_to_files = name of tif and csv to save (extension is added)
    
    """
    csv_path = path_to_files + ".csv"
    if save_image:
        tif_path = path_to_files + ".tif"
        io.imsave(tif_path, img_as_uint(label2rgb(labeled_image)), 'tifffile')
    
    with open(csv_path, 'w', newline='') as write_labels:
        write_csv = csv.writer(write_labels, delimiter=',')
        write_csv.writerow(('x', 'y', 'area_pixels'))
        for region in regionprops(labeled_image):
            center = region.centroid
            x, y = int(center[0]), int(center[1])
            write_csv.writerow((x, y, region.area))
    return
