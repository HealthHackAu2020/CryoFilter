import numpy as np
from skimage import filters
import cv2
from scipy import fftpack, ndimage


class ImageTransform():
    def __init__(self):
        self.name = None
        
    def transform(self, arr):
        '''
        Apply transform to single array
        '''
        return arr
    
    def apply(self, data):
        print("Applying transform: "+self.name)
        return [self.transform(x) for x in data]
    
    
class IdentityTransform(ImageTransform):
    def __init__(self, resized_shape=(28,28)):
        super().__init__()
        self.name = "Identity"
        self.resized_shape = resized_shape
    
    
    
class RobertsTransform(ImageTransform):
    def __init__(self, resized_shape=(28,28)):
        super().__init__()
        self.name = "Roberts"
        self.resized_shape = resized_shape
    
    def transform(self, arr):
        # Resize
        res = cv2.resize(arr, dsize=self.resized_shape, interpolation=cv2.INTER_LINEAR)
        
        # Filter
        return filters.roberts(res)
    

class FFT2Transform(ImageTransform):
    def __init__(self, central_width=50):
        super().__init__()
        self.name = "FFT2"
        self.central_width = central_width
        
    def transform(self, arr):
        # Get absolute values of 2D FFT, with centrally shifting
        fft2 = np.fft.fftshift(np.abs(fftpack.fft2(arr)))
        
        # Get center crop
        H, W = fft2.shape
        C = self.central_width
        return fft2[int((H-C)/2):int((H-C)/2)+C, int((W-C)/2):int((W-C)/2)+C]