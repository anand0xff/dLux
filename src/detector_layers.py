import jax
import jax.numpy as np
import equinox as eqx

class ApplyPixelResponse(eqx.Module):
    """
    
    """
    pixel_response: np.ndarray
    
    def __init__(self, pixel_response):
        """
        
        """
        self.pixel_response = np.array(pixel_response)
        
    def __call__(self, image):
        """
        
        """
        image *= self.pixel_response
        return image
    
class ApplyJitter(eqx.Module):
    """
    Convolves the output image with a gaussian kernal
    """
    kernel_size: int
    sigma: float
    
    def __init__(self, sigma, kernel_size=25):
        self.kernel_size = int(kernel_size)
        self.sigma = np.array(sigma).astype(float)
        
    def __call__(self, image):
        """
        
        """
        # Generate distribution
        x = np.linspace(-10, 10, self.kernel_size)
        window = jax.scipy.stats.norm.pdf(x,          scale=self.sigma) * \
                 jax.scipy.stats.norm.pdf(x[:, None], scale=self.sigma)
        
        # Normalise
        window /= np.sum(window)
        
        # Convolve with image
        image_out = jax.scipy.signal.convolve(image, window, mode='same')
        return image_out
    
class ApplySaturation(eqx.Module):
    """
    Reduces any values above self.saturation to self.saturation
    """
    saturation: float
    
    def __init__(self, saturation):
        self.saturation = np.array(saturation).astype(float)
        
    def __call__(self, image):
        """
        
        """
        # Apply saturation
        image_out = np.minimum(image, self.saturation)
        return image_out
    
class AddConstant(eqx.Module):
    """
    Reduces any values above self.saturation to self.saturation
    """
    value: float
    
    def __init__(self, value):
        self.value = np.array(value).astype(float)
        
    def __call__(self, image):
        """
        
        """
        # Apply saturation
        image_out = image + self.value
        return image_out
    
class IntegerDownsample(eqx.Module):
    """
    Downsamples an input image by an integer number of pixels via a sum.
    The number of pixels in must by integer divisible by the kernel_size,
    ie the following must be True: image.shape[0]%self.kernel_size == 0
    
    Parameters
    ----------
    kernel_size: int
        The size of the downsampling kernel
            
    """
    kernel_size: int
    
    def __init__(self, kernel_size):
        self.kernel_size = int(kernel_size)
        
    def __call__(self, image):
        """
        
        """
        image_out = self.downsample(image, self.kernel_size)
        return image_out
    
    def downsample(self, array, kernel_size):
        """
        Downsamples a kernel_size*n size array by kernel_size
        
        TODO: Properly test arbitrary dimensions
        
        """
        size = array.shape[0]
        size_out = size//kernel_size
        
        # # Iterate over all dimensions
        # for i in range(len(array.shape)):
        #     array = array.reshape([size*size_out, kernel_size]).sum(i)
        #     array = array.reshape([size, size_out]).T

        # Dim 1
        # array = array.reshape([size**2 // kernel_size, kernel_size])
        array = array.reshape([size*size_out, kernel_size]).sum(1)
        array = array.reshape(size, size_out).T
        # array = array.T

        # Dim 2
        array = array.reshape(size_out**2, kernel_size).sum(1)
        # array = np.sum(array, axis=1)
        array = array.reshape(size_out, size_out).T
        # array = array.T
        return array