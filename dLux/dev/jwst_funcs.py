import equinox as eqx
import jax.numpy as np
from dLux import *
from dLux.utils import * 
from layers import *
from jax.scipy.ndimage import map_coordinates


def get_layers(planes, extras, osys, print_vals=False, tilt=True, rotate=True, distort=False, downsample=False, background=False):

    pscale = planes[0].pixelscale.to('m/pix').value
    npix = planes[0].npix
    diam = pscale*npix

    layers = [CreateWavefront(npix, diam)]
    
    if tilt:
        layers.append(TiltWavefront())

    # Iterate planes
    for i in range(len(planes)):
        plane = planes[i]
        
        if print_vals:
            print(i, plane.planetype)
            print(plane)

        # Load and apply amplitude and phase if they exist
        if not isinstance(plane.__dict__['opd'], int):
            if len(plane.__dict__['opd'].shape) == 2:

                ampl = plane.amplitude
                opd = plane.opd

                layers.append(ApplyAperture(ampl))
                layers.append(ApplyOPD(opd))

        # Handle coordinate transform
        if str(plane) == "Coordinate Inversion in y axis":
            layers.append(InvertY())
            layers.append(NormaliseWavefront())

        # Apply Detector layers
        if str(plane.planetype) == "PlaneType.detector":
            
            # MFT 
            det_npix = (plane.fov_pixels * plane.oversample).value
            pscale = plane.pixelscale.to('radian/pix').value / plane.oversample
            layers.append(AngularMFT(pscale, det_npix))
            
            # Jitter
            sigma = extras['jitter_sigma'] # Assumed arcseconds
            pixscale = plane.pixelscale.to('arcsec/pix').value / plane.oversample
            sigma_pix = sigma / pixscale
            det_layers = [ApplyJitter(sigma_pix)]
            
            # Rotation
            if rotate:
                aper = osys._detector_geom_info.aperture
                rotate_value = getattr(aper, "V3IdlYAngle")
                det_layers.append(RotateImage(rotate_value))
            
            # Distortion
            if distort:
                aper = osys._detector_geom_info.aperture
                coeffs_dict = aper.get_polynomial_coefficients()

                det_layers.append(ApplySiafDistortion(coeffs_dict['Sci2IdlX'], 
                                                      coeffs_dict['Sci2IdlY'],
                                                      aper.XSciRef,
                                                      aper.YSciRef,
                                                      # aper.XDetSize/2, # Sci cens, to be loaded from model later
                                                      # aper.YDetSize/2, # Sci cens, to be loaded from model later
                                                      (aper.XDetSize+1)/2, # Sci cens, to be loaded from model later
                                                      (aper.YDetSize+1)/2, # Sci cens, to be loaded from model later
                                                      plane.pixelscale.value / plane.oversample,
                                                      plane.oversample))
                
            # Downsampling
            if downsample:
                det_layers.append(IntegerDownsample(plane.oversample))
            
            # Detector background noise
            if background:
                det_layers.append(AddConstant(0.))

        if print_vals:
            print()

    layers.append(InvertXY())
    
    return layers, det_layers


# Wrapper classes for calling the invertion methods of Wavefront
class InvertY(eqx.Module):
    def __call__(self, params_dict):
        wf = params_dict["Wavefront"]
        wf = wf.invert_y()
        params_dict["Wavefront"] = wf
        return params_dict
    
class InvertX(eqx.Module):
    def __call__(self, params_dict):
        wf = params_dict["Wavefront"]
        wf = wf.invert_x()
        params_dict["Wavefront"] = wf
        return params_dict
    
class InvertXY(eqx.Module):
    
    def __call__(self, params_dict):
        wf = params_dict["Wavefront"]
        wf = wf.invert_x_and_y()
        params_dict["Wavefront"] = wf
        return params_dict

class RotateImage(eqx.Module):
    """
    Rotates an image paraxially
    Angle is in degrees (for now)
    
    TDOO: Fourier rotation!
    
    """
    angle: float

    def __init__(self, angle):
        self.angle = np.array(angle).astype(float)

    def __call__(self, image):
        """

        """
        # Apply Rotation
        image_out = self.paraxial_rotate(image, self.angle)
        return image_out

    def paraxial_rotate(self, array, rotation, order=1):
        """
        array: Array to rotate
        rotation: rotation angle in degrees
        order: interpolation order, supports 0 and 1

        Basically a jax version of scipy.ndimage.rotate - Contribute to Jax?
        """

        # Get coords arrays
        npix = array.shape[0]
        centre = (npix-1)/2
        x_pixels, y_pixels = get_pixel_positions(npix)
        rs, phis = cart2polar(x_pixels, y_pixels)
        phis += deg2rad(rotation)
        coordinates_rot = np.roll(polar2cart(rs, phis) + centre, shift=1, axis=0)
        rotated = map_coordinates(array, coordinates_rot, order=order)
        return rotated


from jax.scipy.ndimage import map_coordinates

class ApplySiafDistortion(eqx.Module):
    """
    Applies Science to Ideal distortion following webbpsf/pysaif
    """
    Sci2IdlX:    float
    Sci2IdlY:    float
    XSciRef:     float
    YSciRef:     float
    xsci_cen:    float
    ysci_cen:    float
    pixel_scale: float
    oversample : int
    
    def __init__(self, Sci2IdlX, Sci2IdlY, XSciRef, YSciRef, xsci_cen, ysci_cen, pixel_scale, oversample):
        self.Sci2IdlX = np.array(Sci2IdlX).astype(float)
        self.Sci2IdlY = np.array(Sci2IdlY).astype(float)
        self.XSciRef = float(XSciRef)
        self.YSciRef = float(YSciRef)
        self.xsci_cen = float(xsci_cen) # Eventually loaded from model
        self.ysci_cen = float(ysci_cen) # Eventually loaded from model
        self.pixel_scale = float(pixel_scale)
        self.oversample = int(oversample)
        
    def __call__(self, image):
        """
        
        """
        image_out = self.apply_Sci2Idl_distortion(image)
        return image_out
    
    # def apply_Sci2Idl_distortion(self, image, pixel_scale, oversample, coeffs, sci_refs, sci_cens):
    def apply_Sci2Idl_distortion(self, image):
        """
        Applies the distortion from the science (ie images) frame to the idealised telescope frame
        """
        # Convert sci cen to idl frame
        xidl_cen, yidl_cen = self.distort_coords(self.Sci2IdlX, self.Sci2IdlY, 
                                                 self.xsci_cen - self.XSciRef, 
                                                 self.ysci_cen - self.YSciRef)

        # Get paraxial pixel coordinates and detector properties
        xarr, yarr = get_pixel_positions(image.shape[0])

        # Scale and shift coordinate arrays to 'idl' frame 
        xidl = xarr * self.pixel_scale + xidl_cen
        yidl = yarr * self.pixel_scale + yidl_cen

        # Scale and shift coordinate arrays to 'sci' frame 
        xnew = xarr / self.oversample + self.xsci_cen
        ynew = yarr / self.oversample + self.ysci_cen

        # Convert requested coordinates to 'idl' coordinates
        xnew_idl, ynew_idl = self.distort_coords(self.Sci2IdlX, self.Sci2IdlY, 
                                                 xnew - self.XSciRef, 
                                                 ynew - self.YSciRef)

        # Create interpolation coordinates
        centre = (xnew_idl.shape[0]-1)/2
        coords_distort = (np.array([ynew_idl, xnew_idl]) / self.pixel_scale) + centre

        # Apply distortion
        distorted = map_coordinates(image, coords_distort, order=1)
        return distorted
    
    def distort_coords(self, A, B, X, Y):
        """
        This is gross. I am aware. Don't @ me aight.
        """
        Xnew =  A[0]  * X**0 * Y**0 + \
                A[1]  * X**1 * Y**0 + \
                A[2]  * X**0 * Y**1 + \
                A[3]  * X**2 * Y**0 + \
                A[4]  * X**1 * Y**1 + \
                A[5]  * X**0 * Y**2 + \
                A[6]  * X**3 * Y**0 + \
                A[7]  * X**2 * Y**1 + \
                A[8]  * X**1 * Y**2 + \
                A[9]  * X**0 * Y**3 + \
                A[10] * X**4 * Y**0 + \
                A[11] * X**3 * Y**1 + \
                A[12] * X**2 * Y**2 + \
                A[13] * X**1 * Y**3 + \
                A[14] * X**0 * Y**4

        Ynew =  B[0]  * X**0 * Y**0 + \
                B[1]  * X**1 * Y**0 + \
                B[2]  * X**0 * Y**1 + \
                B[3]  * X**2 * Y**0 + \
                B[4]  * X**1 * Y**1 + \
                B[5]  * X**0 * Y**2 + \
                B[6]  * X**3 * Y**0 + \
                B[7]  * X**2 * Y**1 + \
                B[8]  * X**1 * Y**2 + \
                B[9]  * X**0 * Y**3 + \
                B[10] * X**4 * Y**0 + \
                B[11] * X**3 * Y**1 + \
                B[12] * X**2 * Y**2 + \
                B[13] * X**1 * Y**3 + \
                B[14] * X**0 * Y**4
        return Xnew, Ynew
    