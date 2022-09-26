# import jax
import jax.numpy as np
import equinox as eqx
import abc
import typing
import dLux

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["Source", "PointSource", "ResolvedSource",
           "GaussianSource", "ExtendedSource", "BinarySource"]

# Base Jax Types
Array =  typing.NewType("Array",  np.ndarray)

# Spectrum = typing.NewType("Spectrum", dLux.spectrums.Specturm)
Spectrum = typing.NewType("Spectrum", object)
Source   = typing.NewType("Source",   object)

"""
TODO: Build out resolved sources properly

High level - what do we want here?
Source, Abstract
    Point Source, Concrete
    Resolved Source, abstract
        Gaussian source, concrete (param'd by some gaussian distribution)
        Extended Source, concrete (Arbitrary grid definition)
"""

class Source(dLux.base.Base, abc.ABC):
    """
    Base class for source objects. The idea of these source classes is to allow
    an arbitrary parametrisation of the underlying astrophyical objects,
    through which cartesain parameters are passed up to the higher level
    classes which is then used to model them through the optics.
    
    Attributes
    ----------
    resolved : bool
        Is this a point source or a resolved source.
    position : Array, radians
        The (x, y) on-sky position of this object. Units are currently in
        radians, but will likely be extended to RA/DEC.
    flux : Array, photons
        The flux of the object.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    name : str
        The name for this object.
    """
    resolved          : bool
    position          : Array
    flux              : Array
    spectrum          : Spectrum
    name              : str = eqx.static_field()
    
    
    def __init__(self              : Source, 
                 position          : Array,
                 flux              : Array,
                 spectrum          : Spectrum,
                 resolved          : bool,
                 name              : str = 'Source') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        resolved : bool
            Is this a point source or a resolved source.
        name : str (optional)
            The name for this object. Defaults to 'Source'
        """
        self.position          = np.asarray(position, dtype=float)
        self.flux              = np.asarray(flux,     dtype=float)
        self.spectrum          = spectrum # Will this error if its not a 'Spectrum' class? I hope so...
        self.resolved          = bool(resolved)
        self.name              = name
    
    
    ### Start Getter Methods ###
    def get_flux(self : Source) -> Array:
        """
        Getter method for the flux.
        
        Returns
        -------
        flux : Array, photons
            The flux of the object.
        """
        return self.flux
    
    
    def get_position(self : Source) -> Array:
        """
        Getter method for the position.
        
        Returns
        -------
        position : Array, radians
            The (x, y) on-sky position of this object.
        """
        return self.position
    
    
    def get_spectrum(self : Source) -> Array:
        """
        Getter method for the spectrum.
        
        Returns
        -------
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        """
        return self.spectrum.get_spectrum()
    
    
    def get_wavelengths(self : Source) -> Array:
        """
        Getter method for the source internal wavelengths.
        
        Returns
        -------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        return self.spectrum.get_wavelengths()
    
    
    def get_weights(self : Source) -> Array:
        """
        Getter method for the source internal weights.
        
        Returns
        -------
        weights : Array
            The relative weights of each wavelength.
        """
        return self.spectrum.get_weights()
    
    
    def is_resolved(self : Source) -> bool:
        """
        Getter method for the resolved parameter.
        
        Returns
        -------
        resolved : bool
            Is this a point source or a resolved source.
        """
        return self.resolved
    ### End Getter Methods ###
    
    
    ### Correctly Formatted Outputs ###
    def _get_wavelengths(self : Source) -> Array:
        """
        Getter method for the source internal wavelengths, formatted correctly
        for the `scene.decompose()` method.
        
        Returns
        -------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        return self.spectrum._get_wavelengths()
    
    
    def _get_weights(self : Source) -> Array:
        """
        Getter method for the source internal weights, formatted correctly for
        the `scene.decompose()` method.
        
        Returns
        -------
        weights : Array
            The relative weights of each wavelength.
        """
        return self.spectrum._get_weights()
    
    
    def _get_flux(self : Source) -> Array:
        """
        Getter method for the flux, formatted correctly for the
        `scene.decompose()` method.
        
        Returns
        -------
        flux : Array, photons
            The flux of the object.
        """
        return np.array([self.flux])
    
    
    def _get_position(self : Source) -> Array:
        """
        Getter method for the position, formatted correctly for the 
        `scene.decompose()` method.
        
        Returns
        -------
        position : Array, radians
            The (x, y) on-sky position of this object.
        """
        nwavels = self.get_wavelengths().shape[-1]
        return np.array([np.tile(self.get_position(), (nwavels, 1))])
        # return np.array([self.get_position()])
    
    
    def _is_resolved(self : Source) -> bool:
        """
        Getter method for the resolved parameter, formatted correctly for the
        `scene.decompose()` method.
        
        Returns
        -------
        resolved : bool
            Is this a point source or a resolved source.
        """
        nwavels = self.get_wavelengths().shape[-1]
        return np.tile(self.is_resolved(), (1, nwavels))
    
    
    ### Start Setter Methods ###
    def set_flux(self : Source, flux : Array) -> Source:
        """
        Setter method for the flux.
        
        Parameters
        ----------
        flux : Array, photons
            The flux of the object.
        
        Returns
        -------
        source : Source
            The source object with the updated flux parameter.
        """
        return eqx.tree_at(
            lambda source : source.flux, self, flux)
    
    
    def set_position(self : Source, position : Array) -> Source:
        """
        Setter method for the position.
        
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object.
        
        Returns
        -------
        source : Source
            The source object with the updated position parameter.
        """
        return eqx.tree_at(
            lambda source : source.position, self, position)
    
    
    def set_spectrum(self : Source, spectrum : Spectrum) -> Source:
        """
        Setter method for the specturm.
        
        Parameters
        ----------
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        
        Returns
        -------
        source : Source
            The source object with the updated spectrum.
        """
        return eqx.tree_at(
            lambda source : source.spectrum, self, spectrum)
    
    
    def set_resolved(self : Source, resolved : bool) -> Source:
        """
        Setter method for the resolved parameter.
        
        Parameters
        ----------
        resolved : bool
            Is this a point source or a resolved source.
        
        Returns
        -------
        source : Source
            The source object with the updated resolved parameter.
        """
        return eqx.tree_at(
            lambda source : source.resolved, self, resolved)
    ### End Setter Methods ###
    
    
    def normalise(self : Source) -> Source:
        """
        Method for returning a new source object with a normalised total
        spectrum.
        
        Returns
        -------
        source : Source
            The soource object with the normalised spectrum.
        """
        normalised_spectrum = self.spectrum.normalise()
        return eqx.tree_at(
            lambda source : source.spectrum, self, normalised_spectrum)
    
    
class PointSource(Source):
    """
    Concrete Class for unresolved point source objects. Essentially just passes
    the resolved boolean value to Source.
    """
    
    
    def __init__(self     : Source,
                 position : Array,
                 flux     : Array,
                 spectrum : Spectrum,
                 name     : str = 'Source') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        name : str (optional)
            The name for this object. Defaults to 'Source'
        """
        super().__init__(position, flux, spectrum, False, name=name)
        
        
class ResolvedSource(Source, abc.ABC):
    """
    Source
        ResolvedSource (abstract)
    
    Abstract Class for resolved source objects
    
    Essentially just passes the resolved boolean value to Source
    """
    
    def __init__(self : Source, 
                 position : Array,
                 flux     : Array,
                 spectrum : Spectrum,
                 name     : str = 'Source') -> Source:
        
        super().__init__(position, flux, spectrum, True, name=name)
        
    # TODO: Implement this, maybe do some clever array shape checking
    # in order to always implement the most efficient convolution type
    def convolve_source(self : Source, psf : Array) -> Array:
        """
        Possible have this exist as an external function that this 
        """
        pass
        
        
class GaussianSource(ResolvedSource):
    """
    Source
        ResolvedSource
            GaussianSource
    
    Concrete class for sources with gaussian flux distribution
    Assumed rotationally symmetric, ie parametrised by a single sigma value
    
    Sigma here can have physical 'on sky' units, allowing the convolution 
    to take place with a gaussian kernel sampled at the same spatial resoltuion
    as the psf at any given time
    
    """
    sigma : Array
    
    def __init__(self : Source, 
                 position : Array,
                 flux     : Array,
                 spectrum : Spectrum,
                 sigma    : Array,
                 name     : str = 'Source') -> Source:
        
        super().__init__(position, flux, spectrum, name=name)
        self.sigma = np.asarray(sigma, dtype=float)
        
    ## TODO: Implement the generation of the convolution
    ## Theres a few upstream qs to answer so I leave this for now
    ## Algorithmically this should be identical to ApplyJitter
    def generate_kernel(self : Source) -> Array:
        """
        
        """
        pass
    
    
class ExtendedSource(ResolvedSource):
    """
    Source
        ResolvedSource
            ExtendedSource
    
    Concrete class for sources with non-parametric (pixel based) on sky
    flux distribution
    
    This class does not adress normalisations yet - Similar to Spectrum
    I believe this should be added as a class method and called before
    being used in a convolution
    
    Basic idea is that there is some underlying source distribution defined
    on a pixel grid, with some defined pixel scale. A point sounce PSF is 
    modelled at the given RA and DEC and the resluting PSF is convolved
    with this distribution. 
    
    The distribution may need to be interpolated over in order to perform 
    the convolution with the PSF at the same sampling.
    
    Should pixel_scale be a jax type? Im really not sure. No for now, can 
    always be changed trivially
    """
    disribution : Array
    pixel_scale : float # Python type non optimisible, maybe change later
    
    def __init__(self         : Source, 
                 position     : Array,
                 flux         : Array,
                 spectrum     : Spectrum,
                 distribution : Array,
                 pixel_scale  : Array,
                 name         : str = 'Source') -> Source:
        
        super().__init__(RA, DEC, flux, spectrum, name=name)
        self.distribution = np.asarray(distribution, dtype=float)
        self.pixel_scale = float(pixel_scale)
        
    ## TODO: Figure out what else we need to implement this fully
    
    
class BinarySource(Source):
    """
    A parameterised binary source. Currently only supports two point-sources
    
    Attributes
    ----------
    separation : Array, radians
        The separation of the two sources in radians.
    field_angle : Array, radians
        The field angle between the two sources measure from the positive
        x axis.
    flux_ratio : Array
        The contrast ratio between the two sources.
    resolved : Array
        An array of booleans denoting if each source is resolved.
    """
    separation  : Array
    field_angle : Array
    flux_ratio  : Array
    resolved    : Array
    
    
    def __init__(self        : Source,
                 position    : Array,
                 flux        : Array,
                 separation  : Array,
                 field_angle : Array,
                 flux_ratio  : Array,
                 spectrum    : Spectrum, # Converted into dict
                 resolved    : list, # Converted into dict
                 name        : str = 'Binary Source'
                 ) -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        separation : Array, radians
            The separation of the two sources in radians.
        field_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        flux_ratio : Array
            The contrast ratio between the two sources.
        resolved : list
            A list of booleans denoting if each source is resolved.
        name : str (optional)
            The name for this object. Defaults to 'Binary Source'
        """
        super().__init__(position, flux, spectrum, None, name=name)
        
        self.separation  = np.asarray(separation, dtype=float)
        self.field_angle = np.asarray(field_angle, dtype=float)
        self.flux_ratio  = np.asarray(flux_ratio, dtype=float)
        
        assert len(resolved) == 2, "Resolved list must contain exactly two \
        values"
        
        self.resolved = np.asarray(resolved, dtype=bool)
    
    
    ### Start Getter Methods ###
    def get_separation(self : Source) -> Array:
        """
        Getter method for the source separation.
        
        Returns
        -------
        separation : Array, radians
            The separation of the two sources in radians.
        """
        return self.separation
    
    
    def get_field_angle(self : Source) -> Array:
        """
        Getter method for the source field angle.
        
        Returns
        -------
        field_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        """
        return self.field_angle
    
    
    def get_flux_ratio(self : Source) -> Array:
        """
        Getter method for the source contrast ratio.
        
        Returns
        -------
        flux_ratio : Array
            The contrast ratio between the two sources.
        """
        return self.flux_ratio
    
    
    def get_flux(self : Source) -> Array:
        """
        Getter method for the fluxes.
        
        Returns
        -------
        flux : Array, photons
            The flux (flux1, flux2) of the binary object.
        """
        flux_A = 2 * self.get_flux_ratio() * super().get_flux() / (1 + self.get_flux_ratio())
        flux_B = 2 * super().get_flux() / (1 + self.get_flux_ratio())
        return np.array([flux_A, flux_B])
    
    
    def get_position(self : Source) -> Array:
        """
        Getter method for the position.
        
        Returns
        -------
        position : Array, radians
            The ((x, y), (x, y)) on-sky position of this object.
        """
        sep_vec = dLux.utils.polar2cart(self.get_separation()/2, 
                                        self.get_field_angle())
        return np.array([super().get_position() + sep_vec, 
                         super().get_position() - sep_vec])
    ### End Getter Methods ###
    
    
    ### Correctly Formatted Outputs ###
    def _get_wavelengths(self : Source) -> Array:
        """
        Getter method for the source internal wavelengths, formatted correctly
        for the `scene.decompose()` method.
        
        Returns
        -------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        return self.spectrum._get_wavelengths()
    
    
    def _get_weights(self : Source) -> Array:
        """
        Getter method for the source internal weights, formatted correctly for
        the `scene.decompose()` method.
        
        Returns
        -------
        weights : Array
            The relative weights of each wavelength.
        """
        return self.spectrum._get_weights()
    
    
    def _get_flux(self : Source) -> Array:
        """
        Getter method for the flux, formatted correctly for the
        `scene.decompose()` method.
        
        Returns
        -------
        flux : Array, photons
            The flux of the object.
        """
        # return self.get_flux()
        return np.expand_dims(self.get_flux(), -1)
    
    
    def _get_position(self : Source) -> Array:
        """
        Getter method for the position, formatted correctly for the 
        `scene.decompose()` method.
        
        Returns
        -------
        position : Array, radians
            The (x, y) on-sky position of this object.
        """
        position_A, position_B = self.get_position()
        nwavels = self.get_wavelengths().shape[-1]
        return np.array([np.tile(position_A, (nwavels, 1)), 
                         np.tile(position_B, (nwavels, 1))])
    
    
    def _is_resolved(self : Source) -> bool:
        """
        Getter method for the resolved parameter, formatted correctly for the
        `scene.decompose()` method.
        
        Returns
        -------
        resolved : bool
            Is this a point source or a resolved source.
        """
        nwavels = self.get_wavelengths().shape[-1]
        resolved_A, resolved_B = self.is_resolved()
        return np.array([np.tile(resolved_A, (nwavels)), 
                         np.tile(resolved_B, (nwavels))])
    
    
    ### Start Setter Methods ###
    def set_separation(self : Source) -> Source:
        """
        Setter method for the source separation.
        
        Parameters
        ----------
        separation : Array, radians
            The separation of the two sources in radians.
        
        Returns
        -------
        source : Source
            The source object with updated separation.
        """
        return eqx.tree_at(
           lambda source: source.separation, self, separation)
    
    
    def set_field_angle(self : Source) -> Source:
        """
        Setter method for the source field angle.
        
        Parameters
        ----------
        field_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        
        Returns
        -------
        source : Source
            The source object with updated field angle.
        """
        return eqx.tree_at(
           lambda source: source.field_angle, self, field_angle)
    
    def set_flux_ratio(self : Source) -> Source:
        """
        Setter method for the source flux ratio.
        
        Parameters
        ----------
        flux_ratio : Array
            The contrast ratio between the two sources.
        
        Returns
        -------
        source : Source
            The source object with updated flux ratio.
        """
        return eqx.tree_at(
           lambda source: source.flux_ratio, self, flux_ratio)
    ### End Setter Methods ###