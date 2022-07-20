import dLux
import typing
import jax 
import jax.numpy as np
import equinox as eqx


Propagator = typing.TypeVar("Propagator")
Wavefront = typing.TypeVar("Wavefront")
Matrix = typing.TypeVar("Matrix")
Tensor = typing.TypeVar("Tensor")
Vector = typing.TypeVar("Vector")


class GaussianWavefront(dLux.Wavefront):
    angular : bool
    spherical : bool
    waist_radius : float 
    position : float
    waist_position : float
    rayleigh_factor : float
    focal_length : float
    
 
    def __init__(self : Wavefront,
            offset : Vector,
            wavelength : float,
            beam_radius : float,
            rayleigh_factor : float = 2.) -> Wavefront:
        super(GaussianWavefront, self).__init__(wavelength, offset)
        self.waist_radius = np.asarray(beam_radius).astype(float)  
        self.position = np.asarray(0.).astype(float)
        self.waist_position = np.asarray(0.).astype(float)
        self.rayleigh_factor = np.asarray(rayleigh_factor).astype(float)
        self.focal_length = np.inf 
        self.angular = False
        self.spherical = False


    # NOTE: This also needs an ..._after name. I could use something
    # like quadratic_phase_after() or phase_after() 
    def quadratic_phase(self : Wavefront, distance : float) -> Matrix:
        positions = self.get_pixel_positions()
        x, y = positions[0], positions[1]
        return np.exp(1.j * np.pi * (x ** 2 + y ** 2) \
                / distance / self.get_wavelength())


    # NOTE: This is plane_to_plane transfer function. I should give it
    # a better name like fraunhofer_phase_after()
    def transfer(self : Wavefront, distance : float) -> Matrix:
        positions = self.get_pixel_positions()
        x, y = positions[0], positions[1]
        rho_squared = \
            (x / (self.pixel_scale ** 2 \
                * self.number_of_pixels())) ** 2 + \
            (y / (self.pixel_scale ** 2 \
                * self.number_of_pixels())) ** 2
        # Transfer Function of diffraction propagation eq. 22, eq. 87
        return np.exp(1.j * np.pi * self.wavelength * \
                distance * rho_squared)


    def rayleigh_distance(self : Wavefront) -> float:
        return np.pi * self.waist_radius ** 2 / self.wavelength


    # NOTE: The pixel scale cannot be set when self.angular == True
    # NOTE: This has the correct units always/
    def get_pixel_scale(self : Wavefront):
        return jax.lax.cond(self.angular,
            lambda : self.pixel_scale / self.focal_length,
            lambda : self.pixel_scale)


    # NOTE: Should only be called when self.angular == True
    def field_of_view(self):
        return self.number_of_pixels() * self.get_pixel_scale()


    # NOTE: naming convention. ..._at indicates absolute position
    # ..._after indicates a distance from current position. 
    # either should make all the same or be clear. 
    def curvature_at(self : Wavefront, position : float) -> float:
        relative_position = position - self.waist_position
        return relative_position + \
            self.rayleigh_distance() ** 2 / relative_position


    def radius_at(self : Wavefront, position : float) -> float:
        relative_position = position - self.waist_position
        return self.waist_radius * \
            np.sqrt(1.0 + \
                (relative_position / self.rayleigh_distance()) ** 2)
           
 
    def is_planar_after(self : Wavefront, distance : float) -> bool:
        return np.abs(distance) < self.rayleigh_distance()

    # NOTE: Also updates, so I want better names for these rather than 
    # after. 
    # NOTE: This is only for transitions from planar to spherical 
    # or vice versa so it needs a much better name than current. 
    def pixel_scale_after(self : Wavefront, distance : float) -> Wavefront:
        pixel_scale = self.get_wavelength() * np.abs(distance) /\
            (self.number_of_pixels() * self.get_pixel_scale())
        return eqx.tree_at(lambda wave : wave.pixel_scale,
            self, pixel_scale, is_leaf = lambda leaf : leaf is None)


    def position_after(self : Wavefront, distance : float) -> Wavefront:
        position = self.position + distance
        return eqx.tree_at(lambda wave : wave.position, self, position)

    
    # NOTE: ordering convention: dunder, _..., ..._at, ..._after, 
    # set_... get_...
    # NOTE: naming convention: position -> absolute place on optical
    # axis and distance -> movement.
    # def set_waist_position(self : Wavefront, waist_position : float) -> Wavefront:
    # def set_waist_radius(self : Wavefront, waist_radius : float) -> Wavefront:
    def set_spherical(self : Wavefront, spherical : bool) -> Wavefront:
        return eqx.tree_at(lambda wave : wave.spherical, self, spherical)
    #def set_angular(self : Wavefront, angular : bool) -> Wavefront:
    # NOTE: focal_length will probably not stay as an attribute of the 
    # wavefront but will be upgraded to an optical element attribute.
    #def set_focal_length(self : Wavefront, focal_length : float) -> Wavefront:


    def get_pixel_positions(self : Wavefront) -> Tensor:
        pixels = self.phase.shape[0]
        positions = np.array(np.indices((pixels, pixels)) - pixels / 2.)
        return self.pixel_scale * positions


class GaussianPropagator(eqx.Module):
    distance : float

    
    def __init__(self : Propagator, distance : float):
        self.distance = np.asarray(distance).astype(float)


    def _fourier_transform(self : Propagator, field : Matrix) -> Matrix:
        # return np.fft.ifft2(field)
        return 1 / field.shape[0] * np.fft.fftshift(np.fft.ifft2(field))


    def _inverse_fourier_transform(self : Propagator, field : Matrix) -> Matrix:
        # return np.fft.fft2(field)
        return field.shape[0] * np.fft.fft2(np.fft.ifftshift(field))


    # NOTE: need to add in the standard FFT normalising factor
    # as in the propagator. 
    def _propagate(self : Propagator, field : Matrix, 
            distance : float) -> Matrix:
        # NOTE: is this diagnosable directly from the stored parameter
        # would be nice if the "transfer" function could be automatically
        # chosen from the "spherical" and one other thing. 
        # should probably avoid logic overcrowding
        return jax.lax.cond(distance > 0,
            lambda : self._fourier_transform(field),
            lambda : self._inverse_fourier_transform(field))
            

    # NOTE: Wavefront must be planar 
    # NOTE: Uses eq. 82, 86, 87
    def _plane_to_plane(self : Propagator, wavefront : Wavefront,
            distance : float):
        # NOTE: Seriously need to change the name to get_field()
        field = self._fourier_transform(wavefront.get_complex_form())
        field *= wavefront.transfer(distance)  # eq. 6.68
        field = self._inverse_fourier_transform(field)
        # NOTE: wavefront.from_field is looking good right about now
        return wavefront\
            .position_after(distance)\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))
 

    # NOTE: I'm thinking that the logic for repacking the wavefront
    # should occur somewhere else although I guess that it can't really
    # NOTE: Must start with a planar wavefront
    def _waist_to_spherical(self : Propagator, wavefront : Wavefront, 
            distance : float) -> Wavefront:
        # Lawrence eq. 83,88
        field = wavefront.get_complex_form()
        field *= np.fft.fftshift(wavefront.quadratic_phase(distance))

        # SIGN CONVENTION: forward optical propagations want a positive sign in the complex exponential, which
        # numpy implements as an "inverse" FFT
        # NOTE: This all needs to be contained within a _propagate method
        field = self._propagate(field, distance)
        # NOTE: future release should look like 
        # TODO: rename wavefront -> wave internally for brevity
        # field = self._propagate(wave.quadratic_phase() * wave.field())
        # TODO: wavefront.number_of_pixels -> wavefront.pixels()
        # TODO: calculation for pixel_scale should live in wavefront
        # as a function. 
        # TODO: Implement an wavefront.move(distance)
        # TODO: The wavefront.move(distance) would automatically update
        # the pixel scale ... this wuld require logic so never mind.
        pixel_scale = wavefront.pixel_scale_after(distance) 
        return wavefront\
            .pixel_scale_after(distance)\
            .position_after(distance)\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))\
            .set_spherical(True)       


    # Wavefront.spherical must be True initially
    def _spherical_to_waist(self : Propagator, wavefront : Wavefront,
            distance : float) -> Wavefront:
        # Lawrence eq. 89
        field = self._propagate(wavefront.get_complex_form(), distance)
        field *= np.fft.fftshift(wavefront.quadratic_phase(distance))
        return wavefront\
            .set_phase(np.angle(field))\
            .set_amplitude(np.abs(field))\
            .set_spherical(True)\
            .pixel_scale_after(distance)\
            .position_after(distance)


    def _inside_to_inside(self : Propagator, wave : Wavefront) -> Wavefront:
         return self._plane_to_plane(wave, self.distance)


    def _inside_to_outside(self : Propagator, wave : Wavefront) -> Wavefront: 
        start = wave.position
        end = wave.position + self.distance
        wave = self._plane_to_plane(wave, wave.waist_position - start)
        wave = self._waist_to_spherical(wave, end - wave.waist_position)
        return wave


    def _outside_to_inside(self : Propagator, wave : Wavefront) -> Wavefront:
        start = wave.position
        end = wave.position + self.distance
        wave = self._spherical_to_waist(wave, wave.waist_position - start)
        wave = self._plane_to_plane(wave, end - wave.waist_position)
        return wave


    def _outside_to_outside(self : Propagator, wave : Wavefront) -> Wavefront:
        start = wave.position
        end = wave.position + self.distance
        wave = self._spherical_to_waist(wave, wave.waist_position - start)
        wave = self._waist_to_spherical(wave, end - wave.waist_position)
        return wave


    # NOTE: So I could attempt to move all of the functionality into 
    # the wavefront class and do very little here. Damn, I need to 
    # fit it into the overall architecture. 
    # TODO: Implement the oversample in the fixed sampling propagator
    # Coordiantes must be in meters for the propagator
    def __call__(self : Propagator, wave : Wavefront) -> Wavefront:
        # NOTE: need to understand this mystery. 
        # field = np.fft.fftshift(wave.get_complex_form())
        # wave = wave.update_phasor(np.abs(field), np.angle(field))

        wave = jax.lax.switch(
            2 * wave.spherical + wave.is_planar_after(self.distance),
            [self._outside_to_inside, self._outside_to_outside,
            self._inside_to_inside, self._inside_to_outside], wave) 

        # field = np.fft.fftshift(wave.get_complex_form())
        # wave = wave.update_phasor(np.abs(field), np.angle(field))
        return wave


    def __imul__(self, optic):
        """Multiply a Wavefront by an OpticalElement or scalar"""
        if isinstance(optic, QuadraticLens):
            # Special case: if we have a lens, call the routine for that,
            # which will modify the properties of this wavefront more fundamentally
            # than most other optics, adjusting beam parameters and so forth
            self.apply_lens_power(optic)
            return self
        elif isinstance(optic, FixedSamplingImagePlaneElement):
            # Special case: if we have an FPM, call the routine for that,
            # which will apply an amplitude transmission to the wavefront. 
            self.apply_image_plane_fftmft(optic)
            return self
        else:
            # Otherwise fall back to the parent class
            return super(FresnelWavefront, self).__imul__(optic)


class GaussianLens(eqx.Module):
    focal_length : float

    def __init__(self : Layer, focal_length : float) -> Layer:
        self.focal_length = np.asarray(focal_length).astype(float)


    def __call__(self : Layer, wave : Wavefront) -> Wavefront:
        # calculate beam radius at current surface
        radius = wave.radius()

        # Is the incident beam planar or spherical?
        # We decided based on whether the last waist is outside the rayleigh distance.
        #  I.e. here we neglect small curvature just away from the waist
        # Based on that, determine the radius of curvature of the output beam
        from_waist_displacement = np.abs(wave.waist_position - wave.position)
        from_waist_threshold = wave.rayleigh_factor * wave.rayleigh_distance()
        spherical = from_waist_displacement > from_waist_threshold

        # NOTE: True case
        # So these are just the curvatures. The first one assumes a sphere 
        # centred at the waist_location. 
        # NOTE: False case
        # we are at a focus or pupil so the new optic is the only curvature
        in_curve, out_curve = jax.lax.cond(spherical,
            lambda : (from_waist_displacement,
                1. / (1. / wave.curvature() - 1. / self.focal_length)),
            lambda : (np.inf, -1. * self.focal_length))

        # NOTE: TODO: From here
        # update the wavefront parameters to the post-lens beam waist
        if self.r_c() == optic.fl:
            self.z_w0 = self.z
            self.w_0 = spot_radius
        else:
            self.z_w0 = -r_output_beam / (
                1.0 + (self.wavelength * r_output_beam / (np.pi * spot_radius ** 2)) ** 2) + self.z
            self.w_0 = spot_radius / np.sqrt(1.0 + (np.pi * spot_radius ** 2 / (self.wavelength * r_output_beam)) ** 2)

        # Update the focal length of the beam. This is closely related to but tracked separately from
        # the beam waist and radius of curvature; we keep track of it to use in optional conversion
        # of coordinates to angular units.
        if not np.isfinite(self.focal_length):
            self.focal_length = 1 * optic.fl
        else:
            # determine magnification as the change in curvature of this optic
            mag = r_output_beam / r_input_beam
            self.focal_length *= mag
        
        # Now we need to figure out the phase term to apply to the wavefront
        # data array
        if not self.spherical:
            if np.abs(self.z_w0 - self.z) < self.z_r:
                z_eff = 1 * optic.fl

            else:
                # find the radius of curvature of the lens output beam
                # curvatures are multiplicative exponentials
                # e^(1/z) = e^(1/x)*e^(1/y) = e^(1/x+1/y) -> 1/z = 1/x + 1/y
                # z = 1/(1/x+1/y) = xy/x+y
                z_eff = 1.0 / (1.0 / optic.fl + 1.0 / (self.z - self.z_w0))
                self.spherical = True

        else:  # spherical input wavefront
            if np.abs(self.z_w0 - self.z) > self.z_r:
                if (self.z - self.z_w0) == 0:
                    z_eff = 1.0 / (1.0 / optic.fl + 1.0 / (self.z - self.z_w0))
                else:
                    z_eff = 1.0 / (1.0 / optic.fl + 1.0 / (self.z - self.z_w0) - 1.0 / r_input_beam)

            else:
                z_eff = 1.0 / (1.0 / optic.fl - 1.0 / r_input_beam)
                self.spherical = False

        # Apply phase to the wavefront array
        effective_optic = QuadPhase(-z_eff, name=optic.name)
        self *= effective_optic


#    def apply_image_plane_fftmft(self, optic):
#        """
#        Apply a focal plane mask using fft and mft methods to highly sample at the focal plane.
#        
#        Parameters
#        ----------
#        optic : FixedSamplingImagePlaneElement
#        """
#        _log.debug("------ Applying FixedSamplingImagePlaneElement using FFT and MFT sequence ------")
#        
#        # readjust pixelscale to wavelength being propagated
#        fpm_pxscl_lamD = ( optic.pixelscale_lamD * optic.wavelength_c.to(u.meter) / self.wavelength.to(u.meter) ).value 
#
#        # get the fpm phasor either using numexpr or numpy
#        scale = 2. * np.pi / self.wavelength.to(u.meter).value
#        if accel_math._USE_NUMEXPR:
#            _log.debug("Calculating FPM phasor from numexpr.")
#            trans = optic.get_transmission(self)
#            opd = optic.get_opd(self)
#            fpm_phasor = ne.evaluate("trans * exp(1.j * opd * scale)")
#        else:
#            _log.debug("numexpr not available, calculating FPM phasor with numpy.")
#            fpm_phasor = optic.get_transmission(self) * np.exp(1.j * optic.get_opd(self) * scale)
#        
#        nfpm = fpm_phasor.shape[0]
#        n = self.wavefront.shape[0]
#        
#        nfpmlamD = nfpm*fpm_pxscl_lamD*self.oversample
#
#        mft = poppy.matrixDFT.MatrixFourierTransform(centering=optic.centering)
#
#        self.wavefront = accel_math._ifftshift(self.wavefront)
#        self.wavefront = accel_math.fft_2d(self.wavefront, forward=False, fftshift=True) # do a forward FFT to virtual pupil
#        self.wavefront = mft.perform(self.wavefront, nfpmlamD, nfpm) # MFT back to highly sampled focal plane
#        self.wavefront *= fpm_phasor
#        self.wavefront = mft.inverse(self.wavefront, nfpmlamD, n) # MFT to virtual pupil
#        self.wavefront = accel_math.fft_2d(self.wavefront, forward=True, fftshift=True) # FFT back to normally-sampled focal plane
#        self.wavefront = accel_math._fftshift(self.wavefront)
#        
#        _log.debug("------ FixedSamplingImagePlaneElement: " + str(optic.name) + " applied ------")
#
#
#    def _resample_wavefront_pixelscale(self, detector):
#        """ Resample a Fresnel wavefront to a desired detector sampling.
#        The interpolation is done via the scipy.ndimage.zoom function, by default
#        using cubic interpolation.  If you wish a different order of interpolation,
#        set the `.interp_order` attribute of the detector instance.
#        Parameters
#        ----------
#        detector : Detector class instance
#            Detector that defines the desired pixel scale
#        Returns
#        -------
#        The wavefront object is modified to have the appropriate pixel scale and spatial extent.
#        """
#
#        if self.angular:
#            raise NotImplementedError("Resampling to detector doesn't yet work in angular coordinates for Fresnel.")
#
#        pixscale_ratio = (self.pixelscale / detector.pixelscale).decompose().value
#
#        if np.abs(pixscale_ratio - 1.0) < 1e-3:
#            _log.debug("Wavefront is already at desired pixel scale "
#                       "{:.4g}.  No resampling needed.".format(self.pixelscale))
#            self.wavefront = utils.pad_or_crop_to_shape(self.wavefront, detector.shape)
#            return
#
#        super(FresnelWavefront, self)._resample_wavefront_pixelscale(detector)
#
#        self.n = detector.shape[0]

