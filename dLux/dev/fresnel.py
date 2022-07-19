import dLux
import typing
import jax 
import jax.numpy as np
import equinox as eqx

Wavefront = typing.TypeVar("Wavefront")
Array = typing.TypeVar("Array")
Matrix = typing.TypeVar("Matrix")
Propagator = typing.TypeVar("Propagator")

class GaussianWavefront(dLux.Wavefront):
    """
    Expresses the behaviour and state of a wavefront propagating in 
    an optical system under the fresnel assumptions. This 
    implementation is based on the same class from the `poppy` 
    library [poppy](https://github.com/spacetelescope/poppy/fresnel.py)
    and Chapter 3 from _Applied Optics and Optical Engineering_
    by Lawrence G. N.


    Approximates the wavefront as a Gaussian Beam parameterised by the 
    radius of the beam, the phase radius, the phase factor and the 
    Rayleigh distance. Propagation is based on two different regimes 
    for a total of four different opertations. 
    
    Attributes
    ----------
    position : float, meters
        The position of the wavefront in the optical system.
    beam_radius : float, meters
        The radius of the beam. 
    phase_radius : float, unitless
        The phase radius of the gaussian beam.
    location_of_waist : float, meters
        The position of the beam waist along the optical axis. 
    """
    position : float 
    beam_waist_radius : float
    location_of_waist : float


    def __init__(self : Wavefront, 
            offset : Array,
            wavelength : float,
            beam_radius : float,
            position : float = 0.) -> Wavefront:
        """
        Creates a wavefront with an empty amplitude and phase 
        arrays but of a given wavelength and phase offset. 
        Assumes that the beam starts at the waist following from 
        the `poppy` convention.

        Parameters
        ----------
        beam_radius : float, meters
            Radius of the beam at the initial optical plane.
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        offset : Array, radians
            The (x, y) angular offset of the `Wavefront` from 
            the optical axis.
        """
        super().__init__(wavelength, offset)
        self.beam_waist_radius = np.asarray(beam_radius).astype(float)
        self.position = np.asarray(position).astype(float)
        self.location_of_waist = self.position
    def set_position(self : Wavefront, 
            position : float) -> Wavefront:
        """
        Mutator for the position of the wavefront. Changes the 
        pixel_scale which is a function of the position.  

        Parameters
        ----------
        position : float
            The new position of the wavefront from its starting point 
            assumed to be in meters. 
        
        Returns
        -------
        wavefront : Wavefront
            This wavefront at the new position. 
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.position, self, position,
            is_leaf = lambda leaf : leaf is None)
    

    def rayleigh_distance(self: Wavefront) -> float:
        """
        Calculates the rayleigh distance of the Gaussian beam.
        
        Returns
        -------
        rayleigh_distance : float
            The Rayleigh distance of the wavefront in metres.
        """
        return np.pi * self.get_beam_waist_radius() ** 2\
            / self.get_wavelength()


    def transfer_function(self: Wavefront, distance: float) -> Array:
        """
        The optical transfer function (OTF) for the gaussian beam.
        Assumes propagation is along the axis. 

        Parameters
        ----------
        distance : float
            The distance to propagate the wavefront along the beam 
            via the optical transfer function in metres.

        Returns
        -------
        phase : float 
            A phase representing the evolution of the wavefront over 
            the distance. 

        References
        ----------
        Wikipedia contributors. (2022, January 3). Direction cosine. 
        In Wikipedia, The Free Encyclopedia. June 23, 2022, from 
        https://en.wikipedia.org/wiki/Direction_cosine

        Wikipedia contributors. (2022, January 3). Spatial frequecy.
        In Wikipedia, The Free Encyclopedia. June 23, 2022, from 
        https://en.wikipedia.org/wiki/Spatial_frequency
        """
        coordinates = self.get_pixel_positions()
        radius = np.sqrt((coordinates ** 2).sum(axis=0))
        xi = coordinates[0, :, :] / radius / self.get_wavelength()
        eta = coordinates[1, :, :] / radius / self.get_wavelength()
        return np.exp(1j * np.pi * self.get_wavelength() \
            * distance * (xi ** 2 + eta ** 2))


    def quadratic_phase_factor(self: Wavefront, 
            distance: float) -> float:
        """
        Convinience function that simplifies many of the diffraction
        equations. Caclulates a quadratic phase factor associated with 
        the beam. 

        Parameters
        ----------
        distance : float
            The distance of the propagation measured in metres. 

        Returns
        -------
        phase : float
            The near-field quadratic phase accumulated by the beam
            from a propagation of distance.
        """      
        return np.exp(1j * np.pi * \
            (self.get_pixel_positions() ** 2).sum(axis=0) \
            / self.get_wavelength() / distance)


    # NOTE: This should only be updated after passing through 
    # an optic like a quadratic lens. 
    def get_location_of_waist(self: Wavefront) -> float:
        """
        Calculates the position of the waist along the direction of 
        propagation based of the current state of the wave. This should 
        only be called after passing through a lens. 

        Returns
        -------
        waist : float
            The position of the waist in metres.
        """
        return - self.get_phase_radius() / \
            (1 + (self.get_phase_radius() / \
            self.rayleigh_distance()) ** 2)


    def get_beam_radius(self : Wavefront) -> float:
        """
        Calculate the radius of the beam at the current coordinate 
        of this wavefront. 

        Returns
        -------
        waist_radius : float, meters
            The waist radius at the current position along the 
            optical axis.
        """
        # TODO: Implement this. 
        return self.get_beam_waist_radius() * np.sqrt(
            1 + (self.position / self.rayleigh_distance()) ** 2)


    # TODO: Determine where this fits into poppy. This implements 
    # (56) from the Lawrence book. 
    def get_beam_waist_radius(self: Wavefront) -> float:
        """
        The radius of the beam at the waist.

        Returns
        -------
        waist_radius : float
            The radius of the beam at the waist in metres.
        """
        return self.get_beam_radius() / \
            np.sqrt(1 + (self.rayleigh_distance() \
                / self.get_beam_radius()) ** 2)


    # Confirm the behaviour is correct. Maps accros to r_c in the 
    # poppy code. 
    def get_phase_radius(self: Wavefront) -> float:
        """
        Calculate the phase radius of the wavefront at its current 
        position. 

        Returns
        -------
        phase_radius : float, radians
            The phase radius of the beam at its current position.
        """
        z = self.position - self.location_of_waist
        return z + self.rayleigh_distance() ** 2 / z

        

    def is_inside(self: Wavefront, distance: float) -> bool:
        """ 
        Determines whether a point at along the axis of propagation 
        at distance away from the current position is inside the 
        rayleigh distance. 

        Parameters
        ----------
        distance : float
            The distance to test in metres.

        Returns
        -------
        inside : bool
            true if the point is within the rayleigh distance false 
            otherwise.
        """
        return np.abs(self.position + distance - \
            self.location_of_waist) <= self.rayleigh_distance()


    def set_beam_waist_radius(self : Wavefront, 
            beam_radius : float) -> Wavefront:
        """
        Mutator for the `beam_radius`.

        Parameters
        ----------
        beam_radius : float
            The new beam_radius in meters.

        Returns
        -------
        wavefront : Wavefront
            A modified Wavefront with the new beam_radius.
        """
        return eqx.tree_at(
            lambda wavefront : wavefront.beam_waist_radius, self, beam_radius,
            is_leaf = lambda leaf : leaf is None)

 
class GaussianPropagator(dLux.FixedSamplingPropagator):
    """
    An intermediate plane fresnel algorithm for propagating the
    `GaussianWavefront` class between the planes. The propagator 
    is separate from the `Wavefront` to emphasise the similarity 
    of these algorithms to the layers in machine learning algorithms.

    Attributes
    ----------
    distance : float, meters
       The distance to propagate. 
    """
    distance : float 


    def __init__(self : Propagator, 
            distance : float) -> Propagator:
        """
        Constructor for the Propagator.

        Parameters
        ----------
        distance : float, meters
            The distance to propagate the wavefront.
        """
        self.distance = distance
        super().__init__(np.sign(distance) > 0)


    def _propagate(self : Propagator, wavefront : Matrix) -> Matrix: 
        """
        Propagate the wave intelligently aware of the inverse-ness.
    
        Parameters
        ----------
        wavefront : Matrix
            The electric field of the wavefront that is to be
            propagated. 

        Returns
        -------
        wavefront : Matrix
            The wavefront fourier transform but not normalised. 
        """
        field = jax.lax.cond(self.is_inverse(),
            lambda wavefront : \
                self._inverse_fourier_transform(wavefront),
            lambda wavefront : \
                self._fourier_transform(wavefront),
            wavefront)

        return field


    # TODO: Note that the new pixel_scale should be calculated here. 
    def get_pixel_scale_out(self: Wavefront) -> float:
        """
        The pixel scale at the position along the axis of propagation.
        Assumes that the wavefront is square. That is:
        ```
        x, y = self.amplitude.shape
        (x == y) == True
        ```

        Parameters
        ----------
        pixel_scale : float
            The pixel_scale after the propagation.
        """
        number_of_pixels = wavefront.number_of_pixels()
        new_pixel_scale = wavefront.get_wavelength() * np.abs(
            wavefront.position + self.distance) / \
            number_of_pixels / wavefront.get_pixel_scale()  
        return new_pixel_scale 


    def planar_to_planar(self : Propagator, 
            wavefront: Wavefront, distance : float) -> Wavefront:
        """
        Modifies the state of the wavefront by propagating a planar 
        wavefront to a planar wavefront. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. Must be `Wavefront`
            or a subclass. 
        distance : float, meters 
            The distance to propagate the wavefront in this 
            regime. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` propagated by `distance`. 
        """
        field = wavefront.get_complex_form()

        # Should this be propagate but with not inverse 
        # negation.
        new_field = self._inverse_fourier_transform(
            wavefront.transfer_function(distance) * \
            self._fourier_transform(field))

        new_amplitude = np.abs(new_field)
        new_phase = np.angle(new_field)
        
        return wavefront\
            .set_position(wavefront.position + distance)\
            .update_phasor(new_amplitude, new_phase)        


    def waist_to_spherical(self : Propagator, 
            wavefront: Wavefront, distance : float) -> Wavefront:
        """
        Modifies the state of the wavefront by propagating it from 
        the waist of the gaussian beam to a spherical wavefront. 

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` that is getting propagated. Must be either 
            a `Wavefront` or a valid subclass.
        distance : float, meters     
            The distance to propagate the wavefront under this regime. 

        Returns 
        -------
        wavefront : Wavefront
            `wavefront` propgated by `distance`.
        """
        coefficient = 1 / 1j / wavefront.get_wavelength() / distance
        field = wavefront.get_complex_form() 

        # NOTE: normalisation constant exists higher up as well.  
        fourier_transform = \
            wavefront.quadratic_phase_factor(distance) *\
            self._propagate(field)

        new_field = coefficient * fourier_transform
        new_phase = np.angle(new_field)
        new_amplitude = np.abs(new_field)

        return wavefront\
            .update_phasor(new_amplitude, new_phase)\
            .set_position(wavefront.position + distance)


    def spherical_to_waist(self : Propagator, 
            wavefront: Wavefront, distance : float) -> Wavefront:
        """
        Modifies the state of the wavefront by propagating it from 
        a spherical wavefront to the waist of the Gaussian beam. 

        Parameters
        ----------
        wavefront : Wavefront 
            The `Wavefront` that is getting propagated. Must be either 
            a `Wavefront` or a direct subclass.
        distance : float, meters
            The distance to propagate the wavefront under this regime.

        Returns
        -------
        wavefront : Wavefront
            The `wavefront` propagated by distance. 
        """
        coefficient = 1 / 1j / wavefront.get_wavelength() / \
            distance * wavefront.quadratic_phase_factor(distance)

        # NOTE: Uses the standard normalising factor in _propagate
        # this may cause normalisation issues. 
        field = wavefront.get_complex_form()
        fourier_transform = self._propagate(field)

        new_wavefront = coefficient * fourier_transform
        new_phase = np.angle(new_wavefront)
        new_amplitude = np.abs(new_wavefront)

        return wavefront\
            .update_phasor(new_amplitude, new_phase)\
            .set_position(wavefront.position + \
                distance)


    def outside_to_outside(self : Wavefront, 
            wavefront : Wavefront) -> Wavefront:
        """
        Propagation from outside the Rayleigh range to another 
        position outside the Rayleigh range. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. Assumed to be either a 
            `Wavefront` or a direct subclass.
    
        Returns
        -------
        wavefront : Wavefront 
            The new `Wavefront` propgated by distance. 
        """
        from_waist_displacement = wavefront.position \
            + self.distance - wavefront.location_of_waist
        to_waist_displacement = wavefront.location_of_waist \
            - wavefront.position

        wavefront_at_waist = self.spherical_to_waist(
            wavefront, to_waist_displacement)
        wavefront_at_distance = self.spherical_to_waist(
            wavefront_at_waist, from_waist_displacement)

        return wavefront_at_distance


    def outside_to_inside(self : Propagator, 
            wavefront: Wavefront) -> Wavefront:
        """
        Propagation from outside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate. Must be either 
            `Wavefront` or a direct subclass.

        Returns
        -------
        wavefront : Wavefront
            The `Wavefront` propagated by `distance` 
        """
        from_waist_displacement = wavefront.position + \
            self.distance - wavefront.location_of_waist
        to_waist_displacement = wavefront.location_of_waist - \
            wavefront.position

        wavefront_at_waist = self.planar_to_planar(
            wavefront, to_waist_displacement)
        wavefront_at_distance = self.spherical_to_waist(
            wavefront_at_waist, from_waist_displacement)

        return wavefront_at_distance


    def inside_to_inside(self : Propagator, 
            wavefront : Wavefront) -> Wavefront:
        """
        Propagation from inside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` to propagate. This must be either a 
            `Wavefront` or a direct subclass.

        Returns
        -------
        wavefront : Wavefront
            The `Wavefront` propagated by `distance`
        """
        return self.planar_to_planar(wavefront, self.distance)


    def inside_to_outside(self : Propagator, 
            wavefront : Wavefront) -> Wavefront:
        """
        Propagation from inside the Rayleigh range to outside the 
        rayleigh range. 

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` to propgate. Must be either a 
            `Wavefront` or a direct subclass.

        Returns
        -------
        wavefront : Wavefront 
            The `Wavefront` propagated `distance`.
        """
        from_waist_displacement = wavefront.position + \
            self.distance - wavefront.location_of_waist
        to_waist_displacement = wavefront.location_of_waist - \
            wavefront.position

        wavefront_at_waist = self.planar_to_planar(
            wavefront, to_waist_displacement)
        wavefront_at_distance = self.waist_to_spherical(
            wavefront_at_waist, from_waist_displacement)

        return wavefront_at_distance


    def __call__(self : Propagator, parameters : dict) -> dict:
        """
        Propagates the wavefront approximated by a Gaussian beam 
        the amount specified by distance. Note that distance can 
        be negative.

        Parameters 
        ----------
        parameters : dict 
            The `Wavefront` to propagate. Must be a `Wavefront`
            or a direct subclass.

        Returns
        -------
        wavefront : Wavefront 
            The `Wavefront` propagated `distance`.
        """
        wavefront = parameters["Wavefront"]

        # This works by considering the current position and distnace 
        # as a boolean array. The INDEX_GENERATOR converts this to 
        # and index according to the following table.
        #
        # sum((0, 0) * (1, 2)) == 0
        # sum((1, 0) * (1, 2)) == 1
        # sum((0, 1) * (1, 2)) == 2
        # sum((1, 1) * (1, 2)) == 3
        #
        # TODO: Test if a simple lookup is faster. 
        # Constants
        INDEX_GENERATOR = np.array([1, 2])

        decision_vector = wavefront.is_inside(np.array([
            wavefront.position, 
            wavefront.position + self.distance]))
        decision_index = np.sum(
            INDEX_GENERATOR * decision_vector)
 
        # Enters the correct function differentiably depending on 
        # the descision.
        new_wavefront = jax.lax.switch(decision_index, 
            [self.inside_to_inside, self.inside_to_outside,
            self.outside_to_inside, self.outside_to_outside],
            wavefront) 

        parameters["Wavefront"] = new_wavefront
        return parameters
