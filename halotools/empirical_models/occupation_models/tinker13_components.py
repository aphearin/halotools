"""
This module contains occupation components used by the Tinker13 composite model.
"""

import numpy as np
import math
from scipy.special import erf
from astropy.utils.misc import NumpyRNGContext
from copy import deepcopy

from .occupation_model_template import OccupationComponent
from .tinker13_parameter_dictionaries import (quiescent_fraction_control_masses,
    param_dict_z1, param_dict_z2, param_dict_z3)

from .. import model_defaults
from ..assembias_models import HeavisideAssembias

from ...utils.array_utils import custom_len
from ... import sim_manager
from ...custom_exceptions import HalotoolsError

__all__ = ('Tinker13Cens', 'Tinker13QuiescentSats',
           'Tinker13ActiveSats', 'AssembiasTinker13Cens')


def _get_closest_redshift(z):
    if z < 0.48:
        return 'z1'
    elif 0.48 <= z < 0.74:
        return 'z2'
    else:
        return 'z3'


class Tinker13Cens(OccupationComponent):
    """ HOD-style model for central galaxy occupation statistics,
    with behavior deriving from
    two distinct active/quiescent stellar-to-halo-mass relations.

    .. note::

        The `Tinker13Cens` model is part of the ``tinker13``
        prebuilt composite HOD-style model. For a tutorial on the ``tinker13``
        composite model, see :ref:`tinker13_composite_model`.

    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift,
            **kwargs):
        """
        Parameters
        ----------
        threshold : float, optional
            Logarithm of the stellar mass threshold of the mock galaxy sample
            assuming h=1.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies, e.g., ``halo_mvir``.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the galaxy population being modeled. This parameter will also
            determine the default values of the model parameters, according to
            Table 2 of arXiv:1308.2974.
            Default is set in `~halotools.sim_manager.sim_defaults`.
        """
        self._littleh = 0.7
        upper_occupation_bound = 1.0

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tinker13Cens, self).__init__(
            gal_type='centrals', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)
        self.redshift = redshift

        self._initialize_param_dict(self.redshift)

        self.sfr_designation_key = 'central_sfr_designation'

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        # Here we are overriding this attribute that is normally defined
        # in the OccupationComponent super class
        self._methods_to_inherit = (
            ['mc_occupation', 'mean_occupation', 'mean_occupation_active', 'mean_occupation_quiescent',
            'mean_stellar_mass_active', 'mean_stellar_mass_quiescent',
            'mean_log_halo_mass_active', 'mean_log_halo_mass_quiescent']
            )

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_sfr_designation', 'mc_occupation']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('halo_num_' + self.gal_type, 'i4'),
            (self.sfr_designation_key, object),
            ('sfr_designation', object),
            ])

    def _initialize_param_dict(self, redshift):
        """
        """
        zchar = _get_closest_redshift(redshift)
        if zchar == 'z1':
            full_param_dict = deepcopy(param_dict_z1)
        elif zchar == 'z2':
            full_param_dict = deepcopy(param_dict_z2)
        else:
            full_param_dict = deepcopy(param_dict_z3)
        self.param_dict = {}
        for key in full_param_dict.keys():
            if 'smhm_' in key:
                self.param_dict[key] = full_param_dict[key]
            elif 'scatter_' in key:
                self.param_dict[key] = full_param_dict[key]
            elif 'quiescent_fraction_ordinates' in key:
                self.param_dict[key] = full_param_dict[key]

    def _mean_log_halo_mass(self, log_sm_h0p7, logm0_h0p7, logm1_h0p7, beta, delta, gamma):
        """ Mean halo mass as a function of stellar mass of central galaxies.

        As in eqn (1) of arXiv:1308.2974, h=0.7 is assumed when quoting values for
        the input stellar mass, M0, M1, and also the returned value of halo mass.
        Inputs and outputs thus need to be scaled by the user or some
        external function in order to calculate results in the h=1 units
        used throughout the rest of Halotools, which is why this is a private function.
        """
        sm_h0p7 = 10.**log_sm_h0p7
        m0 = 10.**logm0_h0p7
        sm_by_m0 = sm_h0p7/m0

        term2 = beta*np.log10(sm_by_m0)

        term3_numerator = sm_by_m0**delta
        term3_denominator = 1 + sm_by_m0**(-gamma)
        term3 = term3_numerator/term3_denominator

        log_mh_h0p7 = logm1_h0p7 + term2 + term3 - 0.5
        return log_mh_h0p7

    def _mean_stellar_mass(self, mh_h0p7, logm0_h0p7, logm1_h0p7, beta, delta, gamma):
        """ Mean stellar mass as a function of halo mass of central galaxies.

        As in eqn (1) of arXiv:1308.2974, h=0.7 is assumed when quoting values for
        the input halo mass, M0, M1, and also the returned value of stellar mass.
        Inputs and outputs thus need to be scaled by the user or some
        external function in order to calculate results in the h=1 units
        used throughout the rest of Halotools, which is why this is a private function.

        """
        logsm_h0p7_table = np.linspace(8., 12.5, 500)
        log_mh_h0p7_table = self._mean_log_halo_mass(logsm_h0p7_table,
                logm0_h0p7, logm1_h0p7, beta, delta, gamma)

        log_sm_h0p7 = np.interp(np.log10(mh_h0p7),
                log_mh_h0p7_table, logsm_h0p7_table)
        sm_h0p7 = 10.**log_sm_h0p7

        return sm_h0p7

    def mean_quiescent_fraction(self, **kwargs):
        """
        Central galaxy quiescent fraction vs. halo mass

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        quiescent_fraction : array
            Quiescent fraction of centrals

        Examples
        ---------
        >>> model = Tinker13Cens()
        >>> frac_q = model.mean_quiescent_fraction(prim_haloprop=1e12)
        >>> frac_q = model.mean_quiescent_fraction(prim_haloprop=np.logspace(10, 15, 100))
        """
        if 'prim_haloprop' in kwargs:
            prim_haloprop = np.atleast_1d(kwargs['prim_haloprop'])
        elif 'table' in kwargs:
            table = kwargs['table']
            try:
                prim_haloprop = table[self.prim_haloprop_key]
            except KeyError:
                msg = ("The ``table`` passed as a keyword argument "
                    "to the mean_quiescent_fraction method\n"
                    "does not have the requested ``%s`` key")
                raise HalotoolsError(msg % self.prim_haloprop_key)

        keys = list('quiescent_fraction_ordinates_param'+str(i) for i in range(1, 6))
        model_ordinates = [self.param_dict[key] for key in keys]

        fraction = np.interp(np.log10(prim_haloprop),
                np.log10(quiescent_fraction_control_masses), model_ordinates)
        fraction = np.where(fraction < 0, 0, fraction)
        fraction = np.where(fraction > 1, 1, fraction)
        return fraction

    def mc_sfr_designation(self, seed=None, **kwargs):
        """ Monte Carlo realization of the SFR designation of centrals
        (quiescent vs. active).

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        sfr_designation : array
            String array storing values of either ``quiescent`` or ``active``.

        Examples
        ---------
        >>> model = Tinker13Cens()
        >>> sfr_designation = model.mc_sfr_designation(prim_haloprop=1e12)
        >>> halo_mass_array = np.logspace(10, 15, 100)
        >>> sfr_designation = model.mc_sfr_designation(prim_haloprop=halo_mass_array)
        >>> sfr_designation = model.mc_sfr_designation(prim_haloprop=halo_mass_array, seed=43)

        """
        quiescent_fraction = self.mean_quiescent_fraction(**kwargs)

        with NumpyRNGContext(seed):
            mc_generator = np.random.random(custom_len(quiescent_fraction))

        result = np.where(mc_generator < quiescent_fraction, 'quiescent', 'active')
        if 'table' in kwargs:
            kwargs['table'][self.sfr_designation_key] = result
            kwargs['table']['sfr_designation'] = result

        return result

    def mean_occupation(self, **kwargs):
        r""" Expected number of central galaxies as a function of halo mass.
        See Equation 3 of arXiv:1308.2974.

        .. note::

            In the Tinker+13 model, :math:`\langle N_{\rm cen}|M_{\rm halo} \rangle`
            depends is distinct for quiescent vs. active samples. Internally, the
            `mean_occupation` function separately calls the functions
            `mean_occupation_quiescent` and `mean_occupation_active`.
            The `mean_occupation` function the applies the
            mapping that is appropriate for the input SFR-designation,
            which can either be specified by an additional input ``sfr_designation``
            argument, or alternatively as a ``central_sfr_designation`` column
            of the input ``table``. See the Examples section below.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
            Halo mass units are in Msun/h, here and throughout Halotools.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies as a function of the input halos

        Examples
        --------
        In the first example call below, we'll use the ``prim_haloprop`` and
        ``sfr_designation`` keyword arguments:

        >>> model = Tinker13Cens(threshold=10, redshift=1)
        >>> halo_mass = 1e12
        >>> sfr_designation = 'quiescent'
        >>> mean_ncen = model.mean_occupation(prim_haloprop=1e12, sfr_designation=sfr_designation)

        In the next example, we'll use the ``table`` argument, as is done during
        mock-population. To do this, we will need to make sure that the input table
        contains a ``sfr_designation`` column. This column will automatically be
        added during mock-population, but for these demonstration purposes we will
        need to add it manually:

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim(redshift=1)
        >>> random_sfr = np.random.choice(['quiescent', 'active'], len(halocat.halo_table))
        >>> halocat.halo_table['central_sfr_designation'] = random_sfr
        >>> mean_ncen = model.mean_occupation(table=halocat.halo_table)

        """
        if 'table' in kwargs:
            table = kwargs['table']
            try:
                prim_haloprop = table[self.prim_haloprop_key]
            except KeyError:
                msg = ("The ``table`` passed as a keyword argument to the ``mean_occupation`` method\n"
                    "does not have the requested ``%s`` key")
                raise HalotoolsError(msg % self.prim_haloprop_key)
            try:
                sfr_designation = table[self.sfr_designation_key]
            except KeyError:
                msg = ("The ``table`` passed as a keyword argument to the ``mean_occupation`` method\n"
                    "does not have the requested ``%s`` key used for SFR designation")
                raise HalotoolsError(msg % self.sfr_designation_key)
        else:
            try:
                prim_haloprop = np.atleast_1d(kwargs['prim_haloprop'])
                sfr_designation = np.atleast_1d(kwargs['sfr_designation'])
            except KeyError:
                msg = ("If not passing a ``table`` keyword argument to the ``mean_occupation`` method,\n"
                    "you must pass both ``prim_haloprop`` and ``sfr_designation`` keyword arguments")
                raise HalotoolsError(msg)
            if type(sfr_designation[0]) in (str, unicode, np.string_, np.unicode_):
                if sfr_designation[0] not in ['active', 'quiescent']:
                    msg = ("The only acceptable values of "
                        "``sfr_designation`` are ``active`` or ``quiescent``")
                    raise HalotoolsError(msg)

        if 'table' in kwargs:
            quiescent_result = self.mean_occupation_quiescent(table=table)
            active_result = self.mean_occupation_active(table=table)
        else:
            quiescent_result = self.mean_occupation_quiescent(prim_haloprop=prim_haloprop)
            active_result = self.mean_occupation_active(prim_haloprop=prim_haloprop)

        result = np.where(sfr_designation == 'quiescent', quiescent_result, active_result)

        return result

    def mean_occupation_active(self, **kwargs):
        r""" Expected number of active central galaxies as a function of halo mass.
        See Equation 3 of arXiv:1308.2974.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
            Halo mass units are in Msun/h, here and throughout Halotools.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of active central galaxies as a function of the input halos

        Examples
        --------
        >>> model = Tinker13Cens(threshold=10.5, redshift=0.5)
        >>> mean_ncen = model.mean_occupation_active(prim_haloprop=1e12)
        """
        if 'table' in list(kwargs.keys()):
            halo_mass_unity_h = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            halo_mass_unity_h = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation_active`` function of the ``Tinker13Cens`` class.\n")
            raise HalotoolsError(msg)

        sm_unity_h = self.mean_stellar_mass_active(halo_mass_unity_h)
        sm_h0p7 = sm_unity_h/self._littleh/self._littleh
        log_sm_h0p7 = np.log10(sm_h0p7)
        logscatter = math.sqrt(2)*self.param_dict['scatter_model_param1_active']

        sm_thresh_unity_h = 10**self.threshold
        sm_thresh_h0p7 = sm_thresh_unity_h/self._littleh/self._littleh
        log_sm_thresh_h0p7 = np.log10(sm_thresh_h0p7)

        erfarg = (log_sm_thresh_h0p7 - log_sm_h0p7)/logscatter
        mean_ncen = 0.5*(1.0 - erf(erfarg))
        mean_ncen *= (1. - self.mean_quiescent_fraction(prim_haloprop=halo_mass_unity_h))

        return mean_ncen

    def mean_occupation_quiescent(self, **kwargs):
        r""" Expected number of quiescent central galaxies as a function of halo mass.
        See Equation 3 of arXiv:1308.2974.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
            Halo mass units are in Msun/h, here and throughout Halotools.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of quiescent central galaxies as a function of the input halos

        Examples
        --------
        >>> model = Tinker13Cens(threshold=10.5, redshift=0.5)
        >>> mean_ncen = model.mean_occupation_quiescent(prim_haloprop=1e12)
        """
        if 'table' in list(kwargs.keys()):
            halo_mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation_active`` function of the ``Tinker13Cens`` class.\n")
            raise HalotoolsError(msg)

        sm_unity_h = self.mean_stellar_mass_quiescent(halo_mass)
        sm_h0p7 = sm_unity_h/self._littleh/self._littleh
        log_sm_h0p7 = np.log10(sm_h0p7)
        logscatter = math.sqrt(2)*self.param_dict['scatter_model_param1_quiescent']

        sm_thresh_unity_h = 10**self.threshold
        sm_thresh_h0p7 = sm_thresh_unity_h/self._littleh/self._littleh
        log_sm_thresh_h0p7 = np.log10(sm_thresh_h0p7)

        erfarg = (log_sm_thresh_h0p7 - log_sm_h0p7)/logscatter
        mean_ncen = 0.5*(1.0 - erf(erfarg))
        mean_ncen *= self.mean_quiescent_fraction(prim_haloprop=halo_mass)

        return mean_ncen

    def mean_stellar_mass_active(self, prim_haloprop):
        r""" Average stellar mass of active central galaxies
        as a function of halo mass.
        See Equation 1 of arXiv:1308.2974.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
            Halo mass units are in Msun/h, here and throughout Halotools.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mstar : array
            stellar mass

        Examples
        --------
        >>> model = Tinker13Cens(threshold=10.5, redshift=0.5)
        >>> mstar = model.mean_stellar_mass_active(prim_haloprop=1e12)
        """
        args = self._retrieve_smhm_param_values('active')

        halo_mass_unity_h = prim_haloprop
        halo_mass_h0p7 = halo_mass_unity_h/self._littleh
        sm_h0p7 = self._mean_stellar_mass(halo_mass_h0p7, *args)
        sm_unity_h = sm_h0p7*self._littleh*self._littleh
        return sm_unity_h

    def mean_stellar_mass_quiescent(self, prim_haloprop):
        r""" Average stellar mass of quiescent central galaxies
        as a function of halo mass.
        See Equation 1 of arXiv:1308.2974.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
            Halo mass units are in Msun/h, here and throughout Halotools.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mstar : array
            stellar mass

        Examples
        --------
        >>> model = Tinker13Cens(threshold=10.5, redshift=0.5)
        >>> mstar = model.mean_stellar_mass_quiescent(prim_haloprop=1e12)
        """
        args = self._retrieve_smhm_param_values('quiescent')

        halo_mass_unity_h = prim_haloprop
        halo_mass_h0p7 = halo_mass_unity_h/self._littleh
        sm_h0p7 = self._mean_stellar_mass(halo_mass_h0p7, *args)
        sm_unity_h = sm_h0p7*self._littleh*self._littleh
        return sm_unity_h

    def mean_log_halo_mass_active(self, log_stellar_mass):
        r""" Average halo mass as a function of stellar mass for active central galaxies.
        See Equation 1 of arXiv:1308.2974.

        Parameters
        ----------
        log_stellar_mass : array, optional
            Array of shape (num_gals, ) storing log10(M*) assuming h=1.

        Returns
        -------
        log_halo_mass : array
            log10(Mh) assuming h=1

        Examples
        --------
        >>> model = Tinker13Cens(threshold=10.5, redshift=0.5)
        >>> halo_mass = 10**model.mean_log_halo_mass_active(10.5)
        """
        args = self._retrieve_smhm_param_values('quiescent')

        sm_unity_h = 10**log_stellar_mass
        sm_h0p7 = sm_unity_h/self._littleh/self._littleh
        log_sm_h0p7 = np.log10(sm_h0p7)
        mh_h0p7 = 10**self._mean_log_halo_mass(log_sm_h0p7, *args)
        mh_unity_h = mh_h0p7*self._littleh
        return np.log10(mh_unity_h)

    def mean_log_halo_mass_quiescent(self, log_stellar_mass):
        r""" Average halo mass as a function of stellar mass for quiescent central galaxies.
        See Equation 1 of arXiv:1308.2974.

        Parameters
        ----------
        log_stellar_mass : array, optional
            Array of shape (num_gals, ) storing log10(M*) assuming h=1.

        Returns
        -------
        log_halo_mass : array
            log10(Mh) assuming h=1

        Examples
        --------
        >>> model = Tinker13Cens(threshold=10.5, redshift=0.5)
        >>> halo_mass = 10**model.mean_log_halo_mass_quiescent(10.5)
        """
        args = self._retrieve_smhm_param_values('quiescent')

        sm_unity_h = 10**log_stellar_mass
        sm_h0p7 = sm_unity_h/self._littleh/self._littleh
        log_sm_h0p7 = np.log10(sm_h0p7)
        mh_h0p7 = 10**self._mean_log_halo_mass(log_sm_h0p7, *args)
        mh_unity_h = mh_h0p7*self._littleh
        return np.log10(mh_unity_h)

    def _retrieve_smhm_param_values(self, sfr_key):
        if sfr_key == 'active':
            keys = ('smhm_m0_0_active', 'smhm_m1_0_active', 'smhm_beta_0_active',
                'smhm_delta_0_active', 'smhm_gamma_0_active')
        elif sfr_key == 'quiescent':
            keys = ('smhm_m0_0_quiescent', 'smhm_m1_0_quiescent', 'smhm_beta_0_quiescent',
                'smhm_delta_0_quiescent', 'smhm_gamma_0_quiescent')
        return list(self.param_dict[key] for key in keys)


class AssembiasTinker13Cens(Tinker13Cens, HeavisideAssembias):
    """ HOD-style model for a central galaxy occupation that derives from
    two distinct active/quiescent stellar-to-halo-mass relations.
    """
    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift,
            **kwargs):
        """
        Parameters
        ----------
        threshold : float, optional
            Logarithm of the stellar mass threshold of the mock galaxy sample
            assuming h=1.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies, e.g., ``halo_mvir``.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the galaxy population being modeled. This parameter will also
            determine the default values of the model parameters, according to
            Table 2 of arXiv:1308.2974.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
            Values of the primary halo property at which the halos are split as described above in
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior),
            the interpolation will be done in the logarithm of the primary halo property.
            Default is to assume a constant 50/50 split.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        """
        Tinker13Cens.__init__(self, **kwargs)
        HeavisideAssembias.__init__(self,
            method_name_to_decorate='mean_quiescent_fraction',
            lower_assembias_bound=0.,
            upper_assembias_bound=1.,
            **kwargs)


class Tinker13QuiescentSats(OccupationComponent):
    """ HOD-style model for a quiescent satellite galaxy occupation that derives
    from a stellar-to-halo-mass relation, as in arXiv:1308.2974.

    .. note::

        The `Tinker13QuiescentSats` model is part of the ``tinker13``
        prebuilt composite HOD-style model. For a tutorial on the ``tinker13``
        composite model, see :ref:`tinker13_composite_model`.
    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift, **kwargs):
        """
        Parameters
        ----------
        threshold : float, optional
            Logarithm of the stellar mass threshold of the mock galaxy sample
            assuming h=1.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies, e.g., ``halo_mvir``.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the galaxy population being modeled. This parameter will also
            determine the default values of the model parameters, according to
            Table 2 of arXiv:1308.2974.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        Examples
        ---------
        >>> model = Tinker13QuiescentSats()
        >>> model = Tinker13QuiescentSats(threshold=10.25, prim_haloprop_key='halo_m200b', redshift=0.5)

        """
        upper_occupation_bound = float("inf")

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tinker13QuiescentSats, self).__init__(
            gal_type='quiescent_satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key, **kwargs)
        self.redshift = redshift

        self.central_occupation_model = Tinker13Cens(threshold=threshold,
            prim_haloprop_key=prim_haloprop_key, redshift=redshift)

        self._initialize_param_dict(redshift)

        self.sfr_designation_key = 'sfr_designation'

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        # Here we are overriding this attribute that is normally defined
        # in the OccupationComponent super class
        self._methods_to_inherit = ['mc_occupation', 'mean_occupation']

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_occupation', 'mc_sfr_designation']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('halo_num_' + self.gal_type, 'i4'),
            (self.sfr_designation_key, object),
            ])

    def mean_occupation(self, **kwargs):
        """ Expected number of quiescent satellite galaxies as a function of halo mass.
        See Equation 4 of arXiv:1308.2974.

        Parameters
        ----------
        prim_haloprop : array, optional
            array of masses of table in the catalog

        table : object, optional
            Data table storing halo catalog.

        Returns
        -------
        mean_nsat : array
            Mean number of quiescent satellite galaxies as a function of the input halos.

        Examples
        --------
        >>> sat_model = Tinker13QuiescentSats()
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop=1e13)

        Notes
        -----
        Assumes constant scatter in the stellar-to-halo-mass relation.
        """
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            function_name = "Tinker13QuiescentSats.mean_occupation"
            raise HalotoolsError(function_name)

        self._update_satellite_params()

        power_law_factor = (mass/self._msat)**self.param_dict['alphasat_quiescent']

        _mh_q = 10**self.central_occupation_model.mean_log_halo_mass_quiescent(self.threshold)
        _mh_a = 10**self.central_occupation_model.mean_log_halo_mass_active(self.threshold)
        _fred_cen = self.central_occupation_model.mean_quiescent_fraction(prim_haloprop=_mh_a)
        _mh = _mh_q*_fred_cen + _mh_a*(1-_fred_cen)
        exp_arg_numerator = self._mcut + _mh
        exp_factor = np.exp(-exp_arg_numerator/mass)

        mean_nsat = exp_factor*power_law_factor

        return mean_nsat

    def mc_sfr_designation(self, table, **kwargs):
        """
        """
        table[self.sfr_designation_key][:] = 'quiescent'

    def _initialize_param_dict(self, redshift):
        """ Set the initial values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """
        zchar = _get_closest_redshift(redshift)
        if zchar == 'z1':
            full_param_dict = deepcopy(param_dict_z1)
        elif zchar == 'z2':
            full_param_dict = deepcopy(param_dict_z2)
        else:
            full_param_dict = deepcopy(param_dict_z3)
        self.param_dict = {}

        keygen = (key for key in full_param_dict.keys()
            if 'quiescent' in key and 'fraction' not in key)
        for key in keygen:
            self.param_dict[key] = full_param_dict[key]

    def _update_satellite_params(self):
        """ Private method to update the model parameters.

        """
        for key, value in self.param_dict.items():
            stripped_key = key[:-len('_quiescent')]
            if stripped_key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[stripped_key] = value

        _f = self.central_occupation_model.mean_log_halo_mass_quiescent
        log_halo_mass_threshold = _f(self.threshold)
        knee_threshold = (10.**log_halo_mass_threshold)

        knee_mass = 1.e12

        self._msat = (
            knee_mass*self.param_dict['bsat_quiescent'] *
            (knee_threshold / knee_mass)**self.param_dict['betasat_quiescent'])

        self._mcut = (
            knee_mass*self.param_dict['bcut_quiescent'] *
            (knee_threshold / knee_mass)**self.param_dict['betacut_quiescent'])


class Tinker13ActiveSats(OccupationComponent):
    """ HOD-style model for an active satellite galaxy occupation that derives
    from a stellar-to-halo-mass relation, as in arXiv:1308.2974.

    .. note::

        The `Tinker13ActiveSats` model is part of the ``tinker13``
        prebuilt composite HOD-style model. For a tutorial on the ``tinker13``
        composite model, see :ref:`tinker13_composite_model`.
    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift, **kwargs):
        """
        Parameters
        ----------
        threshold : float, optional
            Logarithm of the stellar mass threshold of the mock galaxy sample
            assuming h=1.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies, e.g., ``halo_mvir``.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the galaxy population being modeled. This parameter will also
            determine the default values of the model parameters, according to
            Table 2 of arXiv:1308.2974.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        Examples
        ---------
        >>> model = Tinker13ActiveSats()
        >>> model = Tinker13ActiveSats(threshold=10.25, prim_haloprop_key='halo_m200b', redshift=0.5)

        """
        upper_occupation_bound = float("inf")

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tinker13ActiveSats, self).__init__(
            gal_type='active_satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key, **kwargs)
        self.redshift = redshift

        self.central_occupation_model = Tinker13Cens(threshold=threshold,
            prim_haloprop_key=prim_haloprop_key, redshift=redshift)

        self._initialize_param_dict(redshift)

        self.sfr_designation_key = 'sfr_designation'

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        # Here we are overriding this attribute that is normally defined
        # in the OccupationComponent super class
        self._methods_to_inherit = ['mc_occupation', 'mean_occupation']

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_occupation', 'mc_sfr_designation']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('halo_num_' + self.gal_type, 'i4'),
            (self.sfr_designation_key, object),
            ])

    def mean_occupation(self, **kwargs):
        """ Expected number of active satellite galaxies as a function of halo mass.
        See Equation 4 of arXiv:1308.2974.

        Parameters
        ----------
        prim_haloprop : array, optional
            array of masses of table in the catalog

        table : object, optional
            Data table storing halo catalog.

        Returns
        -------
        mean_nsat : array
            Mean number of central galaxies in the halo of the input mass.

        Examples
        --------
        >>> sat_model = Tinker13ActiveSats()
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop = 1.e13)

        Notes
        -----
        Assumes constant scatter in the stellar-to-halo-mass relation.
        """
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            function_name = "Tinker13ActiveSats.mean_occupation"
            raise HalotoolsError(function_name)

        self._update_satellite_params()

        power_law_factor = (mass/self._msat)**self.param_dict['alphasat_active']

        _mh_q = 10**self.central_occupation_model.mean_log_halo_mass_quiescent(self.threshold)
        _mh_a = 10**self.central_occupation_model.mean_log_halo_mass_active(self.threshold)
        _fred_cen = self.central_occupation_model.mean_quiescent_fraction(prim_haloprop=_mh_a)
        _mh = _mh_q*_fred_cen + _mh_a*(1-_fred_cen)
        exp_arg_numerator = self._mcut + _mh
        exp_factor = np.exp(-exp_arg_numerator/mass)

        mean_nsat = exp_factor*power_law_factor

        return mean_nsat

    def mc_sfr_designation(self, table, **kwargs):
        """
        """
        table[self.sfr_designation_key][:] = 'active'

    def _initialize_param_dict(self, redshift):
        """ Set the initial values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """
        zchar = _get_closest_redshift(redshift)
        if zchar == 'z1':
            full_param_dict = deepcopy(param_dict_z1)
        elif zchar == 'z2':
            full_param_dict = deepcopy(param_dict_z2)
        else:
            full_param_dict = deepcopy(param_dict_z3)
        self.param_dict = {}

        keygen = (key for key in full_param_dict.keys()
            if 'active' in key and 'fraction' not in key)
        for key in keygen:
            self.param_dict[key] = full_param_dict[key]

    def _update_satellite_params(self):
        """ Private method to update the model parameters.

        """
        for key, value in self.param_dict.items():
            stripped_key = key[:-len('_active')]
            if stripped_key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[stripped_key] = value

        _f = self.central_occupation_model.mean_log_halo_mass_quiescent
        log_halo_mass_threshold = _f(self.threshold)
        knee_threshold = (10.**log_halo_mass_threshold)

        knee_mass = 1.e12

        self._msat = (
            knee_mass*self.param_dict['bsat_active'] *
            (knee_threshold / knee_mass)**self.param_dict['betasat_active'])

        self._mcut = (
            knee_mass*self.param_dict['bcut_active'] *
            (knee_threshold / knee_mass)**self.param_dict['betacut_active'])
