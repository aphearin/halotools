# -*- coding: utf-8 -*-
"""
Module containing classes used to model the mapping between 
stellar mass and subhalos. 
"""
import numpy as np

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from. import model_defaults
from ..utils.array_utils import array_like_length as custom_len
from ..empirical_models import occupation_helpers as occuhelp
from ..sim_manager import sim_defaults 

from warnings import warn

__all__ = ['SmHmModel', 'Moster13SmHm', 'LogNormalScatterModel']

class LogNormalScatterModel(object):
    """ Simple model used to generate log-normal scatter 
    in the stellar-to-halo-mass_like relation. 

    """

    def __init__(self, 
        prim_haloprop_key=model_defaults.default_smhm_haloprop, 
        **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the level of scatter. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_spline_degree : int, optional keyword argument
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. 
        """
        default_scatter = model_defaults.default_smhm_scatter
        self.prim_haloprop_key = prim_haloprop_key

        if ('scatter_abcissa' in kwargs.keys()) and ('scatter_ordinates' in kwargs.keys()):
            self.abcissa = kwargs['scatter_abcissa']
            self.ordinates = kwargs['scatter_ordinates']
        else:
            self.abcissa = [12]
            self.ordinates = [default_scatter]
        self._initialize_param_dict()

        self._setup_interpol(**kwargs)

    def mean_scatter(self, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halos. 

        Parameters 
        ----------
        mass_like : array, optional keyword argument 
            array of halo mass_likees 

        halos : array or table, optional keyword argument
            Data structure containing halos onto which stellar mass_likees 
            will be painted. Must contain a key that matches ``prim_haloprop_key``. 

        Returns 
        -------
        scatter : array_like 
            Array containing the amount of log-normal scatter evaluated 
            at the input halos. 
        """
        # Interpret the inputs to determine the appropriate array of mass_likees
        if 'mass_like' not in kwargs.keys():
            if 'halos' in kwargs.keys():
                if not hasattr(self, 'prim_haloprop_key'):
                    raise SyntaxError("If you want to be able to pass "
                        " a halo catalog as input to the scatter model, "
                        "you must pass a `prim_haloprop_key` to the constructor")
                kwargs['mass_like'] = kwargs['halos'][self.prim_haloprop_key]
            else:
                raise SyntaxError("You must either pass an input ``mass_like`` keyword "
                    " or an input ``halos`` keyword, received neither")

        if 'param_dict' in kwargs.keys():
            self._update_params(kwargs['param_dict'])
        else:
            self._update_params(self.param_dict)

        return self.spline_function(np.log10(kwargs['mass_like']))

    def scatter_realization(self, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halos. 

        Parameters 
        ----------
        mass_like : array, optional keyword argument 
            array of halo mass_likees 

        halos : array or table, optional keyword argument
            Data structure containing halos onto which stellar mass_likees 
            will be painted. Must contain a key that matches ``prim_haloprop_key``. 

        Returns 
        -------
        scatter : array_like 
            Array containing the amount of log-normal scatter evaluated 
            at the input halos. 
        """

        scatter_scale = self.mean_scatter(**kwargs)

        if 'seed' in kwargs.keys():
            np.random.seed(seed=kwargs['seed'])
            
        return np.random.normal(loc=0, scale=scatter_scale)

    def _setup_interpol(self, **kwargs):
        """ Private method used to configure the behavior of the interpolating function. 
        """        
        scipy_maxdegree = 5
        degree_list = [scipy_maxdegree, custom_len(self.abcissa)-1]
        if 'scatter_spline_degree' in kwargs.keys():
            degree_list.append(kwargs['scatter_spline_degree'])
        self.spline_degree = np.min(degree_list)

        self.spline_function = occuhelp.custom_spline(
            self.abcissa, self.ordinates, k=self.spline_degree)

    def _update_params(self, param_dict):
        self.param_dict = param_dict
        self.ordinates = (
            [self.param_dict[self._get_param_key(ipar)] 
            for ipar in range(len(self.abcissa))]
            )
        self._setup_interpol()

    def _initialize_param_dict(self):

        self.param_dict={}
        for ipar, val in enumerate(self.ordinates):
            key = self._get_param_key(ipar)
            self.param_dict[key] = val

    def _get_param_key(self, ipar):
        """ Private method used to retrieve the key of self.param_dict 
        that corresponds to the appropriately selected i^th ordinate 
        defining the behavior of the scatter model. 
        """
        return 'scatter_model_param'+str(ipar+1)

@six.add_metaclass(ABCMeta)
class SmHmModel(object):
    """ Abstract container class used as a template 
    for how to build a stellar-to-halo-mass_like-style model.

    """

    def __init__(self, 
        prim_haloprop_key = model_defaults.default_smhm_haloprop, 
        scatter_model = LogNormalScatterModel, 
        **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the level of scatter. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional keyword argument 
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_spline_degree : int, optional keyword argument
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. 

        input_param_dict : dict, optional keyword argument
            Dictionary containing values for the parameters specifying the model.

        """
        self.galprop_key = 'stellar_mass'
        self.prim_haloprop_key = prim_haloprop_key

        if 'redshift' in kwargs.keys():
            self.redshift = kwargs['redshift']

        self.scatter_model = scatter_model(
            prim_haloprop_key=self.prim_haloprop_key, 
            **kwargs)

        self._build_param_dict(**kwargs)


    def _build_param_dict(self, **kwargs):

        if 'input_param_dict' in kwargs.keys():
            smhm_param_dict = kwargs['input_param_dict']
        else:
            if hasattr(self, 'retrieve_default_param_dict'):
                smhm_param_dict = self.retrieve_default_param_dict()
            else:
                raise KeyError("If the class has no retrieve_default_param_dict method, "
                    "you must pass param_dict as a keyword argument to the constructor")

        scatter_param_dict = self.scatter_model.param_dict

        self.param_dict = dict(
            smhm_param_dict.items() + 
            scatter_param_dict.items()
            )

    def mc_stellar_mass(self, include_scatter = True, **kwargs):
        """ Return the stellar mass_like of a central galaxy that lives in a 
        halo mass_like ``mass_like`` at the input ``redshift``. 

        Parameters 
        ----------
        prim_haloprop : array, optional keyword argument 
            Array of mass-like variable governing stellar mass. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional keyword argument 
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional keyword argument 
            Data table storing galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        redshift : float, optional keyword argument
            Redshift of the halo hosting the galaxy. 

        include_scatter : boolean, optional keyword argument 
            Determines whether or not the scatter model is applied to add stochasticity 
            to the stellar mass assignment. Default is True. 
            If False, model is purely deterministic. 

        input_param_dict : dict, optional keyword argument 
            Dictionary of parameters governing the model. 
            If not passed, the values already bound to ``self`` will be used. 

        Returns 
        -------
        mstar : array_like 
            Array containing stellar mass living in the input halos. 
        """

        # Interpret the inputs to determine the appropriate redshift
        if 'redshift' not in kwargs.keys():
            if hasattr(self, 'redshift'):
                kwargs['redshift'] = self.redshift
            else:
                warn("\nThe SmHmModel class was not instantiated with a redshift,\n"
                "nor was a redshift passed to the primary function call.\n"
                "Choosing the default redshift z = %.2f\n" % sim_defaults.default_redshift)
                kwargs['redshift'] = sim_defaults.default_redshift

        mean_stellar_mass = self.mean_stellar_mass(**kwargs)

        if include_scatter is False:
            return mean_stellar_mass
        else:
            log10mass_like_with_scatter = (
                np.log10(mean_stellar_mass) + 
                self.scatter_model.scatter_realization(**kwargs)
                )
            return 10.**log10mass_like_with_scatter


class Moster13SmHm(SmHmModel):
    """ Stellar-to-halo-mass_like relation based on 
    Moster et al. (2013), arXiv:1205.5807. 
    """

    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the level of scatter. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional keyword argument 
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_spline_degree : int, optional keyword argument
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. 

        input_param_dict : dict, optional keyword argument
            Dictionary containing values for the parameters specifying the model.
            If none is passed, the `Moster13SmHm` instance will be initialized to 
            the best-fit values taken from Moster et al. (2013). 
        """

        super(Moster13SmHm, self).__init__(**kwargs)

        self.publications = ['arXiv:0903.4682', 'arXiv:1205.5807']

    def mean_stellar_mass(self, **kwargs):
        """ Return the stellar mass_like of a central galaxy that lives in a 
        halo mass_like ``mass_like`` at the input ``redshift``. 

        Parameters 
        ----------
        prim_haloprop : array, optional keyword argument 
            Array of mass-like variable governing stellar mass. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional keyword argument 
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional keyword argument 
            Data table storing galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        redshift : float, keyword argument
            Redshift of the halo hosting the galaxy

        Returns 
        -------
        mstar : array_like 
            Array containing stellar masses living in the input halos. 
        """
        # Retrieve the array storing the mass-like variable
        if 'galaxy_table' in kwargs.keys():
            key = model_defaults.host_haloprop_prefix+self.prim_haloprop_key
            mass = kwargs['galaxy_table'][key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halos``, ``prim_haloprop``, or ``galaxy_table``")

        if 'redshift' in kwargs.keys():
            redshift = kwargs['redshift']
        elif hasattr(self, 'redshift'):
            redshift = self.redshift
        else:
            redshift = sim_defaults.default_redshift

        # compute the parameter values that apply to the input redshift
        a = 1./(1+redshift)
        m1 = self.param_dict['m10'] + self.param_dict['m11']*(1-a)
        n = self.param_dict['n10'] + self.param_dict['n11']*(1-a)
        beta = self.param_dict['beta10'] + self.param_dict['beta11']*(1-a)
        gamma = self.param_dict['gamma10'] + self.param_dict['gamma11']*(1-a)

        # Calculate each term contributing to Eqn 2
        norm = 2.*n*mass
        m_by_m1 = mass/(10.**m1)
        denom_term1 = m_by_m1**(-beta)
        denom_term2 = m_by_m1**gamma

        mstar = norm / (denom_term1 + denom_term2)
        return mstar

    def retrieve_default_param_dict(self):
        """ Method returns a dictionary of all model parameters 
        set to the values in Table 1 of Moster et al. (2013). 

        Returns 
        -------
        d : dict 
            Dictionary containing parameter values. 
        """

        d = {
        'm10': 11.590, 
        'm11': 1.195, 
        'n10': 0.0351, 
        'n11': -0.0247, 
        'beta10': 1.376, 
        'beta11': -0.826, 
        'gamma10': 0.608, 
        'gamma11': 0.329
        }
        return d





