"""
"""
import numpy as np

from ..heaviside_assembias import HeavisideAssembias

from ...occupation_models import Zheng07Cens, Zheng07Sats
from ...factories import HodModelFactory, PrebuiltHodModelFactory

from ....sim_manager import FakeSim


__all__ = ('test_splitting_model1', )


class SplittingModel1(object):

    def __init__(self, **kwargs):

        self.param_dict = dict(split_low_mass=0.25, split_high_mass=0.75)

    def some_function(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Cens`` class.\n")
            raise KeyError(msg)

        return np.where(mass < 1e12, self.param_dict['split_low_mass'],
                self.param_dict['split_high_mass'])


def test_splitting_model1():
    """
    """
    class AssembiasExternalSplit(Zheng07Sats, HeavisideAssembias):
        def __init__(self, **kwargs):
            Zheng07Sats.__init__(self, **kwargs)
            HeavisideAssembias.__init__(self,
                lower_assembias_bound=self._lower_occupation_bound,
                upper_assembias_bound=self._upper_occupation_bound,
                method_name_to_decorate='mean_occupation', **kwargs)

    altocc_model = AssembiasExternalSplit(splitting_model=SplittingModel1(),
            splitting_method_name='some_function')

    low_mass, high_mass = 1e10, 1e15

    split0 = altocc_model.percentile_splitting_function(prim_haloprop=low_mass)
    assert np.all(split0 == altocc_model.param_dict['split_low_mass'])
    split1 = altocc_model.percentile_splitting_function(prim_haloprop=high_mass)
    assert np.all(split1 == altocc_model.param_dict['split_high_mass'])

    altocc_model.param_dict['split_low_mass'] = 0.9
    split2 = altocc_model.percentile_splitting_function(prim_haloprop=low_mass)
    assert np.all(split2 == 0.9)

    result0 = altocc_model.mean_occupation(prim_haloprop=low_mass, sec_haloprop_percentile=0)
    result1 = altocc_model.mean_occupation(prim_haloprop=low_mass, sec_haloprop_percentile=1)

    baseline_model = PrebuiltHodModelFactory('zheng07')
    model = HodModelFactory(baseline_model_instance=baseline_model,
            satellites_occupation=altocc_model)

    halocat = FakeSim()
    model.populate_mock(halocat)


def test_splitting_model2():
    """
    """
    class AssembiasExternalSplit(Zheng07Sats, HeavisideAssembias):
        def __init__(self, **kwargs):
            Zheng07Sats.__init__(self, **kwargs)
            HeavisideAssembias.__init__(self,
                lower_assembias_bound=self._lower_occupation_bound,
                upper_assembias_bound=self._upper_occupation_bound,
                method_name_to_decorate='mean_occupation', **kwargs)

    satocc_model = AssembiasExternalSplit(splitting_model=Zheng07Cens(),
            splitting_method_name='mean_occupation',
            halo_type_tuple=('halo_num_centrals', 1, 0))

    baseline_model = PrebuiltHodModelFactory('zheng07')
    model = HodModelFactory(baseline_model_instance=baseline_model,
            satellites_occupation=satocc_model)

    halocat = FakeSim()

    model.mc_occupation_satellites.additional_kwargs = '_occupation'
    model.populate_mock(halocat)


