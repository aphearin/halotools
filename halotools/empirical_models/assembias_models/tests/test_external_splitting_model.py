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


    halocat = FakeSim(seed=43, num_halos_per_massbin=1000)

    testmass = 10**model.param_dict['logMmin']
    idx = np.argmin(np.abs(halocat.halo_table['halo_mvir'] - testmass))
    closest_mass = halocat.halo_table['halo_mvir'][idx]
    mask = halocat.halo_table['halo_mvir'] == closest_mass
    halocat.halo_table['halo_mvir'][mask] = testmass

    testmass_mask = halocat.halo_table['halo_mvir'] == testmass
    testmass_mask *= halocat.halo_table['halo_upid'] == -1
    num_hosts = np.count_nonzero(testmass_mask)
    print("There are a total of {0} host halos with halo_mvir = {1:.2e}\n".format(num_hosts, testmass))

    model.param_dict['mean_occupation_satellites_assembias_param1'] = 0.5
    model.param_dict['logM1'] = np.log10(10**model.param_dict['logMmin']) - 0.5
    model.param_dict['logM0'] = model.param_dict['logMmin'] - 0.5
    model.param_dict['alpha'] = 0.25

    baseline_model.param_dict.update(model.param_dict)

    mean_ncen = model.mean_occupation_centrals(prim_haloprop=testmass)[0]
    mean_nsat0 = model.mean_occupation_satellites(prim_haloprop=testmass, sec_haloprop_percentile=0)[0]
    mean_nsat1 = model.mean_occupation_satellites(prim_haloprop=testmass, sec_haloprop_percentile=1)[0]
    mean_nsat_baseline = baseline_model.mean_occupation_satellites(prim_haloprop=testmass)[0]

    model.populate_mock(halocat, seed=44)

    host_mask = model.mock.galaxy_table['halo_mvir'] == testmass
    host_mask_sats = host_mask * (model.mock.galaxy_table['gal_type'] == 'satellites')
    host_mask_sats_has_cen = host_mask_sats * (model.mock.galaxy_table['halo_num_centrals'] == 1)
    host_mask_sats_no_cen = host_mask_sats * (model.mock.galaxy_table['halo_num_centrals'] == 0)

    host_mask_cens = host_mask * model.mock.galaxy_table['gal_type'] == 'centrals'
    alt_host_mask_cens = model.mock.halo_table['halo_mvir'] == testmass
    alt_host_mask_cens *= model.mock.halo_table['halo_num_centrals'] == 1

    total_num_cens = np.count_nonzero(host_mask_cens)
    alt_total_num_cens = np.count_nonzero(alt_host_mask_cens)
    total_num_sats = np.count_nonzero(host_mask_sats)
    total_num_sats_no_cen = np.count_nonzero(host_mask_sats_no_cen)
    total_num_sats_has_cen = np.count_nonzero(host_mask_sats_has_cen)

    print("Mean central occupation = {0}".format(mean_ncen))
    print("Monte Carlo mean ncen = {0:.4f}".format(total_num_cens/float(num_hosts)))
    print("Alternate MC mean ncen = {0:.4f}\n".format(alt_total_num_cens/float(num_hosts)))
    print("Mean satellite occupation baseline = {0:.4f}".format(mean_nsat_baseline))
    print("Monte Carlo mean nsat = {0:.4f}\n".format(total_num_sats/float(num_hosts)))
    print("Mean satellite occupation lower-percentile = {0:.4f}".format(mean_nsat0))
    print("Mean satellite occupation upper-percentile = {0:.4f}".format(mean_nsat1))

    print("\nTotal number of centrals in mass bin = {0}\n".format(total_num_cens))
    print("Total number of satellites in mass bin = {0}".format(total_num_sats))
    print("Total number of satellites in mass bin WITH a central= {0}".format(
            total_num_sats_has_cen))
    print("Total number of satellites in mass bin WITHOUT a central= {0}".format(
            total_num_sats_no_cen))

    assert 4 == 5


