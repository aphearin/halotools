.. _cam_disk_bulge_ratios:


Modeling Halo Spin-Dependent Disk-to-Bulge Mass Ratios
=======================================================

In this example, we will show how to use Conditional Abundance Matching to
build a very simple model for :math:`B/T`, the bulge-to-total stellar mass ratio.
In this model, galaxies with increasing stellar mass become "bulgier",
and at fixed stellar mass, halos with low spin have bigger bulges than
halos with large spin. While this model is physically simplistic, it demonstrates
an alternative usage of CAM beyond the log-normal distribution shown in the
tutorial on :ref:`cam_decorated_clf`.
The code used to generate these results can be found here:

    **halotools/docs/notebooks/cam_modeling/cam_disk_bulge_ratios_demo.ipynb**


Baseline model for B/T
------------------------------------------

.. code:: python

    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog()

    from halotools.empirical_models import Moster13SmHm
    model = Moster13SmHm()
    halocat.halo_table['stellar_mass'] = model.mc_stellar_mass(
        prim_haloprop=halocat.halo_table['halo_mpeak'], redshift=0)


.. code:: python

    from halotools.utils import sliding_conditional_percentile

    x = halocat.halo_table['stellar_mass']
    y = halocat.halo_table['halo_spin']
    nwin = 201
    halocat.halo_table['spin_percentile'] = sliding_conditional_percentile(x, y, nwin)

    def powerlaw_index(log_mstar):
        abscissa = [9, 10, 11.5]
        ordinates = [3, 2, 1]
        return np.interp(log_mstar, abscissa, ordinates)

    a = powerlaw_index(np.log10(halocat.halo_table['stellar_mass']))
    u = halocat.halo_table['spin_percentile']
    halocat.halo_table['bulge_to_total_ratio'] = 1 - powerlaw.isf(1 - u, a)


Correlating B/T with halo spin at fixed stellar mass
----------------------------------------------------------------

.. image:: /_static/cam_example_bulge_disk_ratio.png
