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


Correlating B/T with halo spin at fixed stellar mass
----------------------------------------------------------------

.. image:: /_static/cam_example_bulge_disk_ratio.png
