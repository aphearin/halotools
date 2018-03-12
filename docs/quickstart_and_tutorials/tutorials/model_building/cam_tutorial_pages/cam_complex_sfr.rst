.. _cam_complex_sfr:

Modeling Complex Star-Formation Rates
==============================================

In this example, we will show how to use Conditional Abundance Matching to model
a correlation between the mass accretion rate of a halo and the specific
star-formation rate of the galaxy living in the halo.
The code used to generate these results can be found here:

    **halotools/docs/notebooks/cam_modeling/cam_complex_sfr_tutorial.ipynb**

Observed star-formation rate distribution
------------------------------------------

We will work with a distribution of star-formation
rates that would be difficult to model analytically, but that is well-sampled
by some observed galaxy population. The particular form of this distribution
is not important for this tutorial, since our CAM application will directly
use the "observed" population to define the distribution that we recover.

.. image:: /_static/cam_example_complex_sfr.png

The plot above shows the specific star-formation rates of the
toy galaxy distribution we have created for demonstration purposes.
Briefly, there are separate distributions for quenched and star-forming galaxies.
For the quenched galaxies, we model sSFR using an exponential power law;
for star-forming galaxies, we use a log-normal;
implementation details can be found in the notebook.

.. image:: /_static/cam_example_complex_sfr_recovery.png
