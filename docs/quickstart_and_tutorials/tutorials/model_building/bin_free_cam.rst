:orphan:

.. _bin_free_cam_tutorial:

**********************************************************************
Examples of Bin-Free Conditional Abundance Matching
**********************************************************************

Each of the sections below illustrates a different application of the same underlying method.
Each section has an accompanying annotated Jupyter notebook with the code used to generate the plots.

Implementation details of bin-free CAM
=======================================

* `scipy.stats` functions implementing the isf method
* Can use `~halotools.empirical_models.noisy_percentile` method, or something else


Satellite Galaxy Quenching Gradients
=======================================

Observations indicate that satellite galaxies are redder in the
inner regions of their host dark matter halos. One way to model this phenomenon is to use CAM
to correlate the quenching probability with host-centric position.
For example, `Zu and Mandelbaum 2016 <https://arxiv.org/abs/1509.06758/>`_ model satellite
quenching with a simple analytical function :math:`{\rm Prob(\ quenched}\ \vert\ M_{\rm host})`,
where :math:`M_{\rm host}` is the dark matter mass of the satellite's parent halo.
For a standard implementation of this model, you can draw from a random uniform number generator
of the unit interval, and evaluate whether those draws are above or below :math:`{\rm Prob(\ quenched)}`.

Alternatively, to implement CAM you would compute
:math:`p={\rm Prob(< r/R_{vir}}\ \vert\ M_{\rm host})` for each simulated subhalo,
and then evaluate whether each :math:`p`
is above or below :math:`{\rm Prob(\ quenched}\ \vert\ M_{\rm host})`.
This technique lets you generate a series of mocks with exactly the same
:math:`{\rm Prob(\ quenched}\ \vert\ M_{\rm host})`,
but with tunable levels of quenching gradient, ranging from zero gradient
to the statistical extrema.
The `~halotools.utils.sliding_conditional_percentile` function can be used to
calculate :math:`p={\rm Prob(< r/R_{vir}}\ \vert\ M_{\rm host}).`


The plot below demonstrates three different mock catalogs made with CAM in this way.
The left hand plot shows how the quenched fraction of satellites varies
with intra-halo position. The right hand plot confirms that all three mocks have
statistically indistinguishable "halo mass quenching", even though their gradients
are very different.

.. image:: /_static/quenching_gradient_models.png

The next plot compares the 3d clustering between these models.

.. image:: /_static/quenching_gradient_model_clustering.png

For implementation details, the code producing these plots
can be found in the following Jupyter notebook:

    **halotools/docs/notebooks/galcat_analysis/intermediate_examples/quenching_gradient_tutorial.ipynb**


Correlating Galaxy Disk Size with Halo Spin
===========================================

In models where a galaxy's disk acquires roughly the same specific angular momentum
as its dark matter halo, there arises a correlation between halo spin and disk size.
At fixed disk mass, the observed distribution of disk size is roughly log-normal
with 0.2 dex of scatter. To empirically model the scale length
or half-light radius of the disk, a simple model would be to
draw from a log-normal distribution, such that the stochasticity in the
Monte Carlo realization is correlated with the halo spin,
so that above-average draws from the log-normal distribution of sizes
will get mapped onto halos with above-average values of spin, and conversely.


So in this application, CAM introduces a correlation between
:math:`{\rm Prob(<R_{1/2}\vert M_{\ast})}` and
:math:`{\rm Prob(<\lambda_{spin}\vert M_{halo})}.`
The figure below illustrates the result of such a model.

In the left panel, we see a scatter plot of the size-mass relation.
Halos in the bottom quartile of :math:`\lambda_{\rm spin}` host galaxies
with the smallest half-light radius for their stellar mass; halos in the top quartile
host the largest galaxies for their mass, and likewise for the middle quartiles.

In the right panel we show the average disk size as function of halo mass.
The model shown in the left panel is shown as the solid curves in the right panel.
In the model illustrated with the dashed curves,
we have introduced scatter in the correlation between
:math:`\lambda_{\rm spin}` and :math:`R_{1/2}`.

.. image:: /_static/size_mass_spin.png

For implementation details, the code producing these plots
can be found in the following Jupyter notebook:

    **halotools/docs/notebooks/galcat_analysis/intermediate_examples/demo_cam_disk_sizes.ipynb**




