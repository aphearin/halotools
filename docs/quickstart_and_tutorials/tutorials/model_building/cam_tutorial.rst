:orphan:

.. _cam_tutorial:

**********************************************************************
Tutorial on Conditional Abundance Matching
**********************************************************************

Conditional Abundance Matching (CAM) is a technique that you can use to
model a variety of correlations between galaxy and halo properties,
such as the dependence of galaxy quenching upon both halo mass and
halo formation time. This tutorial explains CAM by applying
the technique to a few different problems.
Each of the following worked examples are independent from one another,
and illustrate the range of applications of the technique.


Basic Idea
=================

Forward-modeling the galaxy-halo connection requires specifying
some statistical distribution of the galaxy property being modeled,
so that Monte Carlo realizations can be drawn from the distribution.
The most convenient distribution to use for this purpose is the cumulative
distribution function (CDF), :math:`{\rm CDF}(x) = {\rm Prob}(< x).`
Once the CDF is specified, you only need to generate
a realization of a random uniform distribution and pass those draws to the
CDF inverse,  :math:`{\rm CDF}^{-1}(p),` which evaluates to the variable
:math:`x` being painted on the model galaxies.

CAM introduces correlations between the
galaxy property :math:`x` and some halo property :math:`h,`
without changing :math:`{\rm CDF}(x)`. Rather than evaluating :math:`{\rm CDF}^{-1}(p)`
with random uniform variables,
instead you evaluate with :math:`p = {\rm CDF}(h) = {\rm Prob}(< h),`
introducing a monotonic correlation between :math:`x` and :math:`h`.

The function `~halotools.empirical_models.noisy_percentile` can be used to
add controllable levels of noise to :math:`p = {\rm CDF}(h).`
This allows you to control the correlation coefficient
between :math:`x` and :math:`h,`
always exactly preserving the 1-point statistics of the output distribution.


The "Conditional" part of CAM is that this technique naturally generalizes to
introduce a galaxy property correlation while holding some other property fixed.
Age Matching in `Hearin and Watson 2013 <https://arxiv.org/abs/1304.5557/>`_
is an example of this: the distribution :math:`{\rm Prob}(<SFR\vert M_{\ast})`
is modeled by correlating draws from the observed distribution with
:math:`{\rm Prob}(<\dot{M}_{\rm sub}\vert M_{\rm sub})` in a simulation,
so that galaxies which have
large SFR for their stellar mass are associated with subhalos that have
large mass accretion rates for their mass dark matter mass.

Each of the sections below illustrates a different application of the same underlying method.
Each section has an accompanying annotated Jupyter notebook with the code used to generate the plots.

Satellite Galaxy Quenching Gradients
=====================================

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




