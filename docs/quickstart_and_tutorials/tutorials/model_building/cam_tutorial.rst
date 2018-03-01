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
Once the CDF is specified, the standard Monte Carlo technique is to generate
a realization of a random uniform distribution and pass those draws to the
CDF inverse,  :math:`{\rm CDF}^{-1}(p),` which evaluates to the variable
:math:`x` being painted on the model galaxies.
See Section 3.7 of `The AstroML textbook <http://www.astroml.org/>`_
and `Transformation of Probability tutorial <https://github.com/jbailinua/probability/>`_
for more information about the inverse transformation sampling technique.

CAM introduces correlations between the
galaxy property :math:`x` and some halo property :math:`h,`
without changing :math:`{\rm CDF}(x)`. Rather than evaluating :math:`{\rm CDF}^{-1}(p)`
with random uniform variables,
instead you evaluate with :math:`p = {\rm CDF}(h) = {\rm Prob}(< h),`
introducing a monotonic correlation between :math:`x` and :math:`h`.

The "Conditional" part of CAM is that this technique naturally generalizes to
introduce a galaxy property correlation while holding some other property fixed.
Age Matching in `Hearin and Watson 2013 <https://arxiv.org/abs/1304.5557/>`_
is an example of this: the distribution :math:`{\rm Prob}(<SFR\vert M_{\ast})`
is modeled by correlating draws from the observed distribution with
:math:`{\rm Prob}(<\dot{M}_{\rm sub}\vert M_{\rm sub})` in a simulation,
so that galaxies which have
large SFR for their stellar mass are associated with subhalos that have
large mass accretion rates for their mass dark matter mass.

Bin-based vs. bin-free methods
------------------------------

See :ref:`bin_free_cam_tutorial` or :ref:`bin_based_cam_tutorial`

Implementing correlations of intermediate strength
--------------------------------------------------
This still needs to be written

The function `~halotools.empirical_models.noisy_percentile` can be used to
add controllable levels of noise to :math:`p = {\rm CDF}(h).`
This allows you to control the correlation coefficient
between :math:`x` and :math:`h,`
always exactly preserving the 1-point statistics of the output distribution.


