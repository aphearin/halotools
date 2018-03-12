:orphan:

.. _cam_tutorial:

**********************************************************************
Tutorial on Conditional Abundance Matching
**********************************************************************

Conditional Abundance Matching (CAM) is a technique that you can use to
model a variety of correlations between galaxy and halo properties,
such as the dependence of galaxy quenching upon both halo mass and
halo formation time, or the dependence of galaxy disk size upon halo spin.
This tutorial explains CAM by applying the technique to a few different problems.


Basic Idea
=================

CAM is designed to answer questions of the following form:
*does halo property A correlate with galaxy property B?*
The Halotools approach to answering such questions is via forward modeling:
a mock universe is created in which the A--B correlation exists;
comparing the mock universe to the real one allows you to evaluate the
success of the A--B correlation hypothesis.

Forward-modeling the galaxy-halo connection requires specifying
some statistical distribution of the galaxy property being modeled,
so that Monte Carlo realizations can be drawn from the distribution.
CAM uses the most ubiquitous approach to generating Monte Carlo realizations,
*inverse transformation sampling,* in which the statistical distribution
is specified in terms of the cumulative distribution function (CDF),
:math:`{\rm CDF}(z) \equiv {\rm Prob}(< z).`
Briefly, the way this work is that once you specify the CDF,
you only need to generate a realization of a random uniform distribution,
and pass the values of that realization to the CDF inverse,  :math:`{\rm CDF}^{-1}(p),`
which evaluates to the variable :math:`z` being painted on the model galaxies.
See the `Transformation of Probability tutorial <https://github.com/jbailinua/probability/>`_
for pedagogical derivations associated with inverse transformation sampling,
and the `~halotools.utils.monte_carlo_from_cdf_lookup` function
for a convenient one-liner syntax.

In ordinary applications of inverse transformation sampling,
the use of a random uniform variable guarantees
that the output variables :math:`z` will be distributed according to
:math:`{\rm Prob}(z),` and that each individual :math:`z` will be purely stochastic.
CAM generalizes this common technique so that :math:`{\rm Prob}(z)`
is still recovered exactly, and moreover :math:`z` exhibits residual correlations
with some other variable, :math:`h`. Operationally, the way this works is that
rather than evaluating :math:`{\rm CDF}^{-1}(p)` with random uniform variables,
instead you evaluate with :math:`p = {\rm CDF}(h) = {\rm Prob}(< h),`
introducing a monotonic correlation between :math:`z` and :math:`h`.
In most applications, :math:`h` is some halo property like mass accretion rate,
and :math:`z` is some galaxy property like star-formation rate.
In this way, the galaxy property you paint on to your halos will
trace the distribution :math:`{\rm Prob}(z)`, such that above-average
values of :math:`z` will be painted onto halos with above average values of
:math:`h`, and conversely.

Finally, the "Conditional" part of CAM is that this technique naturally generalizes to
introduce a galaxy property correlation while holding some other property fixed.
For example, at fixed stellar mass, it is natural to hypothesize that
star-forming galaxies live in halos that are rapidly accreting mass,
and that quiescent galaxies live in halos that have already built up most of their mass.
In this kind of CAM application, we have:
:math:`{\rm Prob}(z)\rightarrow{\rm Prob}(<SFR\vert M_{\ast})`,
and :math:`{\rm Prob}(h)\rightarrow{\rm Prob}(<\dot{M}_{\rm sub}\vert M_{\rm sub})`.
That is, SFR at fixed stellar mass is hypothesized to correlate with
halo accretion rate at fixed (sub)halo mass.

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





