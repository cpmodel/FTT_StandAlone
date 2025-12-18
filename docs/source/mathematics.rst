Mathematical formulation
===================================================

Scope
-----

This document describes the mathematical structure of the **Future Technology Transformations (FTT)**
model family used in energy–economy–environment modelling. The focus is on the core diffusion
equations and their derivation. 

FTT formalises technology transitions as an evolutionary competition process under bounded
rationality and increasing returns. Its mathematical structure is designed to study transition
dynamics, lock-in, and policy-driven tipping behaviour rather than optimal equilibrium outcomes.


Technology Shares
-----------------

Consider a sector with :math:`N` competing technologies. Each technology :math:`i` has a market share

.. math::

   S_i(t) \in [0,1], \qquad \sum_{i=1}^N S_i(t) = 1.

Technology diffusion is modelled as **substitution between technologies**, not as optimisation of
total system cost.


Generalised Costs and Heterogeneity
----------------------------------

Each technology has a generalised cost :math:`C_i`, including:

- investment,
- fuel, operation and maintenance costs,
- non-pecuniary preferences,
- policy effects (taxes, subsidies).

Agents are heterogeneous. Costs are therefore represented by probability distributions rather than
point values. Let :math:`F_i(C)` denote the cumulative distribution of perceived costs for technology
:math:`i`.


Pairwise Discrete Choice
-----------------------

Agents compare technologies pairwise. The probability that technology :math:`i` is preferred to
:math:`j` is

.. math::

   P_{i>j} = \int F_i(C) \, dF_j(C).

Under standard assumptions on the cost distributions, this probability can be expressed as a smooth
function of cost differences

.. math::

   P_{i>j} = \Phi(C_j - C_i),

where :math:`\Phi` is typically a logistic or normal cumulative distribution function.


Substitution Dynamics
---------------------

Market share changes arise from bilateral substitution flows. The evolution of :math:`S_i` is

.. math::

   \frac{dS_i}{dt} = \sum_{j \neq i} S_i S_j
   \left( A_{ij} F_{ij} - A_{ji} F_{ji} \right).

Two conceptually distinct matrices appear in this formulation.

**Substitution matrix (A)**  
The matrix :math:`A_{ij}` governs the *rate at which substitutions are possible*. It depends on:

- technology lifetimes and scrappage rates,
- relative speed of capacity expansion between technologies,

The matrix :math:`A` therefore controls *how fast* technologies can replace one another, independently
of their relative attractiveness. :math:`A` is defined as :math:`A_{ij} = \frac{K}{\tau_i t_j}`, where

- :math:`K` is a normalisation constant setting the overall time scale of diffusion, 
- :math:`\tau_i` is the lifetime of technology :math:`i`,
- :math:`t_j` is the build time of technology :math:`j`.

**Preference matrix (F)**  
The matrix :math:`F_{ij}` represents *agent preferences* derived from cost comparisons. It is defined as

.. math::

   F_{ij} = P_{i>j} = \Phi(C_j - C_i),

where :math:`C_i` and :math:`C_j` are generalised costs and :math:`\Phi` is a cumulative distribution
function reflecting cost heterogeneity across agents.

The matrix :math:`F` therefore controls *which direction* substitution occurs, based on relative
costs and preferences.


Dynamic Properties
------------------

The FTT system is:

- non-linear due to the :math:`S_i S_j` interaction terms;
- path-dependent, with outcomes contingent on initial conditions;
- not derivable from a global welfare or cost minimisation problem.


Learning-by-Doing
-----------------

Technology costs decline endogenously with cumulative deployment. For technology :math:`i`

.. math::

   C_i(t) = C_i(0) \left( \frac{W_i(t)}{W_i(0)} \right)^{-b_i},

where :math:`W_i` is cumulative installed capacity and :math:`b_i` is a technology-specific learning exponent, based on literature values.

Learning feeds back into the choice probabilities :math:`P_{i>j}`, creating increasing returns and
reinforcing early adoption advantages.


Constraints
-----------

FTT includes physical and institutional constraints such as:

- limits on resource availability (e.g. scrap for steel),
- infrastructure constraints (e.g. grid capacity for power),
- technology constraints (e.g. restrictions on heating water with air-to-air heat pumps)

Mathematically, these enter as bounds on :math:`S_i` or as modifiers of the substitution and preference matrices
:math:`A_{ij}` or :math:`F_{ij}`.


Model Coupling
--------------

FTT technology modules are coupled to macro-econometric models. Energy demand affects the markets, while technology diffusion (including prices and investment) feeds back
into the macroeconomic system.



