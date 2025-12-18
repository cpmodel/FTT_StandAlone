.. Main FTT documentation. 

Welcome to FTT
==========================

The **Future Technology Transformation** family of models simulate investor 
decision making, which capture the S-curve of technology adoption. It has several modules representing different sectors. 
This framework for the dynamic selection and diffusion of innovations was 
initially developed by J.-F. Mercure for the power sector `(Mercure, J.F., 2012) <https://www.sciencedirect.com/science/article/abs/pii/S0301421512005356>`_ , 
but now covers multiple sectors and industries:

- Power 
- Steel
- Cars
- Freight
- Residential Heating
- Industrial Heating:

  - Chemicals
  - Food, Beverages, and Tobacco
  - Non-Ferrous Metals, Machinery, and Transport Equipment
  - Non-Metallic Minerals
  - Other Industrial Sectors

The FTT models are based on a decision-making core for investors who must choose between 
a list of available technologies. 'Levelised' cost distributions (including capital and running costs) 
are fed into a set of pairwise comparisons, which are conceptually similar to a binary logit model.

The diffusion of technology follows a set of coupled non-linear differential equations, sometimes called 
'Lotka-Volterra' or 'replicator dynamics' equations, which represent the ability of larger or well 
established technologies to capture the market more effectively. The life expectancy of these technologies is also an important 
factor in determining the speed of transition.

Due to learning-by-doing and increasing returns to adoption, FTT results in path-dependent technology 
scenarios that arise from specific sectoral policies.

.. figure:: renewables_S_curve.webp
    :width: 600px
    :align: center
    :alt: frontend

    Example of S-curve technology adoption for renewables in FTT:Power `(Nijsse, 2023) <https://pypsa-earth.readthedocs.io/en/latest/index.html>`_   

=============
Documentation
=============

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   source/runningftt
   source/howto
   source/api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
