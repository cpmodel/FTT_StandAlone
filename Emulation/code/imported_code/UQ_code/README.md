# RotateBasis

R code that finds a calibration-optimal basis for high-dimensional computer model output, by combining important patterns contained in the observations with leading directions of variability in the ensemble.

Based on paper https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1514306.

If you spot a bug or have queries about how to use RotateBasis, contact j.m.salter@exeter.ac.uk

## Files

rotation_functions.R - rotation functions

example.R - a worked toy example

general_help.R - general strategy for how to set up data, W, and use the key functions

FastHM.R - history matching high-dimensional output efficiently (see https://arxiv.org/pdf/1906.05758.pdf)

Gasp.R - wrappers for using RobustGasp emulators (note - for full functionality requires ExeterUQ https://github.com/BayesExeter/ExeterUQ, improvements to this and integration with ATI emulation code coming soon)
