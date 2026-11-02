"""Helpers for the Riemannian flow-matching toy notebooks (annulus and torus).

Modules
-------
geometry
    Base class shared by the toy domains.
annulus, torus
    Sampling, projection, velocity guards, domain diagnostics and drawing for each domain.
kernels
    Kernel-profile comparison: fit, field diagnostics, guarded/unguarded trajectories, refinement.
flows
    CellFlow helpers: toy AnnData, OT matching, training, prediction, learned fields and paths.
plotting
    Figures shared by both notebooks; domain-specific drawing is delegated to the geometry.
comparison
    The four-way Riemannian/Euclidean FM × biharmonic/Euclidean coupling comparison.
loss_modes
    RFM final-point vs. trajectory loss: single run and multi-seed experiment.

Run the notebooks from ``docs/notebooks`` so that this package is importable.
"""
