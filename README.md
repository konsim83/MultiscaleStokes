# C++ Protoype of a 3D-Dynamical Core on a Hypershell 


This C++ code serves as a playground to experiment with multiscale methods for advection-dominated systems on spherical shells. The current prototype is a bouyancy Boussinesq system, i.e., an incompressible Navier-Stokes system with Coriolis force that is driven through density variations caused by temperature changes.

It is planned to use stable (multiscale) reconstruction FEMs to investigate the feedback of small-scale parametrizations on 3D dynamical cores in the long run and to replace standard saddle point discretizations with different formulations using stable pairs of elements constructed in FEEC.