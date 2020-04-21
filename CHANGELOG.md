## Changelog

This is the changelog for the rescomp package

### rescomp 0.1.3 - Current Master

* Rewrote the documentation for gitlab pages
* Rewrote the examples as jupyter files for integration in the new documenation

### rescomp 0.1.2

* Added options to have W_in ordered 
* Implemented squared tanh activation function
* Implemented option to have mixture of activation functions
* Changed the 'b' parameter in the lorenz systems used in 
  rescomp.simulate_trajectory() to 'beta', as that's what the parameter is 
  usually called in the literature

### rescomp 0.1.1

* Allowed kwargs for the Lorenz63 and Roessler systems in simulate_trajectory()
* Added short example for the simulations module usage.
* Rewrote divergence_time to correctly handle perfect predictions and 
  immediate divergence


### rescomp 0.1.0

* Package fundamentally rewritten.  
  See README, HTML documentation and examples for details.

