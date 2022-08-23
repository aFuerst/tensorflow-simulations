# Top

## Issue:

The kinetic energy/velocities/temperature of the system grow too high and escape the z-boundaries. 
This will eventually cause `tf.bincount` to crash in `bin.py`

I believe the problem is either in the focres calculation being too high at each iteration, or the thermostats are failing to reduce the temperature correctly/sufficiently.

You can use the plot scripts to track the temperature of the system for debugging.

## General Debugging:

* pass `--validate` to have it throw an exception if a particle escapes the z-axis or temperature goes too high.
* Have it output data after every iteration with `--freq 1`
* Pipe the the stdout and stderr to a file with `&>` 
```bash
python nano_init.py --freq 1 --steps 1000 --charge-density -0.001 --validate &> out.txt
```
* set the charge density to 0: `--charge-density 0.0`
* Many files have `if __name__ == "__main__"` to test and validate smaller segments of the system
* Plot the output data from thermostats and kinetic energy:
```bash
python plot_ke.py && python plot_therms.py &> therms.txt
```
* Add `print` statements to check tensorflow ops/shapes and verify what values are being passed to/from `session.run`

## TODOs:

* Bug 
* The potential energy functions have not been written
* A variety of `TODOs` are in the code about cleanup and efficiences that can be done
* Implement a cache for tensorflow constants as a possible performance boost (like in `wrap_distances_on_edges`, constants are duplicated on each call)
* Create conversion of particle position data to Ovito format
* Use auto-differentiation of energy calculation as force replacement
* Remove unnecessary `print` statements that are in code
* `wrap_vectorize` is probably unnecessary in `common.py`