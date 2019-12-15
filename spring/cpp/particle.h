// This is the particle class
// It provides features to the particle and also lists how their positions and velocities are updated

#ifndef _PARTICLE_H
#define _PARTICLE_H

#include "vector3d.h"

class PARTICLE 
{
  public:

  // members
  int id;		// id of the particle
  double diameter;	// diameter of the particle
  double m; 		// mass of the particle
  VECTOR3D posvec;	// position vector of the particle
  VECTOR3D velvec;	// velocity vector of the particle
  VECTOR3D forvec;	// force vector on the particle
  double pe;		// potential energy
  double ke;		// kinetic energy
  double energy;	// energy
  
  // member functions
  
  // make a particle	// this function in C++ is known as a constructor
  PARTICLE(int initial_id = 1, double initial_mass = 1.0, VECTOR3D initial_position = VECTOR3D(0,0,0))
  {
    id = initial_id;
    m = initial_mass;
    posvec = initial_position;
  }
  
  // the next two functions are central to the velocity-Verlet algorithm
  // dt is taken from outside; it is the timestep derived from the steps that the user supplies
  // update position of the particle
  void update_position(double dt)		
  {
    posvec = ( posvec + (velvec ^ dt) );	// position updated to a full time-step
    return;
  }
  
  // update velocity of the particle
  void update_velocity(double dt)	
  {
    velvec = ( velvec + ( (forvec) ^ ( 0.5 * dt / m ) ) );	// notice the half time-step
    return;
  }
  
  // calculate kinetic energy of a particle
  void kinetic_energy()				
  {
    ke = 0.5 * m * velvec.GetMagnitude() * velvec.GetMagnitude();	// note that GetMagnitude function is a member of the VECTOR3D class and gets you the magnitude of a 3-D vector.
    return;
  }
};

#endif