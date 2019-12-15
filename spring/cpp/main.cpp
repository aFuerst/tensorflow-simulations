// This is main.
// This is particle dynamics or molecular dynamics (often called MD) simulation of a particle, myparticle, in a potential well (harmonic potential)
// It takes as input: mass, k (potential energy attribute, e.g. spring constant), total duration (time) of the particle trajectory, and time discretizations (slices)
// It outputs a file called myparticle.trajectory.out which stores the relevant data
// comments are entered with //

#include "particle.h"
#include "vector3d.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <sstream> 
#include <vector>

using namespace std;

void run_sim(double mass, double a, double k){
  double totaltime = 10; 
  int steps = 10000;
  // method specific
  double delta_t = totaltime / steps;

  PARTICLE myparticle;
  
  myparticle.m = mass;
  myparticle.posvec = VECTOR3D(a,0,0);
  myparticle.velvec = VECTOR3D(0,0,0);
  myparticle.kinetic_energy();
  myparticle.forvec = VECTOR3D(-k*a,0,0);
  myparticle.pe = 0.5*k*myparticle.posvec.x*myparticle.posvec.x;
  myparticle.energy = myparticle.ke + myparticle.pe;
  
  cout << endl;
  cout << "mass is " << myparticle.m << endl;
  cout << "k (associated with potential) is " << k << endl;
  cout << "initially particle is at " << myparticle.posvec.x << endl;
  cout << "total propagation time is " << totaltime << endl;
  cout << "choosing " << steps << " discretizations (slices) of time" << endl;
  cout << "time-step " << delta_t << endl;
    
  std::stringstream fname;
  fname << "/extra/alfuerst/sim/spring/" << mass << "_" << k << "_" << a << ".txt";
  string name = fname.str();
  ofstream trajectory(name.c_str(), ios::out);
  trajectory << "#" << setw(15) << "position" << setw(15) << "kinetic" << setw(15) << "potential" << setw(15) << "total energy" << setw(15) << "velocity" << endl;
  trajectory << 0 << setw(15) << myparticle.posvec.x << setw(15) << myparticle.ke << setw(15) << myparticle.pe << setw(15) << myparticle.energy << setw(15) << myparticle.velvec.x << endl;
  
  // Particle Dynamics (using velocity Verlet algorithm)
  for (int num = 1; num < steps; num++)
  {
    // velocity-Verlet
    myparticle.update_velocity(delta_t); // update velocity half timestep
    myparticle.update_position(delta_t); // update position full timestep
    myparticle.forvec.x = -k*myparticle.posvec.x; // recalculate force
    myparticle.update_velocity(delta_t); // update velocity next half timestep
    myparticle.kinetic_energy(); //compute kinetic energy
    myparticle.pe = 0.5*k*myparticle.posvec.x*myparticle.posvec.x; // record the potential energy
    myparticle.energy = myparticle.ke + myparticle.pe;
    trajectory << num*delta_t << setw(15) << myparticle.posvec.x << setw(15) << myparticle.ke << setw(15) << myparticle.pe << setw(15) << myparticle.energy << setw(15) << myparticle.velvec.x << endl;
  }
  
  trajectory.close();
}

int main(int argc, char* argv[]) 
{
  cout << "\nProgram starts\n";
  
  double mass = 1;
  // potential energy param
  double k = 1;
  double a = -1;
  
  for(double mass=.5; mass < 10; mass += .1){
    for(double a=-5; a < 5; a += .1){
      for(double k=.5; k < 10; k += .1){
        run_sim(mass, a, k);
      }
    }
  }

  cout << "Program ends \n\n";
  
  return 0;
} 
// End of main
