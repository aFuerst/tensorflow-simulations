// This file contains the routine that computes the LJ force on the particle

using namespace std;

#include <vector>
#include "particle.h"
#include "simulationbox.h"

void update_forces(vector<PARTICLE>& ljatom, SIMULATIONBOX& box, double dcut) 
{
  long double r, r2;
  double r6, r12, d, d2, d6, d12;
  
  double elj = 1.0;
  
  vector<VECTOR3D> lj_atom_atom (ljatom.size(), VECTOR3D(0,0,0));
  VECTOR3D r_vec, flj;
  
  for (unsigned int i = 0; i < ljatom.size(); i++)
  {
    flj = VECTOR3D(0,0,0);
    for (unsigned int j = 0; j < ljatom.size(); j++)
    {
      if (j == i) continue;
      r_vec = ljatom[i].posvec - ljatom[j].posvec;
      
      // the next 6 lines take into account the periodic nature of the boundaries of our simulation box
      if (r_vec.x>box.lx/2) r_vec.x -= box.lx;
      if (r_vec.x<-box.lx/2) r_vec.x += box.lx;
      if (r_vec.y>box.ly/2) r_vec.y -= box.ly;
      if (r_vec.y<-box.ly/2) r_vec.y += box.ly;
      if (r_vec.z>box.lz/2) r_vec.z -= box.lz;
      if (r_vec.z<-box.lz/2) r_vec.z += box.lz;
      r = r_vec.Magnitude();
      d = 1; // recall that we are working in reduced units where the unit of length is the diameter of the particle
      r2 = (r_vec.Magnitude()) * (r_vec.Magnitude());
      d2 = d * d;
      if (r < dcut * d)
      {
			r6 = r2 * r2 * r2;
			r12 = r6 * r6;
			d6 = d2 * d2 * d2;
			d12 = d6 * d6;
			flj = flj + ( r_vec * ( 48 * elj * (  (d12 / r12)  - 0.5 *  (d6 / r6) ) * ( 1 / r2 ) ) );
      }
      else
			flj = flj + VECTOR3D(0,0,0);
    }
    lj_atom_atom[i] =  flj;
  }
  
  // force on the particle stored
  for (unsigned int i = 0; i < ljatom.size(); i++)
    ljatom[i].forvec = lj_atom_atom[i];

  return; 
}