#include "orientation.h"

#include <stdlib.h>
#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

Orientation::Orientation(double quaternionVec[4], string crystalLattice){
    quaternion = Quaterniond(quaternionVec[0],
                             quaternionVec[1], 
                             quaternionVec[2],
                             quaternionVec[3]);
                            
    lattice = crystalLattice;
}

/**
 * print to console 
 */
void Orientation::Print(){
    cout << "Lattice: " << lattice << endl;
    printf("%f + %fi + %fj + %fk\n",quaternion.w(), 
                                    quaternion.x(),
                                    quaternion.y(),
                                    quaternion.z());
}