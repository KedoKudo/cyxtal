#include "orientation.h"

#include <stdlib.h>
#include <iostream>
#include <string>
#define _USE_MATH_DEFINES
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

Orientation
::Orientation(double quaternionVec[4], 
              string crystalLattice){
    quaternion = Quaterniond(quaternionVec[0],
                             quaternionVec[1], 
                             quaternionVec[2],
                             quaternionVec[3]);
                            
    lattice = crystalLattice;
}

Orientation
::Orientation(double angle, double axis[3], 
              string crystalLattice,
              bool inDegree){
    lattice = crystalLattice;
    
    double ang = ((inDegree) ? angle = angle / 180.0 * M_PI : angle);

    double axisNorm = 0.0;
    for (int i=0; i<3; i++){
        axisNorm += pow(axis[i], 2);
    }
    axisNorm = sqrt(axisNorm);

    quaternion = Quaterniond(cos(angle/2.0), 
                             sin(angle/2.0) * axis[0] / axisNorm,
                             sin(angle/2.0) * axis[1] / axisNorm,
                             sin(angle/2.0) * axis[2] / axisNorm);
}

/**
 * Contruct orientation from Euler angles in metallurgy
 * Note:
 *      Bunge Euler angles order used here is zxz.
 */
Orientation
::Orientation(double eulers[3],  
              string crystalLattice, 
              bool isBunge, bool inDegree){
    lattice = crystalLattice;

    double ang;
    double c[3], s[3];  // cos and sin of half angle

    for(int i=0; i<3; i++){
        ang = (inDegree)? eulers[i]/180*M_PI : eulers[i];
        c[i] = cos(ang/2.0);
        s[i] = sin(ang/2.0);
    }

    // either Bunge (zxz) or the other default type
    quaternion = (isBunge)?  Quaterniond(  
                                c[0] * c[1] * c[2] - s[0] * c[1] * s[2],
                                c[0] * s[1] * c[2] + s[0] * s[1] * s[2],
                              - c[0] * s[1] * s[2] + s[0] * s[1] * c[2], 
                                c[0] * c[1] * s[2] + s[0] * c[1] * c[2]
                             ) :
                             Quaterniond(
                                 c[0] * c[1] * c[2] - s[0] * s[1] * s[2],
                                 s[0] * s[1] * c[2] + c[0] * c[1] * s[2],
                                 s[0] * c[1] * c[2] + c[0] * s[1] * s[2],
                                 c[0] * s[1] * c[2] - s[0] * c[1] * s[2]
                             );
}

/**
 * print to console 
 */
void Orientation
::Print(){
    cout << "Lattice: " << lattice << endl;
    printf("%f + %fi + %fj + %fk\n",quaternion.w(), 
                                    quaternion.x(),
                                    quaternion.y(),
                                    quaternion.z());
}