#ifndef __CYXTAL_ORIENTATION_H
#define __CYXTAL_ORIENTATION_H

#include <stdlib.h>
#include <string>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class Orientation{
    protected:
    Quaterniond quaternion;  // use quaternion to represent orientation
    string      lattice;     // given lattice structure

    public:
    Orientation(double quaternionVec[4],      string crystalLattice);

    Orientation(double eulers[3],             string crystalLattice, 
                bool isBunge, bool inDegree);

    Orientation(double angle, double axis[3], string crystalLattice, bool inDegree);

    Orientation(double rotationMatrix[3][3],  string crystalLattice);

    ~Orientation(){};

    //output to terminal 
    void Print();

};

#endif