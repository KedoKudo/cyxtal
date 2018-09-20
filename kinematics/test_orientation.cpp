#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include "orientation.h"

using namespace std;

int main(){

    /// constructor testing
    double qvec[4] = {1, 0, 0, 0};
    string lattice = "bcc";
    Orientation o1 = Orientation(qvec, lattice);
    o1.Print();

    double angd = 45;  // in deg
    double angr = angd / 180 * M_PI;
    double axis[3] = {1, 2, 1};
    Orientation o2d = Orientation(angd, axis, lattice, true);
    o2d.Print();
    Orientation o2r = Orientation(angr, axis, lattice, false);
    o2r.Print();

    double eulersd[3] = {45, 90, 0};
    double eulersr[3] = {M_PI/4, M_PI/2, 0};
    Orientation oed = Orientation(eulersd, lattice, true, true);
    oed.Print();
    Orientation oer = Orientation(eulersr, lattice, true, false);
    oer.Print();

    return 0;
}