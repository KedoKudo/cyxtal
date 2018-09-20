#include <iostream>
#include <string>
#include "orientation.h"

using namespace std;

int main(){

    double qvec[4] = {1, 0, 0, 0};
    string lattice = "bcc";

    Orientation o1 = Orientation(qvec, lattice);

    o1.Print();
    
    return 0;
}