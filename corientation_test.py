#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

import time,sys
import numpy as np
from cyxtal import Quaternion, Orientation
#import damask.orientation.Orientation as pOrientation

###########################xs
# Testing Quaternion class
print "*"*10+"Test Quaternion"+"*"*10

a = Quaternion.fromRandom()
b = Quaternion.fromRandom()
c = Quaternion(a)

print "a:{}\nb:{}\nc:{}\n".format(a,b,c)
print "a**2 : {}".format(a**2)
print "a/=2 : {}".format(a/2)
print "a+b  : {}".format(a+b)
print "a-b  : {}".format(a-b)
print "a*b  : {}".format(a*b)
print "-a   : {}".format(-a)
print "|a|  : {}".format(abs(a))
print "|a|^2: {}".format(a.magnitude_squared())
print "a==b : {}".format(a==b)
print "a!=c : {}; a==c : {}".format(a!=c, a==c)
print "Identity:{}".format(Quaternion.fromIdentity())
print "a inversed: {}".format(a.inverse())
print "as list: {}".format(a.asList())


############################
# Testing for Symmetry class
print "*"*10+"Test Symmetry"+"*"*10


############################
# Testing for Orientation class
print "*"*10+"Test Orientation"+"*"*10
a = Orientation(quaternion=Quaternion([0.2230,  0.9533, 0.2004, 0.0374]),
                symmetry='hexagonal')
b = Orientation(quaternion=Quaternion([0.2662, -0.8247,-0.1579, 0.4734]),
                symmetry='hexagonal')
print np.degrees(a.disorientation(b).asEulers())


###################
# Testing for Speed
print "*"*10+"Test Speed"+"*"*10
start = time.clock()

N = int(sys.argv[1])  # testing size

data = [Orientation(random=True, symmetry='hexagonal') for i in range(N)]
ref = Orientation(random=True, symmetry='hexagonal')

for item in data:
    ref.disorientation(item)

runtime = time.clock() - start
print "Total runtime: {}s".format(runtime)

###################
# Testing IPF color
a = np.random.random((5,3)) * 180
co = [ Orientation(Eulers=np.radians(a[i]), symmetry=3).IPFcolor(np.array([0,1.0,0])) for i in range(5)]
#po = [pOrientation(Eulers=np.radians(a[i]), symmetry='hexagonal').IPFcolor(np.array([0,1.0,0])) for i in range(5)]
for i in range(5):
    print co[i]#, po[i]
