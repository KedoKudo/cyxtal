#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
   ________  ___  ___________    __
  / ____/\ \/ / |/ /_  __/   |  / /
 / /      \  /|   / / / / /| | / /
/ /___    / //   | / / / ___ |/ /___
\____/   /_//_/|_|/_/ /_/  |_/_____/

Copyright (c) 2016, C. Zhang.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DESCRIPTION
-----------
Standard geometry class for 2D/3D geometry calculation.
"""

import numpy as np


class Point(object):
    """
    DESCRIPTION
    -----------
    Point(x,y,z)
        Point in 3D space, base class provide bare bone abstraction
        for point related calculation.
    PARAMETERS
    ----------
    x,y,z: float
        Standard Cartesian coordinates for location description.
    coord: array
        Vector of the Cartesian coordinates.
    METHODS
    -------
    dist2point(Point other)
        Return the distance to another instance of Point.
    dist2line(Line other)
        Return the distance to given instance of Line.
    on_line(Line other)
        Whether the current instance lies on a given instance of Line.
    in_plane(Plane other)
        Whether the current point lies in a given instance of Plane.
    CLASSMETHOD
    -----------
    """
    def __init__(self, x, y, z):
        self._coord = [x, y, z]

    @property
    def x(self):
        return self._coord[0]

    @x.setter
    def x(self, val):
        self._coord[0] = val

    @property
    def y(self):
        return self._coord[1]

    @y.setter
    def y(self, val):
        self._coord[1] = val

    @property
    def z(self):
        return self._coord[2]

    @z.setter
    def z(self, val):
        self._coord[2] = val

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, val):
        if len(val) != 3:
            raise ValueError("Need 3 coordinates")
        self._coord = val

    def __str__(self):
        """String representation of Point"""
        return "(" + str(self._coord)[1: -1] + ")"

    def __eq__(self, other):
        if np.absolute(self.x - other.x) < 1e-6:
            if np.absolute(self.y - other.y) < 1e-6:
                if np.absolute(self.z - other.z) < 1e-6:
                    return True
        return False

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return 3

    def dist2point(self, other):
        """Return the distance to another point"""
        distance = (self.x - other.x)**2 + \
                   (self.y - other.y)**2 + \
                   (self.z - other.z)**2
        return np.sqrt(distance)

    def dist2line(self, line):
        """Return the distance to another line"""
        return line.dist2point(self)

    def on_line(self, line):
        """Quick test is the point is on the given line"""
        return line.contain_point(self)

    def in_plane(self, plane):
        """Quick test if a point is in a given plane"""
        return plane.contain_point(self)


class Line(object):
    """
    DESCRIPTION
    -----------
    Line(Point pt_0, Point pt_1)
        A line(segment) in 3D space defined with 2 Point instances.
    PARAMETERS
    ----------
    start_pt: Point
        Start point of the line instance.
    end_pt: Point
        End point of the line instance.
    length: float
        Return the length of the line segment.
    direction: numpy.array
        Return the direction vector of the line segment.
        [start_pt->end_pt]
    METHODS
    -------
    contain_point(Point pt)
        Test if self contains pt.
    parallel_to(Line other)
        Test if self is parallel to other.
    skewed_from(Line other)
        Test if self is skewed from other.
    intercepted_by(Line other)
        Test if self is intercepted by other.
    get_intercept(Line other)
        Return the intercept point.
    dist2point(Point pt)
        Return the distance between self and pt (shortest).
    dist2line(Line other)
        Return the distance between self and other (shortest).
    angle2line(Line other, inDegree=True)
        Return the angle between self and other.
    CLASSMETHOD
    -----------
    """
    def __init__(self, pt_start, pt_end):
        if pt_start == pt_end:
            raise ValueError("0 length line.")
        else:
            self._start = pt_start
            self._end = pt_end

    @property
    def start_pt(self):
        return self._start

    @start_pt.setter
    def start_pt(self, new_start):
        self._start = new_start

    @property
    def end_pt(self):
        return self._end

    @end_pt.setter
    def end_pt(self, new_end):
        self._end = new_end

    @property
    def length(self):
        temp = [self.start_pt.x - self.end_pt.x,
                self.start_pt.y - self.end_pt.y,
                self.start_pt.z - self.end_pt.z]
        result = temp[0]**2 + temp[1]**2 + temp[2]**2
        return np.sqrt(result)

    @property
    def direction(self):
        temp = [self.end_pt.x - self.start_pt.x,
                self.end_pt.y - self.start_pt.y,
                self.end_pt.z - self.start_pt.z]
        result = [float(item/self.length) for item in temp]
        return np.array(result)

    def __str__(self):
        return str(self.start_pt) + "-->" + str(self.end_pt)

    def __neg__(self):
        return Line(self.end_pt, self.start_pt)

    def __eq__(self, other):
        if self.start_pt == other.start_pt:
            if self.end_pt == other.end_pt:
                return True
        return False

    def __ne__(self, other):
        return not self == other

    def contain_point(self, point):
        """Test if self contains point"""
        if point == self.start_pt:
            return True  # special case of start point
        elif point == self.end_pt:
            return True  # special case of end point
        else:
            line1 = Line(point, self.start_pt)
            line2 = Line(point, self.end_pt)
            # when point online, the angle between
            # line1 and line2 should be 180 degree
            if np.dot(line1.direction, line2.direction) + 1 < 1e-4:
                return True
        return False

    def parallel_to(self, other):
        """Test if two Line are parallel to each other"""
        if 1 - np.absolute(np.dot(self.direction, other.direction)) < 1e-4:
            return True
        return False

    def skewed_from(self, other):
        """Quick test if one line is skewed from the other"""
        if self.parallel_to(other):
            return False
        elif self.contain_point(other.start_pt):
            return False  # intercepted at the end point
        elif self.contain_point(other.end_pt):
            return False  # intercepted at the end point
        else:
            normal = np.cross(self.direction, other.direction)
            normal = [item/np.linalg.norm(normal) for item in normal]
            test_line = Line(self.start_pt, other.start_pt)
            # test if two lines are coplanar --> intercept
            if np.absolute(np.dot(normal, test_line.direction)) < 1e-4:
                return False
            else:
                return True

    def intercepted_by(self, other):
        """Quick test if one line is intercepted by another"""
        return not self.get_intercept(other) is None

    def get_intercept(self, other):
        """Return the intercept point is exist, or return None"""
        if self.parallel_to(other) or self.skewed_from(other):
            return None
        elif self.contain_point(other.start_pt):
            return other.start_pt
        elif self.contain_point(other.end_pt):
            return other.end_pt
        else:
            pt_a = self.start_pt
            pt_b = self.end_pt
            pt_c = other.start_pt
            pt_d = other.end_pt
            matrix = np.array([[pt_b.x - pt_a.x, pt_c.x - pt_d.x],
                               [pt_b.y - pt_a.y, pt_c.y - pt_d.y],
                               [pt_b.z - pt_a.z, pt_c.z - pt_d.z]])
            vector = np.array([pt_c.x - pt_a.x,
                               pt_c.y - pt_a.y,
                               pt_c.z - pt_a.z])
            co_vector = np.dot(matrix.T, vector)
            co_matrix = np.dot(matrix.T, matrix)
            # use least-square to solve a overdetermined situation
            results = np.linalg.solve(co_matrix, co_vector)
            temp_pt = Point(pt_a.x + (pt_b.x - pt_a.x)*results[0],
                            pt_a.y + (pt_b.y - pt_a.y)*results[0],
                            pt_a.z + (pt_b.z - pt_a.z)*results[0])
            if self.contain_point(temp_pt) and other.contain_point(temp_pt):
                return temp_pt
            else:
                return None

    def dist2point(self, point):
        """Return the distance to a given point"""
        if self.contain_point(point):
            return 0.0
        else:
            temp_line = Line(point, self.start_pt)
            # find the normal of the plane defined by the point and line
            plane_normal = np.cross(temp_line.direction, self.direction)
            plane_normal = [item/np.linalg.norm(plane_normal)
                            for item in plane_normal]
            direction = np.cross(self.direction, plane_normal)
            direction = [item/np.linalg.norm(direction) for item in direction]
            result = temp_line.length * np.dot(temp_line.direction, direction)
            return np.absolute(result)

    def dist2line(self, other):
        """Return the distance between two skewed or parallel lines"""
        if self.parallel_to(other):
            if self.contain_point(other.start_pt):
                return 0.0  # two line intercept at start_pt
            elif self.contain_point(other.end_pt):
                return 0.0  # two line intercept at end_pt
            else:
                return self.dist2point(other.start_pt)
        elif self.skewed_from(other):
            normal = np.cross(self.direction, other.direction)
            normal = [item/np.linalg.norm(normal) for item in normal]
            test_line = Line(self.start_pt, other.start_pt)
            result = test_line.length * np.dot(test_line.direction, normal)
            return np.absolute(result)
        else:
            # self intercepts with other
            return 0.0

    def angle2line(self, other, inDegree=True):
        """Return angle between self and other"""
        angle = np.arccos(np.dot(self.direction, other.direction))
        if inDegree:
            angle = np.rad2deg(angle)
        return angle


class Plane(object):
    """
    DESCRIPTION
    -----------
    Plane(Point pt_1, Point pt_2, Point pt_3)
        A plane in 3D space defined with 3 points.
    PARAMETERS
    ----------
    pt_0, pt_1, pt_2: Point
        Three non collinear points defines the flat plane (self).
    normal: numpy.array
        Plane normal
    METHODS
    -------
    contain_point(Point point)
        Test if self contains pt.
    contain_line(Line l)
        Test if self contains l.
    parallel_to(Plane other)
        Test if self and other are parallel to each other.
    CLASSMETHOD
    -----------
    """
    def __init__(self, point1, point2, point3):
        # test if 3 points are on the same line
        if Line(point1, point2).parallel_to(Line(point2, point3)):
            raise ValueError("3 points are collinear ")
        self._point = [point1, point2, point3]

    @property
    def normal(self):
        """Plane normal"""
        normal = np.cross(Line(self._point[0], self._point[1]).direction,
                          Line(self._point[1], self._point[2]).direction)
        normal = [item/np.linalg.norm(normal) for item in normal]
        return np.array(normal)

    def __str__(self):
        out_string = "{}(x-{})+{}(y-{})+{}(z-{})=0".format(self.normal[0],
                                                           self._point[0].x,
                                                           self.normal[1],
                                                           self._point[0].y,
                                                           self.normal[2],
                                                           self._point[0].z)
        return out_string

    def __eq__(self, other):
        if 1 - np.absolute(np.dot(self.normal, other.normal)) < 1e-4:
            return other.contain_point(self._point[0])
        else:
            return False

    def contain_point(self, point):
        """Quick test to see if a point is in plane"""
        test_val = [point.x - self._point[0].x,
                    point.y - self._point[0].y,
                    point.z - self._point[0].z]
        if np.absolute(np.dot(test_val, self.normal)) < 1e-4:
            return True
        else:
            return False

    def contain_line(self, line):
        """Quick test to see if a line lies in a plane"""
        if self.contain_point(line.start_pt):
            if self.contain_point(line.end_pt):
                return True
        return False

    def parallel_to(self, other):
        """Quick test if two planes are parallel to each other"""
        if 1 - np.absolute(np.dot(self.normal, other.normal)) < 1e-4:
            return True
        else:
            return False


class Point2D(Point):
    """
    DESCRIPTION
    -----------
    Point2D(x,y)
        A 2D point (derived from the 3D Point class).
    PARAMETERS
    ----------
    METHODS
    -------
    CLASSMETHOD
    -----------
    """
    def __init__(self, x, y):
        super(Point2D, self).__init__(x, y, 0)

    def __len__(self):
        return 2

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Line2D(Line):
    """
    DESCRIPTION
    -----------
    Line2D(Point2D pt_start, Point2D pt_end)
        A 2D line (derived from the 3D Line class).
    PARAMETERS
    ----------
    METHODS
    -------
    get_discrete_pts(step=5)
        Return a numpy.array of coordinates discretize the 2D line.
    get_segments(step=5)
        Return a numpy.array of segments.
    CLASSMETHOD
    -----------
    """
    def __init__(self, pt_start, pt_end):
        """Using two 2D point to define a 2D line"""
        assert isinstance(pt_start, Point2D)
        assert isinstance(pt_end, Point2D)
        super(Line2D, self).__init__(pt_start, pt_end)

    def __str__(self):
        out_string = "(" + str(self.start_pt.x) + ", " + str(self.start_pt.y)
        out_string += ") --> ("
        out_string += str(self.end_pt.x) + ", " + str(self.end_pt.y) + ")"
        return out_string

    @property
    def direction(self):
        temp_vector = [self.end_pt.x - self.start_pt.x,
                       self.end_pt.y - self.start_pt.y]
        temp_vector = [item/np.linalg.norm(temp_vector)
                       for item in temp_vector]
        return temp_vector

    def parallel_to(self, other):
        vec_1 = self.direction
        vec_2 = other.direction
        if 1 - np.absolute(np.dot(vec_1, vec_2)) < 1e-4:
            return True
        else:
            return False

    @staticmethod
    def skewed_from(self, other):
        """2D lines do not skew from each other"""
        raise TypeError("2D line do not skew from each other")

    def get_intercept(self, other):
        """Return the intercept of two lines"""
        if self.parallel_to(other):
            return None
        elif self.contain_point(other.start_pt):
            return other.start_pt
        elif self.contain_point(other.end_pt):
            return other.end_pt
        else:
            pt_a = self.start_pt
            pt_b = self.end_pt
            pt_c = other.start_pt
            pt_d = other.end_pt
            matrix = np.array([[pt_a.y - pt_b.y, pt_b.x - pt_a.x],
                               [pt_c.y - pt_d.y, pt_d.x - pt_c.x]])
            vector = np.array([pt_b.x * pt_a.y - pt_a.x * pt_b.y,
                               pt_d.x * pt_c.y - pt_c.x * pt_d.y])
            results = np.linalg.solve(matrix, vector)
            temp_pt = Point2D(results[0], results[1])
            if self.contain_point(temp_pt) and other.contain_point(temp_pt):
                return temp_pt
            else:
                return None

    def get_discrete_pts(self, step=5):
        """ return a list of coordinates discretize the line """
        # get number of segments for current line
        step_size = int(self.length / float(step)) + 2
        t = np.linspace(0, 1, step_size)
        pts = []
        # chop, chop
        for item in t:
            x = self.start_pt.x + (self.end_pt.x - self.start_pt.x) * item
            y = self.start_pt.y + (self.end_pt.y - self.start_pt.y) * item
            pts.append((x, y))
        return np.array(pts)

    def get_segments(self, step=5):
        """ return a list of segments"""
        # This can be directly used by matplotlib
        pt_list = self.get_discrete_pts(step=step)
        segments = []
        for i in range(len(pt_list) - 1):
            segments.append([pt_list[i], pt_list[i+1]])
        return np.array(segments)


class Polygon2D(object):
    """
    DESCRIPTION
    -----------
    Polygon2D()
        A 2D polygon class.
    PARAMETERS
    ----------
    edges: list
        List of segments/edges of the 2D polygon.
    vertices: list
        List of 2D points serve as the vertices of the 2D polygon.
    center: Point2D
        Gravity center of the polygon.
    METHODS
    -------
    add_vertex(Point new_vtx)
        Add new vertex to self.
    get_shortest()
        Return the shortest distance between the center and vertices.
    contains_point(Point point, ray_origin=None)
        Test if given point lies inside self.
    CLASSMETHOD
    -----------
    """
    def __init__(self):
        """Initialize a 2D polygon with empty vertices list"""
        self.__vertices = []
        self.__ordered = False

    def __str__(self):
        """Formatted output for 2D polygon"""
        return "2D {}-Polygon".format(len(self.__vertices))

    @property
    def edges(self):
        if not self.__ordered:
            self.__update()  # use lazy-evaluation, only update when needed
        # compute edge list
        edge_list = []
        for i in range(len(self.__vertices) - 1):
            edge_list.append(Line2D(self.__vertices[i], self.__vertices[i+1]))
        edge_list.append(Line2D(self.__vertices[-1], self.__vertices[0]))
        return edge_list

    @property
    def vertices(self):
        if not self.__ordered:
            # use lazy-evaluation, only update when needed
            self.__update()
        return self.__vertices

    @property
    def center(self):
        """return the gravity center"""
        center_x = 0.0
        center_y = 0.0
        for vertex in self.__vertices:
            center_x += float(vertex.x)
            center_y += float(vertex.y)
        center_x /= len(self.__vertices)
        center_y /= len(self.__vertices)
        return Point2D(center_x, center_y)

    def add_vertex(self, point):
        """Add one more vertex to the current Polygon"""
        self.__vertices.append(point)
        self.__ordered = False

    def __update(self):
        point_list = []
        for vertex in self.__vertices:
            point_list.append((vertex.x, vertex.y))
        # build an ordered vertices list use convex_hull method
        self.__vertices = []
        for point in convex_hull(point_list):
            self.__vertices.append(Point2D(point[0], point[1]))
        self.__ordered = True

    def get_shortest(self):
        """return the shortest distance between the center and vertices"""
        center = self.center
        dist = Line2D(center, self.__vertices[-1]).length
        for vertex in self.__vertices[:-1]:
            temp = Line2D(center, vertex).length
            if temp < dist:
                dist = temp
        return dist

    def contains_point(self, point, ray_origin=None):
        """quick test if a Point2D instance is inside the polygon."""
        assert isinstance(point, Point2D)
        # First test if the point happens to be on the edges
        for edge in self.edges:
            if edge.contain_point(point):
                return True
        # now start with other settings
        if ray_origin is None:
            center = self.center
            temp_x = center.x + 10 * (self.__vertices[-1].x - center.x)
            temp_y = center.y + 10 * (self.__vertices[-1].y - center.y)
            test_point = Point2D(temp_x, temp_y)
            test_line = Line2D(test_point, point)
        else:
            assert isinstance(ray_origin, Point2D)
            test_line = Line2D(ray_origin, point)
        count = 0
        for edge in self.edges:
            if edge.intercepted_by(test_line):
                count += 1
        if count % 2 == 0:
            return False
        else:
            return True


def convex_hull(point_list):
    """
    DESCRIPTION
    -----------
    convex_hull(points)
        Computes the convex hull of a set of 2D points.
    PARAMETERS
    ----------
    points: list
        An iterable sequence of (x, y) pairs representing the points.
    RETURNS
    -------
        A list of vertices of the convex hull in counter-clockwise order,
        starting from the vertex with the lexicographically smallest
        coordinates. Implements Andrew's monotone chain algorithm.
        O(n log n) complexity.
    NOTES
    -----
    """
    # Sort the points lexicographically
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(point_list))
    # Boring case: no points or a single point, possibly repeated
    # multiple times.
    if len(points) <= 1:
        return points
    # 2D cross product of OA and OB vectors, i.e. z-component of their
    # 3D cross product. Returns a positive value, if OAB makes a
    # counter-clockwise turn, negative for clockwise turn, and zero if the
    # points are collinear.

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the
    # beginning of the other list.
    return lower[:-1] + upper[:-1]

# ----------- #
# END OF FILE #
# ----------- #
