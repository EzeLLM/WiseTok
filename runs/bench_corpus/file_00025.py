# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 00:33:08 2015

@author: Fahim
"""

import unittest
import numpy as np
import numpy.testing as nptest
from Intersectable import Sphere
from Ray import Ray, IntersectionResult
import GeomTransform as GT
from TesterCommon import test_intersection_with_result, test_no_intersection

class TestSphereAtOriginNoIntersectionWithRay(unittest.TestCase):
    ''' Test basic cases where the ray doesn't intersect the sphere '''
    def setUp(self):
        self.center = [0., 0., 0.]
        self.radius = 1.0
        self.sphere = Sphere({'center': self.center, 'radius': self.radius})
        
    def test_sphere_creation(self):
        ''' sanity check '''
        nptest.assert_array_equal(self.sphere.center, self.center)
        nptest.assert_equal(self.sphere.radius, self.radius)
        
    def test_no_intersection_opposite(self):
        ''' intersection at negative distance '''
        origin = [0, 0, 10]
        direction = [0, 0, 1]
        test_no_intersection(self.sphere, origin, direction)
        
    def test_no_intersection_miss_v0(self):
        origin = [0, 0, 10]
        direction = [1, 0, 0]
        test_no_intersection(self.sphere, origin, direction)
        
    def test_no_intersection_miss_v1(self):
        test_no_intersection(self.sphere, origin=[0, 0, 10], direction=[5, 0, -10])
        
    def test_ray_originates_on_sphere(self):
        test_no_intersection(self.sphere, origin=[1, 0, 0], direction=[1, 0, 0])
        
class TestSphereAtOriginIntersection(unittest.TestCase):
    ''' Tests intersection with a sphere centered at the origin '''
    def setUp(self):
        self.center = [0., 0., 0.]
        self.radius = 1.0
        self.sphere = Sphere({'center': self.center, 'radius': self.radius})
        
    def test_sphere_creation(self):
        ''' sanity check '''
        nptest.assert_array_equal(self.sphere.center, self.center)
        nptest.assert_equal(self.sphere.radius, self.radius)
        
    def test_basic_ray_intersection(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        eye = [0, 0, 10]
        viewDir = [0, 0, -1]
        test_intersection_with_result(self.sphere, origin=eye, 
                                      direction=viewDir,
                                      isect_pt = [0., 0., 1.], 
                                      isect_normal= [0., 0., 1.],
                                      isect_dist = 9.0)

    def test_intersection_pos_diag_ray(self):
        ''' Shoots a ray from some point not on the axis ''' 
        eye = [1, 0, 1]
        viewDir = [-1, 0, -1]
        test_intersection_with_result(self.sphere, origin=eye, 
                                      direction=viewDir,
                                      isect_pt = [np.cos(np.pi/4.), 0., np.sin(np.pi/4.)], 
                                      isect_normal= [np.cos(np.pi/4.), 0., np.sin(np.pi/4.)],
                                      isect_dist = np.sqrt(2.) - self.radius)
    
    def test_intersection_neg_diag_ray(self):
        ''' Shoots a ray from some point not on the axis from the opposite 
        direction i.e. neg eye coord. The points and normals should have
        opposite sign but the distance from eye should be the same ''' 
        eye = [-1, 0, -1]
        viewDir = [1, 0, 1]
        test_intersection_with_result(self.sphere, origin=eye, 
                                      direction=viewDir,
                                      isect_pt = [-np.cos(np.pi/4.), 0., -np.sin(np.pi/4.)], 
                                      isect_normal= [-np.cos(np.pi/4.), 0., -np.sin(np.pi/4.)],
                                      isect_dist = np.sqrt(2.) - self.radius)

    def test_intersection_pos_diag_ray_consistency(self):
        eye = [10, 0, 10]
        viewDir = [-1, 0, -1]
        
        ray = Ray(eye, GT.normalize(viewDir))
        
        
        result0 = self.sphere.intersect(ray)
        #print(result0)
        
        eye = [5, 0, 5]
        viewDir = [-1, 0, -1]
        
        ray = Ray(eye, GT.normalize(viewDir))
        
        result1 = self.sphere.intersect(ray)
        #print(result1)        
        
        nptest.assert_array_almost_equal(result0.p, result1.p)
        nptest.assert_array_almost_equal(result0.n, result1.p)
        nptest.assert_almost_equal(result0.t, 10 * np.sqrt(2.) - 1)    
        nptest.assert_almost_equal(result1.t, 5 * np.sqrt(2.) - 1)    
    
    def test_rand_eye_intersection_distance_correctness(self, NTEST = 100):    
        ''' Shoot rays from random eye position towards the center. The intersection
        point should be radius distance from the center. '''
        np.random.seed(9125)
        for testNo in range(NTEST):
            # generate a random point outside of the range [-5, 5]
            sgn = 1 if np.random.rand(1) <= .5 else -1
            
            eye = (np.random.rand(1, 3).flatten() + 5) * sgn # generate a random point
            viewDir = GT.normalize(-eye) # direction towards the center

            #print(sgn, eye, viewDir)
            
            ray = Ray(eye, viewDir)
                        
            result = self.sphere.intersect(ray)
            distance = np.linalg.norm(result.p - self.center)
            nptest.assert_almost_equal(distance, self.radius)

    def test_rand_eye_intersection_distance_from_eye_correctness(self, NTEST = 100):    
        ''' Shoot rays from random eye position towards the center. Test whether
        the distance to the intersection point from the eye position is distance
        to the center minus the radius.'''
        np.random.seed(9125) # uses the same random sequence as point to center distance test
        for testNo in range(NTEST):
            # generate a random point outside of the range [-5, 5]
            sgn = 1 if np.random.rand(1) <= .5 else -1
            
            eye = (np.random.rand(1, 3).flatten() + 5) * sgn # generate a random point
            viewDir = GT.normalize(-eye) # direction towards the center

            #print(sgn, eye, viewDir)
            
            ray = Ray(eye, viewDir)
                        
            result = self.sphere.intersect(ray)
            isect_pt_from_eye_distance = np.linalg.norm(result.p - eye) # for sanity check
            eye_distance_from_center = np.linalg.norm(eye - self.center)
            #print(isect_pt_from_eye_distance, result.t)
            nptest.assert_almost_equal(result.t, isect_pt_from_eye_distance) # sanity check
            nptest.assert_almost_equal(result.t, eye_distance_from_center - self.radius) # sanity check
            
    def test_ray_intersection_inside_sphere(self):
        ''' eye is inside the sphere and ray intersect the inside of the sphere '''
        test_intersection_with_result(self.sphere, origin=[0, 0, 0], 
                                      direction=[0, 0, 1],
                                      isect_pt = [0, 0, 1], 
                                      isect_normal=[0, 0, 1],
                                      isect_dist = self.radius)

    def test_ray_touches_sphere(self):
        ''' test when ray just touches the sphere '''
        origin = [1, 0, 5]
        test_intersection_with_result(self.sphere, origin=origin, 
                                      direction=[0, 0, -1],
                                      isect_pt = [1, 0, 0], 
                                      isect_normal=[1, 0, 0],
                                      isect_dist = origin[2])
        
    def test_ray_originates_on_sphere_goes_inside(self):
        test_intersection_with_result(self.sphere, origin=[1, 0, 0], 
                                      direction=[-1, 0, 0],
                                      isect_pt = [-self.radius, 0, 0], 
                                      isect_normal=GT.normalize([-self.radius, 0, 0]),
                                      isect_dist = 2. * self.radius)
                                      
    def test_ray_originates_on_sphere_goes_inside_v2(self):
        test_intersection_with_result(self.sphere, origin=[1, 0, 0], 
                                      direction=[-1, 1, 0],
                                      isect_pt = [0, 1, 0], 
                                      isect_normal=GT.normalize([0, 1, 0]),
                                      isect_dist = np.sqrt(2.))
               
 
def main(): # to make it easier to import this file and run the tests
    unittest.main()
    
if __name__ == '__main__':
    main()
    