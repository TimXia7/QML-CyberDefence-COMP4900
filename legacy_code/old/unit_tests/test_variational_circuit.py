
# Some quick unit tests for all variational circuit related
# As of 2025-03-02, they're pretty simple as all we have is the circuit itself, -Tim

import unittest
import numpy as np
import sys
import os

# Import variational_circuit.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from variational_circuit import variational_circuit  

class TestVariationalCircuit(unittest.TestCase):

    # Test the circuit with boundary values (0 and 2 pi)
    # The result should be the same
    def test_circuit_boundary_values(self):
        result = variational_circuit(0, 0)
        self.assertAlmostEqual(result, 1.0)

        # Test with theta_x = 2pi and theta_y = 2pi
        result = variational_circuit(2*np.pi, 2*np.pi)
        self.assertAlmostEqual(result, 1.0, delta=0.1)  # should be near 1 (ket 0)

    # Test that base values of pi/2 equal 0, and pi/4 doesn't, as they math matically should
    def test_circuit_zero(self):
        theta_x = np.pi / 2  
        theta_y = np.pi / 2  
        result = variational_circuit(theta_x, theta_y)
        self.assertEqual(result, 0)  
    def test_circuit_non_zero(self):
        theta_x = np.pi / 4  
        theta_y = np.pi / 4  
        result = variational_circuit(theta_x, theta_y)
        self.assertNotEqual(result, 0)  

    # with an array of random values between 0 and 2pi, ensure the value is always within -1 and 1
    def test_circuit_within_range(self):
        # Random value of 300, cause, why not
        for _ in range(300):
            theta_x = np.random.uniform(0, 2 * np.pi) 
            theta_y = np.random.uniform(0, 2 * np.pi)  
            result = variational_circuit(theta_x, theta_y)
            self.assertGreaterEqual(result, -1) 
            self.assertLessEqual(result, 1) 

if __name__ == '__main__':
    unittest.main()
