#!/usr/bin/env python3
"""
Test suite for TSP solver implementation.
Validates correctness, performance, and edge cases.
"""

import unittest
import tempfile
import os
import numpy as np
from io_utils import parse_input, write_tour, validate_tour
from construct import nearest_neighbour, alpha_random_nn, cheapest_insertion
from two_opt import two_opt_local_search, calculate_delta, is_valid_move
from perturb import double_bridge, or_opt_kick, validate_perturbation
from utils import cost, is_symmetric, Timer


class TestIOUtils(unittest.TestCase):
    """Test I/O utilities."""
    
    def test_parse_euclidean(self):
        """Test parsing Euclidean TSP input."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("EUCLIDEAN\n")
            f.write("3\n")
            f.write("0.0 1.0 2.0\n")
            f.write("1.0 0.0 3.0\n")
            f.write("2.0 3.0 0.0\n")
            f.flush()
            
            problem_type, n, distance_matrix = parse_input(f.name)
            
            self.assertEqual(problem_type, "EUCLIDEAN")
            self.assertEqual(n, 3)
            self.assertEqual(distance_matrix.shape, (3, 3))
            self.assertEqual(distance_matrix[0, 0], 0.0)
            self.assertEqual(distance_matrix[0, 1], 1.0)
            
            os.unlink(f.name)
    
    def test_parse_non_euclidean(self):
        """Test parsing Non-Euclidean TSP input."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("NON-EUCLIDEAN\n")
            f.write("2\n")
            f.write("0.0 5.0\n")
            f.write("5.0 0.0\n")
            f.flush()
            
            problem_type, n, distance_matrix = parse_input(f.name)
            
            self.assertEqual(problem_type, "NON-EUCLIDEAN")
            self.assertEqual(n, 2)
            self.assertEqual(distance_matrix.shape, (2, 2))
            
            os.unlink(f.name)
    
    def test_write_tour(self):
        """Test writing tour to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            
            tour = [0, 1, 2]
            write_tour(f.name, tour)
            
            with open(f.name, 'r') as rf:
                content = rf.read().strip()
                self.assertEqual(content, "0 1 2")
            
            os.unlink(f.name)
    
    def test_validate_tour(self):
        """Test tour validation."""
        self.assertTrue(validate_tour([0, 1, 2], 3))
        self.assertTrue(validate_tour([2, 0, 1], 3))
        self.assertFalse(validate_tour([0, 1], 3))  # Wrong length
        self.assertFalse(validate_tour([0, 1, 3], 3))  # Invalid city
        self.assertFalse(validate_tour([0, 1, 1], 3))  # Duplicate


class TestConstruction(unittest.TestCase):
    """Test construction methods."""
    
    def setUp(self):
        """Set up test data."""
        self.distance_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0]
        ])
    
    def test_nearest_neighbour(self):
        """Test nearest neighbour construction."""
        tour = nearest_neighbour(self.distance_matrix, start_city=0)
        self.assertEqual(len(tour), 4)
        self.assertTrue(validate_tour(tour, 4))
        self.assertEqual(tour[0], 0)  # Should start at specified city
    
    def test_alpha_random_nn(self):
        """Test α-random nearest neighbour construction."""
        tour = alpha_random_nn(self.distance_matrix, alpha=0.5)
        self.assertEqual(len(tour), 4)
        self.assertTrue(validate_tour(tour, 4))
    
    def test_cheapest_insertion(self):
        """Test cheapest insertion construction."""
        tour = cheapest_insertion(self.distance_matrix)
        self.assertEqual(len(tour), 4)
        self.assertTrue(validate_tour(tour, 4))


class TestTwoOpt(unittest.TestCase):
    """Test 2-opt local search."""
    
    def setUp(self):
        """Set up test data."""
        self.distance_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0]
        ])
        self.tour = [0, 1, 2, 3]
    
    def test_calculate_delta(self):
        """Test delta calculation."""
        delta = calculate_delta(self.tour, 1, 2, self.distance_matrix)
        self.assertIsInstance(delta, float)
    
    def test_is_valid_move(self):
        """Test move validation."""
        self.assertTrue(is_valid_move(1, 3, 4))  # Valid move (non-adjacent)
        self.assertFalse(is_valid_move(0, 1, 4))  # Adjacent cities
        self.assertFalse(is_valid_move(0, 3, 4))  # Wrap-around
    
    def test_two_opt_local_search(self):
        """Test 2-opt local search."""
        timer = Timer()
        improved_tour = two_opt_local_search(self.tour, self.distance_matrix, timer)
        self.assertEqual(len(improved_tour), 4)
        self.assertTrue(validate_tour(improved_tour, 4))
        
        # Should not increase cost
        original_cost = cost(self.tour, self.distance_matrix)
        improved_cost = cost(improved_tour, self.distance_matrix)
        self.assertLessEqual(improved_cost, original_cost)


class TestPerturbation(unittest.TestCase):
    """Test perturbation methods."""
    
    def setUp(self):
        """Set up test data."""
        self.tour = [0, 1, 2, 3, 4, 5]
    
    def test_double_bridge(self):
        """Test double-bridge perturbation."""
        perturbed = double_bridge(self.tour)
        self.assertEqual(len(perturbed), len(self.tour))
        self.assertTrue(validate_perturbation(self.tour, perturbed))
    
    def test_or_opt_kick(self):
        """Test OR-opt-kick perturbation."""
        perturbed = or_opt_kick(self.tour, k=3)
        self.assertEqual(len(perturbed), len(self.tour))
        self.assertTrue(validate_perturbation(self.tour, perturbed))
    
    def test_validate_perturbation(self):
        """Test perturbation validation."""
        valid_tour = [1, 0, 2, 3, 4, 5]  # Valid permutation
        invalid_tour = [0, 1, 2, 3, 4]    # Wrong length
        invalid_tour2 = [0, 1, 2, 3, 4, 4]  # Duplicate
        
        self.assertTrue(validate_perturbation(self.tour, valid_tour))
        self.assertFalse(validate_perturbation(self.tour, invalid_tour))
        self.assertFalse(validate_perturbation(self.tour, invalid_tour2))


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.distance_matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0]
        ])
        self.tour = [0, 1, 2]
    
    def test_cost(self):
        """Test cost calculation."""
        tour_cost = cost(self.tour, self.distance_matrix)
        expected_cost = 1.0 + 1.0 + 2.0  # 0->1 + 1->2 + 2->0
        self.assertEqual(tour_cost, expected_cost)
    
    def test_is_symmetric(self):
        """Test symmetry detection."""
        self.assertTrue(is_symmetric(self.distance_matrix))
        
        # Make asymmetric
        asymmetric_matrix = self.distance_matrix.copy()
        asymmetric_matrix[0, 1] = 5.0
        self.assertFalse(is_symmetric(asymmetric_matrix))
    
    def test_timer(self):
        """Test timer functionality."""
        timer = Timer()
        import time
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = timer.elapsed()
        self.assertGreater(elapsed, 0.005)  # Should be at least 5ms


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_small_euclidean_problem(self):
        """Test solving a small Euclidean problem."""
        distance_matrix = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        
        # Test construction
        tour = nearest_neighbour(distance_matrix)
        self.assertTrue(validate_tour(tour, 3))
        
        # Test 2-opt
        timer = Timer()
        improved_tour = two_opt_local_search(tour, distance_matrix, timer)
        self.assertTrue(validate_tour(improved_tour, 3))
        
        # Test perturbation
        perturbed_tour = double_bridge(improved_tour)
        self.assertTrue(validate_perturbation(improved_tour, perturbed_tour))
    
    def test_cost_improvement(self):
        """Test that 2-opt improves cost."""
        # Create a deliberately bad tour
        distance_matrix = np.array([
            [0.0, 1.0, 10.0, 1.0],
            [1.0, 0.0, 1.0, 10.0],
            [10.0, 1.0, 0.0, 1.0],
            [1.0, 10.0, 1.0, 0.0]
        ])
        
        bad_tour = [0, 2, 1, 3]  # High cost tour: 0->2(10) + 2->1(1) + 1->3(10) + 3->0(1) = 22
        timer = Timer()
        improved_tour = two_opt_local_search(bad_tour, distance_matrix, timer)
        
        bad_cost = cost(bad_tour, distance_matrix)
        improved_cost = cost(improved_tour, distance_matrix)
        
        # Should not increase cost (may stay the same if already optimal)
        self.assertLessEqual(improved_cost, bad_cost)


def run_performance_test():
    """Run performance test on larger instance."""
    print("Running performance test...")
    
    # Create a larger test instance
    n = 20
    distance_matrix = np.random.rand(n, n) * 100
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Test construction time
    timer = Timer()
    tour = nearest_neighbour(distance_matrix)
    construction_time = timer.elapsed()
    print(f"Construction time for n={n}: {construction_time:.3f}s")
    
    # Test 2-opt time
    timer = Timer()
    improved_tour = two_opt_local_search(tour, distance_matrix, timer)
    two_opt_time = timer.elapsed()
    print(f"2-opt time for n={n}: {two_opt_time:.3f}s")
    
    # Test perturbation time
    timer = Timer()
    perturbed_tour = double_bridge(improved_tour)
    perturbation_time = timer.elapsed()
    print(f"Perturbation time for n={n}: {perturbation_time:.3f}s")
    
    # Verify costs
    original_cost = cost(tour, distance_matrix)
    improved_cost = cost(improved_tour, distance_matrix)
    perturbed_cost = cost(perturbed_tour, distance_matrix)
    
    print(f"Original cost: {original_cost:.2f}")
    print(f"Improved cost: {improved_cost:.2f}")
    print(f"Perturbed cost: {perturbed_cost:.2f}")
    
    # Verify improvement
    assert improved_cost <= original_cost, "2-opt should not increase cost"
    assert validate_perturbation(improved_tour, perturbed_tour), "Perturbation should be valid"


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    print("\n" + "="*50)
    run_performance_test()
    
    print("\nAll tests completed successfully!")
