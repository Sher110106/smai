"""
Utility functions for TSP solver.
Includes cost calculation, random seed management, and timing utilities.
"""

import numpy as np
import random
import time
from typing import List, Optional, Tuple


def cost(tour: List[int], distance_matrix: np.ndarray) -> float:
    """
    Calculate total cost of a tour.
    
    Args:
        tour: list of city indices
        distance_matrix: n×n distance matrix
        
    Returns:
        total tour cost
    """
    if len(tour) <= 1:
        return 0.0
    
    total_cost = 0.0
    n = len(tour)
    
    for i in range(n):
        from_city = tour[i]
        to_city = tour[(i + 1) % n]
        total_cost += distance_matrix[from_city, to_city]
    
    return total_cost


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    """Simple timer utility."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def reset(self) -> None:
        """Reset timer."""
        self.start_time = time.time()
    
    def should_continue(self) -> bool:
        """Always return True for simple timer (no time limit)."""
        return True


def validate_tour(tour: List[int], n: int) -> bool:
    """
    Validate that tour is a valid permutation.
    
    Args:
        tour: list of city indices
        n: number of cities
        
    Returns:
        True if valid tour
    """
    if len(tour) != n:
        return False
    
    if set(tour) != set(range(n)):
        return False
    
    return True


def format_tour(tour: List[int]) -> str:
    """
    Format tour as space-separated string.
    
    Args:
        tour: list of city indices
        
    Returns:
        formatted tour string
    """
    return ' '.join(map(str, tour))


def calculate_gap(tour_cost: float, best_known_cost: float) -> float:
    """
    Calculate optimality gap percentage.
    
    Args:
        tour_cost: cost of current tour
        best_known_cost: best known cost (or lower bound)
        
    Returns:
        gap percentage
    """
    if best_known_cost == 0:
        return 0.0
    
    return ((tour_cost - best_known_cost) / best_known_cost) * 100


def is_symmetric(distance_matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if distance matrix is symmetric.
    
    Args:
        distance_matrix: n×n distance matrix
        tolerance: tolerance for floating-point comparison
        
    Returns:
        True if matrix is symmetric
    """
    n = distance_matrix.shape[0]
    
    for i in range(n):
        for j in range(i + 1, n):
            if abs(distance_matrix[i, j] - distance_matrix[j, i]) > tolerance:
                return False
    
    return True


def optimize_distance_matrix_storage(distance_matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Optimize distance matrix storage for large instances.
    
    Args:
        distance_matrix: n×n distance matrix
        
    Returns:
        optimized matrix and whether it's stored as upper triangle
    """
    n = distance_matrix.shape[0]
    
    # Switch to float32 for large instances
    if n >= 150:
        optimized_matrix = distance_matrix.astype(np.float32)
    else:
        optimized_matrix = distance_matrix.astype(np.float64)
    
    # Store upper triangle only for symmetric matrices
    if n >= 150 and is_symmetric(distance_matrix):
        # Extract upper triangle
        upper_triangle = []
        for i in range(n):
            for j in range(i, n):
                upper_triangle.append(distance_matrix[i, j])
        
        return np.array(upper_triangle), True
    else:
        return optimized_matrix, False


def get_distance(distance_matrix: np.ndarray, 
                i: int, j: int, 
                is_upper_triangle: bool = False) -> float:
    """
    Get distance between cities i and j.
    
    Args:
        distance_matrix: distance matrix (full or upper triangle)
        i, j: city indices
        is_upper_triangle: whether matrix is stored as upper triangle
        
    Returns:
        distance between cities i and j
    """
    if not is_upper_triangle:
        return distance_matrix[i, j]
    
    # Handle upper triangle storage
    if i <= j:
        # Calculate index in upper triangle
        idx = i * distance_matrix.shape[0] - i * (i + 1) // 2 + j - i
        return distance_matrix[idx]
    else:
        # Symmetric: return distance_matrix[j, i]
        idx = j * distance_matrix.shape[0] - j * (j + 1) // 2 + i - j
        return distance_matrix[idx]


def profile_function(func, *args, **kwargs):
    """
    Simple function profiler.
    
    Args:
        func: function to profile
        *args, **kwargs: function arguments
        
    Returns:
        tuple of (result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    return result, execution_time


def create_sample_problem(n: int, problem_type: str = 'EUCLIDEAN') -> Tuple[str, int, np.ndarray]:
    """
    Create a sample TSP problem for testing.
    
    Args:
        n: number of cities
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        
    Returns:
        tuple of (problem_type, n, distance_matrix)
    """
    if problem_type == 'EUCLIDEAN':
        # Generate random Euclidean coordinates
        coords = np.random.rand(n, 2) * 100
        
        # Calculate Euclidean distances
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                    distance_matrix[i, j] = dist
    else:
        # Generate random non-Euclidean distances
        distance_matrix = np.random.rand(n, n) * 100
        
        # Make symmetric and zero diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
    
    return problem_type, n, distance_matrix
