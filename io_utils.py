"""
I/O utilities for TSP input/output handling.
Handles EUCLIDEAN and NON-EUCLIDEAN input formats.
"""

import numpy as np
from typing import Tuple, List


def parse_input(filename: str) -> Tuple[str, int, np.ndarray]:
    """
    Parse TSP input file.
    
    Returns:
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        n: number of cities
        distance_matrix: n×n numpy array
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse problem type
    problem_type = lines[0].strip()
    if problem_type not in ['EUCLIDEAN', 'NON-EUCLIDEAN']:
        raise ValueError(f"Invalid problem type: {problem_type}")
    
    # Parse number of cities
    n = int(lines[1].strip())
    if n <= 0:
        raise ValueError(f"Invalid number of cities: {n}")
    
    # Parse distance matrix
    distance_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        row_values = lines[2 + i].strip().split()
        if len(row_values) != n:
            raise ValueError(f"Row {i+2} has {len(row_values)} values, expected {n}")
        
        for j in range(n):
            try:
                distance_matrix[i, j] = float(row_values[j])
            except ValueError:
                raise ValueError(f"Invalid distance value at ({i}, {j}): {row_values[j]}")
    
    # Validate diagonal elements
    for i in range(n):
        if abs(distance_matrix[i, i]) > 1e-9:
            raise ValueError(f"Diagonal element ({i}, {i}) should be 0, got {distance_matrix[i, i]}")
    
    # Validate non-negative distances
    for i in range(n):
        for j in range(n):
            if distance_matrix[i, j] < 0:
                raise ValueError(f"Negative distance at ({i}, {j}): {distance_matrix[i, j]}")
    
    return problem_type, n, distance_matrix


def write_tour(filename: str, tour: List[int]) -> None:
    """
    Write tour to output file.
    
    Args:
        filename: output file path
        tour: list of city indices (0-based)
    """
    with open(filename, 'a') as f:  # Append mode for progressive output
        tour_str = ' '.join(map(str, tour))
        f.write(tour_str + '\n')


def validate_tour(tour: List[int], n: int) -> bool:
    """
    Validate that tour is a valid permutation of cities.
    
    Args:
        tour: list of city indices
        n: number of cities
        
    Returns:
        True if valid, False otherwise
    """
    if len(tour) != n:
        return False
    
    if set(tour) != set(range(n)):
        return False
    
    return True


def format_tour_cost(tour: List[int], distance_matrix: np.ndarray) -> str:
    """
    Format tour with its cost for display.
    
    Args:
        tour: list of city indices
        distance_matrix: n×n distance matrix
        
    Returns:
        formatted string
    """
    from utils import cost
    tour_cost = cost(tour, distance_matrix)
    return f"Tour: {' '.join(map(str, tour))} | Cost: {tour_cost:.6f}"
