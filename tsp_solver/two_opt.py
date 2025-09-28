"""
2-Opt local search implementation with candidate lists and don't-look bits.
Implements first-improvement strategy with vectorized delta checks.
"""

import numpy as np
from typing import List, Tuple
from utils import cost, Timer


def get_candidate_size(n: int, is_euclidean: bool) -> int:
    """
    Calculate adaptive candidate list size.
    
    Args:
        n: number of cities
        is_euclidean: whether problem is Euclidean
        
    Returns:
        candidate list size
    """
    if is_euclidean:
        # For Euclidean: keep 25 or min(25, N//3)
        return min(25, max(25, n // 3))
    else:
        # For Non-Euclidean: 25 + 5 = 30 often helps
        return 30


def build_candidate_lists(distance_matrix: np.ndarray, 
                         tour: List[int],
                         is_euclidean: bool = True) -> List[List[int]]:
    """
    Build candidate lists for each city in the tour.
    
    Args:
        distance_matrix: n×n distance matrix
        tour: current tour
        is_euclidean: whether problem is Euclidean
        
    Returns:
        list of candidate lists (one per city)
    """
    n = distance_matrix.shape[0]
    candidates = []
    
    for i in range(n):
        # Get distances to all other cities
        distances = [(j, distance_matrix[i, j]) for j in range(n) if j != i]
        distances.sort(key=lambda x: x[1])
        
        # Take top candidates
        candidate_size = get_candidate_size(n, is_euclidean)
        city_candidates = [j for j, _ in distances[:candidate_size]]
        candidates.append(city_candidates)
    
    return candidates


def calculate_delta(tour: List[int], 
                   i: int, j: int, 
                   distance_matrix: np.ndarray) -> float:
    """
    Calculate cost change for 2-opt move (i, j).
    
    Args:
        tour: current tour
        i, j: positions for 2-opt move
        distance_matrix: n×n distance matrix
        
    Returns:
        cost change (negative means improvement)
    """
    n = len(tour)
    
    # Handle wrap-around
    prev_i = tour[i - 1] if i > 0 else tour[n - 1]
    curr_i = tour[i]
    curr_j = tour[j]
    next_j = tour[j + 1] if j < n - 1 else tour[0]
    
    # Original edges: (prev_i, curr_i) and (curr_j, next_j)
    # New edges: (prev_i, curr_j) and (curr_i, next_j)
    old_cost = distance_matrix[prev_i, curr_i] + distance_matrix[curr_j, next_j]
    new_cost = distance_matrix[prev_i, curr_j] + distance_matrix[curr_i, next_j]
    
    return new_cost - old_cost


def is_valid_move(i: int, j: int, n: int) -> bool:
    """
    Check if 2-opt move (i, j) is valid.
    
    Args:
        i, j: positions for 2-opt move
        n: tour length
        
    Returns:
        True if move is valid
    """
    # Valid if j > i+1 and not wrapping around the entire tour
    return j > i + 1 and not (i == 0 and j == n - 1)


def two_opt_move(tour: List[int], i: int, j: int) -> List[int]:
    """
    Perform 2-opt move: reverse segment from i to j.
    
    Args:
        tour: current tour
        i, j: positions for 2-opt move
        
    Returns:
        new tour after 2-opt move
    """
    new_tour = tour.copy()
    new_tour[i:j+1] = reversed(new_tour[i:j+1])
    return new_tour


def two_opt_local_search(tour: List[int], 
                        distance_matrix: np.ndarray, 
                        timer: Timer,
                        is_euclidean: bool = True) -> List[int]:
    """
    First-improvement 2-opt local search with candidate lists and don't-look bits.
    
    Args:
        tour: initial tour
        distance_matrix: n×n distance matrix
        timer: timer object for time management
        is_euclidean: whether problem is Euclidean
        
    Returns:
        locally optimal tour
    """
    current_tour = tour.copy()
    n = len(current_tour)
    
    # Initialize don't-look bits
    dont_look = [False] * n
    
    # Build candidate lists
    candidates = build_candidate_lists(distance_matrix, current_tour, is_euclidean)
    
    improved = True
    iterations = 0
    total_improvements = 0
    
    while improved and timer.should_continue():
        improved = False
        iterations += 1
        
        # Check every 100 iterations for time limit
        if iterations % 100 == 0 and not timer.should_continue():
            break
        
        for i in range(n):
            if dont_look[i]:
                continue
                
            # Check candidates for city at position i
            city_i = current_tour[i]
            city_candidates = candidates[city_i]
            
            for candidate_city in city_candidates:
                # Find position of candidate city in tour
                try:
                    j = current_tour.index(candidate_city)
                except ValueError:
                    continue
                
                # Check if move is valid
                if not is_valid_move(i, j, n):
                    continue
                
                # Calculate delta with floating-point guard
                delta = calculate_delta(current_tour, i, j, distance_matrix)
                
                if delta < -1e-9:  # Significant improvement
                    # Perform the move
                    current_tour = two_opt_move(current_tour, i, j)
                    improved = True
                    total_improvements += 1
                    
                    # Reset don't-look bits for affected cities
                    k = get_candidate_size(n, is_euclidean)
                    for pos in range(max(0, i-k), min(n, i+k+1)):
                        dont_look[pos] = False
                    for pos in range(max(0, j-k), min(n, j+k+1)):
                        dont_look[pos] = False
                    
                    break
            
            if improved:
                break
        
        # Set don't-look bits for cities that weren't improved
        if not improved:
            for i in range(n):
                if not dont_look[i]:
                    dont_look[i] = True
    
    if __debug__:
        print(f"  2-opt completed {iterations} iterations, {total_improvements} improvements")
    return current_tour


def vectorized_delta_check(tour: List[int], 
                          i: int, 
                          candidate_positions: List[int],
                          distance_matrix: np.ndarray) -> List[float]:
    """
    Vectorized delta calculation for multiple candidate positions.
    
    Args:
        tour: current tour
        i: position to check moves from
        candidate_positions: list of positions to check moves to
        distance_matrix: n×n distance matrix
        
    Returns:
        list of delta values
    """
    n = len(tour)
    deltas = []
    
    prev_i = tour[i - 1] if i > 0 else tour[n - 1]
    curr_i = tour[i]
    
    for j in candidate_positions:
        if not is_valid_move(i, j, n):
            deltas.append(float('inf'))  # Invalid move
            continue
            
        curr_j = tour[j]
        next_j = tour[j + 1] if j < n - 1 else tour[0]
        
        old_cost = distance_matrix[prev_i, curr_i] + distance_matrix[curr_j, next_j]
        new_cost = distance_matrix[prev_i, curr_j] + distance_matrix[curr_i, next_j]
        
        deltas.append(new_cost - old_cost)
    
    return deltas
