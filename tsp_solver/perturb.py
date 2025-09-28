"""
Perturbation methods for escaping local optima.
Implements double-bridge and OR-opt-kick perturbations.
"""

import random
from typing import List, Tuple
from utils import Timer


def double_bridge(tour: List[int]) -> List[int]:
    """
    Double-bridge perturbation: removes 4 edges and reconnects differently.
    
    Args:
        tour: current tour
        
    Returns:
        perturbed tour
    """
    n = len(tour)
    
    # Select 4 positions with gap constraints
    max_gap = max(2, n // 10)
    
    # Find valid positions
    valid_positions = []
    for i in range(n):
        for j in range(i + max_gap, n):
            for k in range(j + max_gap, n):
                for l in range(k + max_gap, n):
                    if l - k >= max_gap and (i + n - l) >= max_gap:
                        valid_positions.append((i, j, k, l))
    
    if not valid_positions:
        # Fallback: use any 4 positions, but handle small tours
        if n < 4:
            # For small tours, just return the original tour
            return tour.copy()
        positions = sorted(random.sample(range(n), 4))
        a, b, c, d = positions
    else:
        a, b, c, d = random.choice(valid_positions)
    
    # Create new tour by reconnecting segments
    # Original: [0...a][a+1...b][b+1...c][c+1...d][d+1...n-1]
    # New: [0...a][c+1...d][b+1...c][a+1...b][d+1...n-1]
    
    new_tour = []
    new_tour.extend(tour[:a+1])  # [0...a]
    new_tour.extend(tour[c+1:d+1])  # [c+1...d]
    new_tour.extend(tour[b+1:c+1])  # [b+1...c]
    new_tour.extend(tour[a+1:b+1])  # [a+1...b]
    new_tour.extend(tour[d+1:])  # [d+1...n-1]
    
    return new_tour


def or_opt_kick(tour: List[int], k: int = 3) -> List[int]:
    """
    OR-opt-kick perturbation: moves k consecutive cities to a new position.
    
    Args:
        tour: current tour
        k: number of consecutive cities to move
        
    Returns:
        perturbed tour
    """
    n = len(tour)
    
    if n <= k:
        return tour.copy()
    
    # Select starting position for the k cities
    start_pos = random.randint(0, n - k)
    
    # Extract the k cities
    moved_cities = tour[start_pos:start_pos + k]
    
    # Create tour without the moved cities
    remaining_tour = tour[:start_pos] + tour[start_pos + k:]
    
    # Select new insertion position
    insert_pos = random.randint(0, len(remaining_tour))
    
    # Insert the moved cities at new position
    new_tour = remaining_tour[:insert_pos] + moved_cities + remaining_tour[insert_pos:]
    
    return new_tour


def perturb_tour(tour: List[int], 
                problem_type: str,
                timer: Timer,
                n: int = None) -> List[int]:
    """
    Apply perturbation to escape local optimum.
    
    Args:
        tour: current tour
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        timer: timer object for time management
        n: number of cities (for adaptive mixing)
        
    Returns:
        perturbed tour
    """
    if not timer.should_continue():
        return tour.copy()
    
    # Adaptive mixing based on problem type and size
    if problem_type == 'EUCLIDEAN' and n and n >= 200:
        # If Euclidean-200 still oscillates, move to 60% / 40%
        db_probability = 0.6
    elif problem_type == 'NON-EUCLIDEAN':
        # If Non-Euclid stagnates, raise DB to 80%
        db_probability = 0.8
    else:
        # Default: 70% double-bridge, 30% OR-opt-kick
        db_probability = 0.7
    
    if random.random() < db_probability:
        return double_bridge(tour)
    else:
        return or_opt_kick(tour, k=3)


def adaptive_perturbation(tour: List[int], 
                         iteration: int,
                         max_iterations: int,
                         problem_type: str) -> List[int]:
    """
    Adaptive perturbation that changes strategy based on iteration.
    
    Args:
        tour: current tour
        iteration: current ILS iteration
        max_iterations: maximum ILS iterations
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        
    Returns:
        perturbed tour
    """
    # Early iterations: more aggressive perturbations
    # Later iterations: more conservative perturbations
    
    progress = iteration / max_iterations
    
    if progress < 0.3:  # Early phase
        # More double-bridge perturbations
        if random.random() < 0.8:
            return double_bridge(tour)
        else:
            return or_opt_kick(tour, k=random.randint(2, 4))
    elif progress < 0.7:  # Middle phase
        # Balanced perturbations
        if random.random() < 0.7:
            return double_bridge(tour)
        else:
            return or_opt_kick(tour, k=3)
    else:  # Late phase
        # More conservative perturbations
        if random.random() < 0.6:
            return double_bridge(tour)
        else:
            return or_opt_kick(tour, k=random.randint(2, 3))


def validate_perturbation(original_tour: List[int], 
                         perturbed_tour: List[int]) -> bool:
    """
    Validate that perturbation produces a valid tour.
    
    Args:
        original_tour: original tour
        perturbed_tour: perturbed tour
        
    Returns:
        True if perturbation is valid
    """
    if len(original_tour) != len(perturbed_tour):
        return False
    
    if set(original_tour) != set(perturbed_tour):
        return False
    
    return True
