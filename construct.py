"""
Construction methods for diverse initial tours.
Implements Nearest-Neighbour, α-Random NN, and Cheapest-Insertion.
"""

import numpy as np
import random
from typing import List, Tuple
from utils import cost, Timer


def nearest_neighbour(distance_matrix: np.ndarray, start_city: int = None) -> List[int]:
    """
    Nearest Neighbour construction starting from start_city.
    
    Args:
        distance_matrix: n×n distance matrix
        start_city: starting city (random if None)
        
    Returns:
        tour as list of city indices
    """
    n = distance_matrix.shape[0]
    
    if start_city is None:
        start_city = random.randint(0, n - 1)
    
    tour = [start_city]
    unvisited = set(range(n)) - {start_city}
    
    current = start_city
    while unvisited:
        # Find nearest unvisited city
        nearest = min(unvisited, key=lambda city: distance_matrix[current, city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour


def alpha_random_nn(distance_matrix: np.ndarray, alpha: float = 0.2) -> List[int]:
    """
    α-Random Nearest Neighbour construction.
    
    Args:
        distance_matrix: n×n distance matrix
        alpha: randomness parameter (0.0 = greedy, 1.0 = random)
        
    Returns:
        tour as list of city indices
    """
    n = distance_matrix.shape[0]
    start_city = random.randint(0, n - 1)
    
    tour = [start_city]
    unvisited = set(range(n)) - {start_city}
    
    current = start_city
    while unvisited:
        # Get distances to unvisited cities
        distances = [(city, distance_matrix[current, city]) for city in unvisited]
        distances.sort(key=lambda x: x[1])
        
        # Select from top α% of candidates
        num_candidates = max(1, int(len(distances) * alpha))
        candidates = distances[:num_candidates]
        
        # Randomly select from candidates
        chosen_city = random.choice(candidates)[0]
        tour.append(chosen_city)
        unvisited.remove(chosen_city)
        current = chosen_city
    
    return tour


def cheapest_insertion(distance_matrix: np.ndarray, seed_tour: List[int] = None) -> List[int]:
    """
    Cheapest Insertion construction.
    
    Args:
        distance_matrix: n×n distance matrix
        seed_tour: initial partial tour (Christofides-style if None)
        
    Returns:
        tour as list of city indices
    """
    n = distance_matrix.shape[0]
    
    if seed_tour is None:
        # Start with a small subtour (Christofides-style)
        if n >= 3:
            # Create a triangle with minimum perimeter
            best_triangle = None
            best_perimeter = float('inf')
            
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        perimeter = (distance_matrix[i, j] + 
                                   distance_matrix[j, k] + 
                                   distance_matrix[k, i])
                        if perimeter < best_perimeter:
                            best_perimeter = perimeter
                            best_triangle = [i, j, k]
            
            tour = best_triangle
        else:
            tour = list(range(n))
    else:
        tour = seed_tour.copy()
    
    unvisited = set(range(n)) - set(tour)
    
    while unvisited:
        best_city = None
        best_position = None
        best_cost_increase = float('inf')
        
        # Try inserting each unvisited city at each position
        for city in unvisited:
            for pos in range(len(tour) + 1):
                # Calculate cost increase
                if pos == 0:
                    cost_increase = (distance_matrix[city, tour[0]] + 
                                   distance_matrix[tour[-1], city] - 
                                   distance_matrix[tour[-1], tour[0]])
                elif pos == len(tour):
                    cost_increase = (distance_matrix[tour[-1], city] + 
                                   distance_matrix[city, tour[0]] - 
                                   distance_matrix[tour[-1], tour[0]])
                else:
                    cost_increase = (distance_matrix[tour[pos-1], city] + 
                                   distance_matrix[city, tour[pos]] - 
                                   distance_matrix[tour[pos-1], tour[pos]])
                
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_city = city
                    best_position = pos
        
        # Insert the best city
        tour.insert(best_position, best_city)
        unvisited.remove(best_city)
    
    return tour


def construct_diverse_tours(distance_matrix: np.ndarray, 
                          problem_type: str, 
                          timer: Timer) -> List[List[int]]:
    """
    Construct diverse initial tours using multiple methods.
    
    Args:
        distance_matrix: n×n distance matrix
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        timer: timer object for time management
        
    Returns:
        list of diverse tours
    """
    tours = []
    construction_start = timer.elapsed()
    max_construction_time = 15.0  # seconds
    
    # Method 1: Nearest-Neighbour with 3 random starts
    print("  Constructing Nearest-Neighbour tours...")
    for _ in range(3):
        if timer.elapsed() - construction_start > max_construction_time:
            break
        tour = nearest_neighbour(distance_matrix)
        tours.append(tour)
        print(f"    NN tour cost: {cost(tour, distance_matrix):.2f}")
    
    # Method 2: α-Random NN (2 runs)
    print("  Constructing α-Random NN tours...")
    for _ in range(2):
        if timer.elapsed() - construction_start > max_construction_time:
            break
        tour = alpha_random_nn(distance_matrix, alpha=0.2)
        tours.append(tour)
        print(f"    α-NN tour cost: {cost(tour, distance_matrix):.2f}")
    
    # Method 3: Cheapest-Insertion
    print("  Constructing Cheapest-Insertion tour...")
    if timer.elapsed() - construction_start <= max_construction_time:
        tour = cheapest_insertion(distance_matrix)
        tours.append(tour)
        print(f"    CI tour cost: {cost(tour, distance_matrix):.2f}")
    
    # Stop if we have 5 tours or spent 15s
    if len(tours) >= 5 or timer.elapsed() - construction_start >= max_construction_time:
        print(f"  Construction stopped: {len(tours)} tours in {timer.elapsed() - construction_start:.2f}s")
        return tours[:5]  # Return at most 5 tours
    
    return tours
