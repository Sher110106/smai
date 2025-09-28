"""
Iterated Local Search (ILS) implementation with dynamic acceptance criteria.
Implements adaptive ε and restart mechanisms.
"""

import time
from typing import List, Tuple
from utils import cost, Timer
from two_opt import two_opt_local_search
from perturb import perturb_tour


def calculate_acceptance_threshold(iteration: int, 
                                 max_iterations: int,
                                 problem_type: str) -> float:
    """
    Calculate dynamic acceptance threshold ε.
    
    Args:
        iteration: current iteration
        max_iterations: maximum iterations
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        
    Returns:
        acceptance threshold ε
    """
    if problem_type == 'EUCLIDEAN':
        return 0.0  # Only accept better solutions
    
    # For NON-EUCLIDEAN: ε(t) = 0.015 * 0.1^(t / maxIter)
    # Changed from 0.01 to 0.015 (1.5%) to help with early cost plateaus
    progress = iteration / max_iterations
    epsilon = 0.015 * (0.1 ** progress)
    return epsilon


def should_accept_solution(current_cost: float,
                         new_cost: float,
                         epsilon: float) -> bool:
    """
    Determine if new solution should be accepted.
    
    Args:
        current_cost: cost of current solution
        new_cost: cost of new solution
        epsilon: acceptance threshold
        
    Returns:
        True if solution should be accepted
    """
    if epsilon == 0.0:
        return new_cost < current_cost  # Only better solutions
    
    # Accept if improvement or within epsilon threshold
    return new_cost <= current_cost + epsilon


def iterated_local_search(initial_tour: List[int],
                         distance_matrix,
                         problem_type: str,
                         timer: Timer,
                         is_euclidean: bool = True) -> List[int]:
    """
    Iterated Local Search with dynamic acceptance and restart.
    
    Args:
        initial_tour: starting tour
        distance_matrix: n×n distance matrix
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        timer: timer object for time management
        
    Returns:
        best tour found
    """
    current_tour = initial_tour.copy()
    best_tour = initial_tour.copy()
    best_cost = cost(best_tour, distance_matrix)
    
    iteration = 0
    max_iterations = 10000  # Large number, time-limited
    no_improvement_count = 0
    restart_threshold = 60  # seconds for restart (Non-Euclidean only)
    last_improvement_time = timer.elapsed()
    
    print(f"Starting ILS with initial cost: {best_cost:.2f}")
    
    while timer.should_continue() and iteration < max_iterations:
        iteration += 1
        
        # Check for restart (Non-Euclidean only)
        if (problem_type == 'NON-EUCLIDEAN' and 
            timer.elapsed() - last_improvement_time > restart_threshold):
            print(f"Restarting ILS after {restart_threshold}s without improvement")
            # Restart with best tour found so far
            current_tour = best_tour.copy()
            last_improvement_time = timer.elapsed()
            no_improvement_count = 0
        
        # Perturbation phase
        n = len(current_tour)
        perturbed_tour = perturb_tour(current_tour, problem_type, timer, n)
        
        # Local search phase
        improved_tour = two_opt_local_search(perturbed_tour, distance_matrix, timer, is_euclidean)
        improved_cost = cost(improved_tour, distance_matrix)
        
        # Calculate acceptance threshold
        epsilon = calculate_acceptance_threshold(iteration, max_iterations, problem_type)
        
        # Acceptance decision
        current_cost = cost(current_tour, distance_matrix)
        
        if should_accept_solution(current_cost, improved_cost, epsilon):
            current_tour = improved_tour.copy()
            
            # Update best if improved
            if improved_cost < best_cost:
                best_tour = improved_tour.copy()
                best_cost = improved_cost
                last_improvement_time = timer.elapsed()
                no_improvement_count = 0
                # Only print improvements in debug mode
                if __debug__:
                    print(f"Iteration {iteration}: New best cost {best_cost:.2f} (ε={epsilon:.4f})")
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
        
        # Reduced frequency progress reporting (only in debug mode)
        if iteration % 1000 == 0 and __debug__:
            elapsed = timer.elapsed()
            print(f"Iteration {iteration}: Current {current_cost:.2f}, Best {best_cost:.2f}, "
                  f"ε={epsilon:.4f}, Time {elapsed:.1f}s")
        
        # Check if we should continue
        if not timer.should_continue():
            break
    
    print(f"ILS completed: {iteration} iterations, best cost {best_cost:.2f}")
    return best_tour


def multi_start_ils(initial_tours: List[List[int]],
                   distance_matrix,
                   problem_type: str,
                   timer: Timer,
                   is_euclidean: bool = True) -> List[int]:
    """
    Multi-start Iterated Local Search.
    
    Args:
        initial_tours: list of starting tours
        distance_matrix: n×n distance matrix
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        timer: timer object for time management
        
    Returns:
        best tour across all starts
    """
    global_best_tour = None
    global_best_cost = float('inf')
    
    for i, initial_tour in enumerate(initial_tours):
        if not timer.should_continue():
            break
            
        print(f"Starting ILS run {i+1}/{len(initial_tours)}")
        
        # Run ILS from this starting tour
        best_tour = iterated_local_search(initial_tour, distance_matrix, problem_type, timer, is_euclidean)
        best_cost = cost(best_tour, distance_matrix)
        
        # Update global best
        if best_cost < global_best_cost:
            global_best_tour = best_tour.copy()
            global_best_cost = best_cost
            print(f"New global best: {global_best_cost:.2f}")
    
    return global_best_tour


def adaptive_ils(initial_tour: List[int],
                distance_matrix,
                problem_type: str,
                timer: Timer,
                is_euclidean: bool = True) -> List[int]:
    """
    Adaptive ILS that adjusts parameters based on problem characteristics.
    
    Args:
        initial_tour: starting tour
        distance_matrix: n×n distance matrix
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        timer: timer object for time management
        
    Returns:
        best tour found
    """
    n = distance_matrix.shape[0]
    
    # Adaptive parameters based on problem size
    if n <= 50:
        max_iterations = 5000
        restart_threshold = 30
    elif n <= 100:
        max_iterations = 8000
        restart_threshold = 45
    else:  # n > 100
        max_iterations = 12000
        restart_threshold = 60
    
    # Adjust restart threshold for Euclidean problems
    if problem_type == 'EUCLIDEAN':
        restart_threshold = restart_threshold * 2  # Less frequent restarts
    
    print(f"Adaptive ILS: max_iter={max_iterations}, restart_threshold={restart_threshold}s")
    
    # Run standard ILS with adaptive parameters
    return iterated_local_search(initial_tour, distance_matrix, problem_type, timer, is_euclidean)
