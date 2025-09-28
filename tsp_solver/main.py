#!/usr/bin/env python3
"""
TSP Solver - Main CLI and orchestration
Implements Iterated Local Search with 2-Opt and diverse construction methods.
"""

import argparse
import signal
import sys
import time
from typing import Optional

from io_utils import parse_input, write_tour, smart_open
from construct import construct_diverse_tours
from two_opt import two_opt_local_search
from perturb import perturb_tour
from ils import iterated_local_search
from utils import cost, set_random_seed, Timer

# Global verbose flag
VERBOSE = False


class TSPTimer:
    """Timer with graceful exit handling."""
    
    def __init__(self, max_time: float = 295.0):
        self.max_time = max_time
        self.start_time = time.time()
        self.best_tour = None
        self.best_cost = float('inf')
        
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def remaining(self) -> float:
        return max(0, self.max_time - self.elapsed())
    
    def should_continue(self) -> bool:
        return self.elapsed() < self.max_time
    
    def update_best(self, tour, cost_val):
        if cost_val < self.best_cost:
            self.best_tour = tour.copy()
            self.best_cost = cost_val
            return True
        return False


def signal_handler(signum, frame):
    """Handle SIGTERM/Ctrl-C gracefully."""
    print(f"\nReceived signal {signum}. Exiting gracefully...")
    sys.exit(0)


def solve_tsp(input_file: str, output_file: str, seed: Optional[int] = None, max_time: float = 295.0) -> None:
    """
    Main TSP solving pipeline following the implementation plan.
    """
    # Set up signal handling
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize timer
    timer = TSPTimer(max_time)
    
    # Parse input
    try:
        problem_type, n, distance_matrix = parse_input(input_file)
        print(f"Loaded {problem_type} TSP with {n} cities")
    except Exception as e:
        print(f"Error parsing input: {e}")
        sys.exit(1)
    
    # Handle edge cases
    if n <= 3:
        trivial_tour = list(range(n))
        write_tour(output_file, trivial_tour)
        print(f"Trivial solution for n={n}: {trivial_tour}")
        return
    
    # Set random seed
    if seed is not None:
        set_random_seed(seed)
    
    # Determine if symmetric and Euclidean
    is_symmetric = True
    is_euclidean = (problem_type == 'EUCLIDEAN')
    for i in range(n):
        for j in range(i+1, n):
            if abs(distance_matrix[i][j] - distance_matrix[j][i]) > 1e-6:
                is_symmetric = False
                break
        if not is_symmetric:
            break
    
    if __debug__:
        print(f"Problem type: {problem_type}, Symmetric: {is_symmetric}, Euclidean: {is_euclidean}")
    
    # Phase 1: Diverse construction (≤15s)
    if __debug__:
        print("Phase 1: Constructing diverse tours...")
    construction_timer = Timer()
    tours = construct_diverse_tours(distance_matrix, problem_type, timer)
    construction_time = construction_timer.elapsed()
    print(f"Construction completed in {construction_time:.2f}s, found {len(tours)} tours")
    
    # Phase 2: First 2-opt pass (≤45s)
    if __debug__:
        print("Phase 2: First 2-opt local search...")
    two_opt_timer = Timer()
    improved_tours = []
    for i, tour in enumerate(tours):
        if not timer.should_continue():
            break
        improved_tour = two_opt_local_search(tour, distance_matrix, timer, is_euclidean)
        improved_tours.append(improved_tour)
        timer.update_best(improved_tour, cost(improved_tour, distance_matrix))
        print(f"Tour {i+1}: {cost(tour, distance_matrix):.2f} -> {cost(improved_tour, distance_matrix):.2f}")
    
    two_opt_time = two_opt_timer.elapsed()
    print(f"2-opt completed in {two_opt_time:.2f}s")
    
    # Phase 3: Iterated Local Search (~235s)
    if __debug__:
        print("Phase 3: Iterated Local Search...")
    ils_timer = Timer()
    final_tour = iterated_local_search(
        timer.best_tour, 
        distance_matrix, 
        problem_type, 
        timer,
        is_euclidean
    )
    ils_time = ils_timer.elapsed()
    
    # Ensure we have a valid final tour
    if final_tour is None or len(final_tour) != n:
        print("Warning: ILS returned invalid tour, using best tour from construction phase")
        final_tour = timer.best_tour.copy()
    
    # Write final result
    write_tour(output_file, final_tour)
    final_cost = cost(final_tour, distance_matrix)
    
    # Debug assertions for tour validation
    if __debug__:
        assert len(final_tour) == n, f"Final tour length {len(final_tour)} != n {n}"
        assert len(set(final_tour)) == n, f"Final tour contains duplicates: {final_tour}"
        assert all(0 <= v < n for v in final_tour), f"Final tour contains invalid cities: {final_tour}"
    
    print(f"Final solution cost: {final_cost:.2f}")
    print(f"Total time: {timer.elapsed():.2f}s")
    print(f"ILS time: {ils_time:.2f}s")
    
    # Validate output file was created successfully
    try:
        with smart_open(output_file) as f:
            lines = f.readlines()
            if not lines:
                print("Warning: Output file is empty")
            else:
                print(f"Output file contains {len(lines)} tour(s)")
    except Exception as e:
        print(f"Warning: Could not verify output file: {e}")


def main():
    parser = argparse.ArgumentParser(description="TSP Solver using Iterated Local Search")
    parser.add_argument("input_file", help="Input TSP file")
    parser.add_argument("output_file", help="Output tour file")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--max-time", type=float, default=295.0, help="Maximum time limit in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose
    
    solve_tsp(args.input_file, args.output_file, args.seed, args.max_time)


if __name__ == "__main__":
    main()
