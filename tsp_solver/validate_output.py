#!/usr/bin/env python3
"""
Output validation checker for TSP solver.
Validates tour format and cost monotonicity as recommended in Test.md
"""

import sys
from io_utils import parse_input
from utils import cost


def check_output(input_path: str, output_path: str) -> bool:
    """
    Validate output format and cost monotonicity.
    
    Args:
        input_path: Path to input TSP file
        output_path: Path to output tour file
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Load input data
        problem_type, n, distance_matrix = parse_input(input_path)
        
        # Load output tours
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print("ERROR: No tours produced")
            return False
        
        tours = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    tour = list(map(int, line.split()))
                    tours.append(tour)
                except ValueError as e:
                    print(f"ERROR: Invalid tour format: {line}")
                    return False
        
        if not tours:
            print("ERROR: No valid tours found")
            return False
        
        # Validate each tour
        prev_cost = float('inf')
        for i, tour in enumerate(tours):
            # Check tour validity
            if len(tour) != n:
                print(f"ERROR: Tour {i+1} length {len(tour)} != n {n}")
                return False
            
            if len(set(tour)) != n:
                print(f"ERROR: Tour {i+1} contains duplicate cities: {tour}")
                return False
            
            if not all(0 <= v < n for v in tour):
                print(f"ERROR: Tour {i+1} contains invalid city indices: {tour}")
                return False
            
            # Check cost monotonicity
            tour_cost = cost(tour, distance_matrix)
            if tour_cost > prev_cost + 1e-6:
                print(f"ERROR: Tour {i+1} cost {tour_cost:.6f} > previous cost {prev_cost:.6f}")
                return False
            
            prev_cost = tour_cost
            print(f"Tour {i+1}: cost = {tour_cost:.6f}")
        
        print(f"SUCCESS: {len(tours)} tours validated")
        print(f"Best cost: {prev_cost:.6f}")
        
        # Additional validation assertions
        if __debug__:
            assert prev_cost == cost(tours[-1], distance_matrix), "Last tour cost mismatch"
            print("✓ Last tour is truly the best")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_output.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = check_output(input_file, output_file)
    sys.exit(0 if success else 1)
