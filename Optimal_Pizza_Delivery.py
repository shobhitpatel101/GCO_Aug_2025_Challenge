import itertools
import numpy as np
from typing import List, Tuple
import time

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def calculate_route_cost(route: List[int], locations: List[Tuple[int, int, int]], depot: Tuple[int, int] = (0, 0)) -> int:
    """Calculate total cost of a route including return to depot."""
    total_cost = 0
    current_pos = depot
    
    # Visit all locations in route order
    for location_idx in route:
        next_pos = (locations[location_idx][0], locations[location_idx][1])
        total_cost += manhattan_distance(current_pos, next_pos)
        current_pos = next_pos
    
    # Return to depot
    total_cost += manhattan_distance(current_pos, depot)
    return total_cost

def solve_tsp_brute_force(locations: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """Solve TSP using brute force - optimal but slow for large instances."""
    n = len(locations)
    best_cost = float('inf')
    best_route = None
    
    # Try all possible permutations
    for perm in itertools.permutations(range(n)):
        cost = calculate_route_cost(list(perm), locations)
        if cost < best_cost:
            best_cost = cost
            best_route = list(perm)
    
    return best_cost, best_route

def solve_tsp_nearest_neighbor(locations: List[Tuple[int, int, int]]) -> Tuple[int, List[int]]:
    """Solve TSP using nearest neighbor heuristic - fast but not optimal."""
    n = len(locations)
    unvisited = set(range(n))
    route = []
    current_pos = (0, 0)  # depot
    
    while unvisited:
        # Find nearest unvisited location
        nearest_idx = min(unvisited, 
                         key=lambda i: manhattan_distance(current_pos, (locations[i][0], locations[i][1])))
        
        route.append(nearest_idx)
        unvisited.remove(nearest_idx)
        current_pos = (locations[nearest_idx][0], locations[nearest_idx][1])
    
    cost = calculate_route_cost(route, locations)
    return cost, route

def solve_tsp_2opt(locations: List[Tuple[int, int, int]], initial_route: List[int] = None) -> Tuple[int, List[int]]:
    """Improve a route using 2-opt local search."""
    n = len(locations)
    
    if initial_route is None:
        # Start with nearest neighbor solution
        _, route = solve_tsp_nearest_neighbor(locations)
    else:
        route = initial_route.copy()
    
    improved = True
    while improved:
        improved = False
        best_cost = calculate_route_cost(route, locations)
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Create new route by reversing segment between i and j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_cost = calculate_route_cost(new_route, locations)
                
                if new_cost < best_cost:
                    route = new_route
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    
    return calculate_route_cost(route, locations), route

def solve_pizza_delivery_optimal(locations: List[Tuple[int, int, int]]) -> Tuple[int, List[int], dict]:
    """Solve pizza delivery problem using multiple approaches and return the best."""
    n = len(locations)
    results = {}
    
    print(f"Solving pizza delivery problem for {n} locations...")
    print("Locations:", locations)
    print()
    
    # Method 1: Brute Force (optimal for small instances)
    if n <= 10:  # Only use brute force for reasonable size
        print("Method 1: Brute Force (Optimal)")
        start_time = time.time()
        bf_cost, bf_route = solve_tsp_brute_force(locations)
        bf_time = time.time() - start_time
        results['brute_force'] = {'cost': bf_cost, 'route': bf_route, 'time': bf_time}
        print(f"  Cost: {bf_cost}")
        print(f"  Route: {bf_route}")
        print(f"  Time: {bf_time:.4f} seconds")
        print()
    
    # Method 2: Nearest Neighbor Heuristic
    print("Method 2: Nearest Neighbor Heuristic")
    start_time = time.time()
    nn_cost, nn_route = solve_tsp_nearest_neighbor(locations)
    nn_time = time.time() - start_time
    results['nearest_neighbor'] = {'cost': nn_cost, 'route': nn_route, 'time': nn_time}
    print(f"  Cost: {nn_cost}")
    print(f"  Route: {nn_route}")
    print(f"  Time: {nn_time:.4f} seconds")
    print()
    
    # Method 3: 2-opt improvement on nearest neighbor
    print("Method 3: 2-opt Local Search")
    start_time = time.time()
    opt2_cost, opt2_route = solve_tsp_2opt(locations, nn_route)
    opt2_time = time.time() - start_time
    results['2opt'] = {'cost': opt2_cost, 'route': opt2_route, 'time': opt2_time}
    print(f"  Cost: {opt2_cost}")
    print(f"  Route: {opt2_route}")
    print(f"  Time: {opt2_time:.4f} seconds")
    print()
    
    # Find best solution
    best_method = min(results.keys(), key=lambda k: results[k]['cost'])
    best_cost = results[best_method]['cost']
    best_route = results[best_method]['route']
    
    print("=" * 50)
    print("OPTIMAL SOLUTION")
    print("=" * 50)
    print(f"Best method: {best_method}")
    print(f"Minimum total travel cost: {best_cost}")
    print(f"Optimal route (location indices): {best_route}")
    
    # Print detailed route
    print("\nDetailed route:")
    current_pos = (0, 0)
    total_distance = 0
    print(f"Start: Depot {current_pos}")
    
    for i, loc_idx in enumerate(best_route):
        next_pos = (locations[loc_idx][0], locations[loc_idx][1])
        distance = manhattan_distance(current_pos, next_pos)
        total_distance += distance
        pizzas = locations[loc_idx][2]
        print(f"Step {i+1}: Go to location {loc_idx} {next_pos} (deliver {pizzas} pizzas) - Distance: {distance}")
        current_pos = next_pos
    
    # Return to depot
    depot_distance = manhattan_distance(current_pos, (0, 0))
    total_distance += depot_distance
    print(f"Final: Return to depot (0, 0) - Distance: {depot_distance}")
    print(f"Total distance: {total_distance}")
    
    return best_cost, best_route, results

# Sample input data
sample_locations = [
    (5, -10, 3),
    (-12, 8, 5),
    (15, 20, 2),
    (-8, -15, 7),
    (25, -5, 4),
    (-20, 18, 6),
    (10, 30, 1),
    (-30, -22, 8),
    (18, 14, 2),
    (-7, 28, 5)
]

# Run the optimization
optimal_cost, optimal_route, all_results = solve_pizza_delivery_optimal(sample_locations)

print(f"\n🍕 FINAL ANSWER: {optimal_cost}")
