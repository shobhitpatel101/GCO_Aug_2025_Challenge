import itertools
from typing import List, Tuple, Dict, Optional
import math

class PizzaDeliveryOptimizer:
    """
    Optimizes pizza delivery routes using Manhattan distance with capacity constraints.
    """
    
    def __init__(self, depot: Tuple[int, int] = (0, 0)):
        """Initialize with depot location."""
        self.depot = depot
    
    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def calculate_route_cost(self, locations: List[Tuple[int, int]], return_to_depot: bool = True) -> int:
        """
        Calculate total cost of visiting locations in given order.
        Starts from depot and optionally returns to depot.
        """
        if not locations:
            return 0
        
        total_cost = 0
        current = self.depot
        
        # Visit each location
        for location in locations:
            total_cost += self.manhattan_distance(current, location)
            current = location
        
        # Return to depot if required
        if return_to_depot:
            total_cost += self.manhattan_distance(current, self.depot)
        
        return total_cost
    
    def solve_tsp_brute_force(self, locations: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
        """
        Solve TSP using brute force (only feasible for small number of locations).
        Returns optimal route and minimum cost.
        """
        if not locations:
            return [], 0
        
        min_cost = float('inf')
        best_route = []
        
        # Try all permutations of locations
        for perm in itertools.permutations(locations):
            cost = self.calculate_route_cost(list(perm), return_to_depot=True)
            if cost < min_cost:
                min_cost = cost
                best_route = list(perm)
        
        return best_route, min_cost
    
    def solve_tsp_nearest_neighbor(self, locations: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
        """
        Solve TSP using nearest neighbor heuristic (faster for larger instances).
        Returns route and cost.
        """
        if not locations:
            return [], 0
        
        unvisited = set(locations)
        route = []
        current = self.depot
        
        while unvisited:
            # Find nearest unvisited location
            nearest = min(unvisited, key=lambda loc: self.manhattan_distance(current, loc))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        cost = self.calculate_route_cost(route, return_to_depot=True)
        return route, cost
    
    def solve_basic_delivery(self, locations: List[Tuple[int, int]], use_exact: bool = True) -> Dict:
        """
        Solve basic delivery problem without capacity constraints.
        """
        if len(locations) <= 8 and use_exact:  # Use brute force for small instances
            route, cost = self.solve_tsp_brute_force(locations)
            method = "Exact (Brute Force)"
        else:
            route, cost = self.solve_tsp_nearest_neighbor(locations)
            method = "Heuristic (Nearest Neighbor)"
        
        return {
            'method': method,
            'route': route,
            'total_cost': cost,
            'num_locations': len(locations)
        }
    
    def solve_capacity_constrained(self, locations: List[Tuple[int, int]], capacity: int) -> Dict:
        """
        Solve capacity-constrained delivery problem.
        Uses a greedy approach to group locations into trips.
        """
        if not locations:
            return {'trips': [], 'total_cost': 0, 'num_trips': 0}
        
        remaining_locations = locations.copy()
        trips = []
        total_cost = 0
        
        while remaining_locations:
            # Plan one trip with up to 'capacity' locations
            trip_locations = remaining_locations[:capacity]
            remaining_locations = remaining_locations[capacity:]
            
            # Optimize this trip using TSP
            if len(trip_locations) <= 8:
                optimized_route, trip_cost = self.solve_tsp_brute_force(trip_locations)
            else:
                optimized_route, trip_cost = self.solve_tsp_nearest_neighbor(trip_locations)
            
            trips.append({
                'locations': optimized_route,
                'cost': trip_cost,
                'num_deliveries': len(optimized_route)
            })
            total_cost += trip_cost
        
        return {
            'trips': trips,
            'total_cost': total_cost,
            'num_trips': len(trips),
            'capacity': capacity
        }
    
    def solve_capacity_constrained_smart(self, locations: List[Tuple[int, int]], capacity: int) -> Dict:
        """
        Smarter capacity-constrained solver that tries to group nearby locations.
        """
        if not locations:
            return {'trips': [], 'total_cost': 0, 'num_trips': 0}
        
        remaining_locations = set(locations)
        trips = []
        total_cost = 0
        
        while remaining_locations:
            # Start new trip
            trip_locations = []
            current = self.depot
            
            # Greedily add nearest locations until capacity is reached
            for _ in range(min(capacity, len(remaining_locations))):
                if not remaining_locations:
                    break
                
                # Find nearest remaining location
                nearest = min(remaining_locations, 
                            key=lambda loc: self.manhattan_distance(current, loc))
                trip_locations.append(nearest)
                remaining_locations.remove(nearest)
                current = nearest
            
            # Optimize this trip
            if len(trip_locations) <= 8:
                optimized_route, trip_cost = self.solve_tsp_brute_force(trip_locations)
            else:
                optimized_route, trip_cost = self.solve_tsp_nearest_neighbor(trip_locations)
            
            trips.append({
                'locations': optimized_route,
                'cost': trip_cost,
                'num_deliveries': len(optimized_route)
            })
            total_cost += trip_cost
        
        return {
            'trips': trips,
            'total_cost': total_cost,
            'num_trips': len(trips),
            'capacity': capacity,
            'method': 'Smart Grouping'
        }

def print_solution(result: Dict, problem_type: str):
    """Pretty print solution results."""
    print(f"\n=== {problem_type} ===")
    
    if problem_type == "Basic Delivery Solution":
        print(f"Method: {result['method']}")
        print(f"Total Travel Cost: {result['total_cost']}")
        print(f"Number of Locations: {result['num_locations']}")
        print("Route:", end=" ")
        print("Depot", end="")
        for loc in result['route']:
            print(f" -> {loc}", end="")
        print(" -> Depot")
    
    else:  # Capacity constrained
        print(f"Number of Trips: {result['num_trips']}")
        print(f"Total Travel Cost (All Trips Combined): {result['total_cost']}")
        print(f"Vehicle Capacity: {result['capacity']} pizzas per trip")
        if 'method' in result:
            print(f"Method: {result['method']}")
        
        print(f"\nTrip Details:")
        for i, trip in enumerate(result['trips'], 1):
            print(f"  Trip {i}:")
            print(f"    Locations Visited: {trip['locations']}")
            print(f"    Number of Deliveries: {trip['num_deliveries']}")
            print(f"    Trip Cost: {trip['cost']}")
            print("    Route: Depot", end="")
            for loc in trip['locations']:
                print(f" -> {loc}", end="")
            print(" -> Depot")

# Example usage and testing
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = PizzaDeliveryOptimizer(depot=(0, 0))
    
    # Example delivery locations
    example_locations = [
        (2, 3), (5, 1), (3, 4), (1, 2), (4, 5), (6, 2), (2, 1), (5, 4)
    ]
    
    print("Pizza Delivery Route Optimization")
    print("=" * 50)
    print(f"Depot Location: {optimizer.depot}")
    print(f"Delivery Locations: {example_locations}")
    
    # Solve basic problem
    basic_solution = optimizer.solve_basic_delivery(example_locations, use_exact=True)
    print_solution(basic_solution, "Basic Delivery Solution")
    
    # Solve with capacity constraints
    capacity = 3
    constrained_solution = optimizer.solve_capacity_constrained_smart(example_locations, capacity)
    print_solution(constrained_solution, "Capacity-Constrained Delivery Solution")
    
    # Compare with simple grouping
    simple_constrained = optimizer.solve_capacity_constrained(example_locations, capacity)
    print_solution(simple_constrained, "Simple Capacity-Constrained Solution")
    
    # Test with different capacity
    print(f"\n{'='*50}")
    print("Capacity Analysis Summary:")
    print("Capacity | Trips | Total Cost | Locations per Trip")
    print("-" * 48)
    
    for cap in [2, 4, 6, 10]:
        result = optimizer.solve_capacity_constrained_smart(example_locations, cap)
        avg_locations = len(example_locations) / result['num_trips']
        print(f"   {cap:2d}    |   {result['num_trips']:2d}  |    {result['total_cost']:3d}     |      {avg_locations:.1f}")
    
    print(f"\n{'='*50}")
    print("Detailed Example with Capacity = 3:")
    detailed_example = optimizer.solve_capacity_constrained_smart(example_locations, 3)
    
    print(f"\nNumber of Trips: {detailed_example['num_trips']}")
    print(f"Total Travel Cost (All Trips): {detailed_example['total_cost']}")
    print(f"\nTrip Breakdown:")
    for i, trip in enumerate(detailed_example['trips'], 1):
        print(f"Trip {i}: Locations {trip['locations']} → Cost: {trip['cost']}")
    print(f"\nTotal locations delivered: {len(example_locations)}")
    print(f"Average cost per location: {detailed_example['total_cost'] / len(example_locations):.1f}")
