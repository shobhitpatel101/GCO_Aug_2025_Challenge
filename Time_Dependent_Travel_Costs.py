from typing import List, Tuple
import math

Location = Tuple[int, int, int]  # (x, y, P)

# ---------- Helpers
def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def parse_time_str(t: str) -> int:
    hh, mm = t.split(":")
    return int(hh)*60 + int(mm)

def in_window(t_min: int, start_h: int, end_h: int) -> bool:
    """Return True if absolute minutes t_min is within [start_h, end_h) hours."""
    start = start_h*60
    end = end_h*60
    return start <= t_min < end

# ---------- Base Problem: TSP (Held-Karp) with Manhattan distances
def tsp_min_cost_manhattan(locations: List[Location]) -> int:
    """
    Depot is index 0 at (0,0); nodes 1..N are locations.
    Returns minimal tour cost (return to depot).
    """
    coords = [(0,0)] + [(x,y) for (x,y,_) in locations]
    n = len(coords)
    # Distances
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = manhattan(coords[i], coords[j])

    size = n-1
    INF = 10**15
    dp = [[INF]*n for _ in range(1<<size)]
    parent = [[-1]*n for _ in range(1<<size)]

    # Base: 0 -> i
    for i in range(1, n):
        mask = 1<<(i-1)
        dp[mask][i] = dist[0][i]
        parent[mask][i] = 0

    # Transitions
    for mask in range(1<<size):
        for last in range(1, n):
            if not (mask & (1<<(last-1))):
                continue
            prev_mask = mask ^ (1<<(last-1))
            if prev_mask == 0:
                continue
            best = dp[mask][last]
            pm = prev_mask
            k = 1
            while pm:
                if pm & 1:
                    node = k
                    cand = dp[prev_mask][node] + dist[node][last]
                    if cand < best:
                        best = cand
                        parent[mask][last] = node
                        dp[mask][last] = cand
                pm >>= 1
                k += 1

    full = (1<<size) - 1
    best_total = INF
    best_last = -1
    for last in range(1, n):
        cand = dp[full][last] + dist[last][0]
        if cand < best_total:
            best_total = cand
            best_last = last
    return best_total

# ---------- Time-dependent multipliers & DP over subsets with arrival times
def time_multiplier(dest_x: int, dest_y: int, departure_min: int) -> float:
    """
    Multipliers:
      - rush_east:   if destination x > 0 and depart in [17:00, 19:00) -> x2.0
      - evening_north: if destination y > 10 and depart in [16:00, 18:00) -> x1.5
    """
    m = 1.0
    if dest_x > 0 and in_window(departure_min, 17, 19):
        m *= 2.0
    if dest_y > 10 and in_window(departure_min, 16, 18):
        m *= 1.5
    return m

def solve_time_dependent(locations: List[Location], start_time_str: str):
    """
    Exact DP that stores best (earliest) arrival time at each (subset,last) state.
    Leg time = Manhattan distance * multiplier(dest, departure_time).
    """
    coords = [(0,0,0)] + locations  # include depot
    n = len(coords)

    # Distances
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = manhattan(coords[i][:2], coords[j][:2])

    size = n-1
    INF = 10**15
    dp = [[INF]*n for _ in range(1<<size)]
    parent = [[-1]*n for _ in range(1<<size)]
    start_min = parse_time_str(start_time_str)

    # Init: 0 -> j
    for j in range(1, n):
        m = time_multiplier(coords[j][0], coords[j][1], start_min)
        travel = dist[0][j] * m
        dp[1<<(j-1)][j] = start_min + travel
        parent[1<<(j-1)][j] = 0

    # Transitions
    for mask in range(1<<size):
        for last in range(1, n):
            if not (mask & (1<<(last-1))):
                continue
            arrival_at_last = dp[mask][last]
            if arrival_at_last >= INF:
                continue
            for j in range(1, n):
                if mask & (1<<(j-1)):
                    continue
                depart_time = arrival_at_last  # leave immediately
                m = time_multiplier(coords[j][0], coords[j][1], math.floor(depart_time))
                travel = dist[last][j] * m
                new_arrival = depart_time + travel
                next_mask = mask | (1<<(j-1))
                if new_arrival < dp[next_mask][j]:
                    dp[next_mask][j] = new_arrival
                    parent[next_mask][j] = last

    # Close tour (j -> 0), no multiplier for depot
    full = (1<<size) - 1
    best_total_time = INF
    best_last = -1
    for last in range(1, n):
        if dp[full][last] >= INF:
            continue
        total_time = dp[full][last] + dist[last][0]
        if total_time < best_total_time:
            best_total_time = total_time
            best_last = last

    # Reconstruct route indices (0-based with depot=0)
    route = [0]
    mask = full
    last = best_last
    order_rev = []
    while last != 0 and last != -1:
        order_rev.append(last)
        prev = parent[mask][last]
        mask ^= 1<<(last-1)
        last = prev
    route += list(reversed(order_rev))
    route.append(0)

    # Build leg-by-leg timings
    timeline = []
    t = start_min
    for idx in range(len(route)-1):
        u = route[idx]
        v = route[idx+1]
        mult = 1.0 if v == 0 else time_multiplier(coords[v][0], coords[v][1], math.floor(t))
        base = dist[u][v]
        cost = base * mult
        leg = {
            "from": u,
            "to": v,
            "depart_time_min": t,
            "depart_time_hhmm": f"{int(t//60):02d}:{int(round(t%60)):02d}",
            "multiplier": mult,
            "base_distance": base,
            "leg_cost": cost,
            "arrive_time_min": t + cost,
            "arrive_time_hhmm": f"{int((t+cost)//60):02d}:{int(round((t+cost)%60)):02d}"
        }
        timeline.append(leg)
        t += cost

    return {
        "total_time_dependent_cost": best_total_time - start_min,
        "finish_time_hhmm": f"{int(best_total_time//60):02d}:{int(round(best_total_time%60)):02d}",
        "route_indices": route,
        "timeline": timeline
    }

# ---------- Example usage with your sample
if __name__ == "__main__":
    locations = [
        (5, -10, 3), (-12, 8, 5), (15, 20, 2), (-8, -15, 7), (25, -5, 4),
        (-20, 18, 6), (10, 30, 1), (-30, -22, 8), (18, 14, 2), (-7, 28, 5)
    ]

    base_min_cost = tsp_min_cost_manhattan(locations)
    print("Base Problem — Minimum Manhattan tour cost (return to depot):", base_min_cost)

    td_result = solve_time_dependent(locations, start_time_str="16:30")
    print("\nTime-Dependent Solution")
    print("Total time-dependent travel cost:", td_result["total_time_dependent_cost"])
    print("Finish time:", td_result["finish_time_hhmm"])
    print("Route (by index):", td_result["route_indices"])
    print("Index 0 is the depot at (0,0). Indices 1..N map to the sample locations in the given order.\n")
    for i, leg in enumerate(td_result["timeline"], 1):
        print(f"Leg {i}: {leg['from']} -> {leg['to']} | Depart {leg['depart_time_hhmm']} | "
              f"mult {leg['multiplier']} | base {leg['base_distance']} | "
              f"cost {leg['leg_cost']:.1f} | Arrive {leg['arrive_time_hhmm']}")
