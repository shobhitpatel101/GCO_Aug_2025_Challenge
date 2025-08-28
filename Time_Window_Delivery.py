from itertools import combinations
from math import inf
from typing import List, Tuple, Dict, Optional

# -----------------------------
# Data model and utilities
# -----------------------------
Point = Tuple[int, int]
Stop  = Tuple[str, int, int, int]  # (name, x, y, pizzas)  pizzas not used here

DEPOT: Point = (0, 0)

def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def hm_to_min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def min_to_hm(m: int) -> str:
    m = int(m)
    h = (m // 60) % 24
    return f"{h:02d}:{m % 60:02d}"

# -----------------------------
# Base problem: Held–Karp TSP (exact)
# -----------------------------
def tsp_held_karp(points: List[Point]) -> Tuple[List[int], int]:
    """
    Returns (order_indices, total_cost), where order_indices are indices in `points`
    (excluding depot) in visiting order; total_cost includes depot->...->depot.
    """
    n = len(points)
    if n == 0:
        return [], 0

    # Distances
    d = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d[i][j] = manhattan(points[i], points[j])
    dep_to = [manhattan(DEPOT, p) for p in points]

    # DP: dp[(mask, j)] = (cost, prev)
    dp: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for j in range(n):
        dp[(1 << j, j)] = (dep_to[j], -1)

    for m in range(2, n+1):
        for subset_tuple in combinations(range(n), m):
            mask = 0
            for i in subset_tuple: mask |= (1 << i)
            for j in subset_tuple:
                prev_mask = mask ^ (1 << j)
                best = (inf, -1)
                for k in subset_tuple:
                    if k == j: continue
                    prev_cost, _ = dp[(prev_mask, k)]
                    cand = prev_cost + d[k][j]
                    if cand < best[0]:
                        best = (cand, k)
                dp[(mask, j)] = best

    full = (1 << n) - 1
    best_end = (inf, -1)
    for j in range(n):
        cost_to_j, _ = dp[(full, j)]
        cand = cost_to_j + dep_to[j]
        if cand < best_end[0]:
            best_end = (cand, j)

    # Reconstruct order
    order = []
    mask = full
    j = best_end[1]
    while j != -1:
        order.append(j)
        _, j = dp[(mask, j)]
        mask ^= (1 << order[-1])
    order.reverse()
    return order, best_end[0]

# -----------------------------
# TSP with Time Windows (exact label-setting DP)
# -----------------------------
def tsp_time_windows(points: List[Point],
                     earliest: List[int],
                     latest: List[int],
                     depot_earliest: int = 0,
                     depot_latest: int = 10**9) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[int]]:
    """
    Single-vehicle TSP with time windows.
    - points: list of N customer points (no depot).
    - earliest[i], latest[i]: time-window bounds (inclusive) in minutes for customer i.
    - depot has window [depot_earliest, depot_latest] (latest typically very large).
    Travel time = Manhattan distance (1 minute per unit).
    Returns (order_indices, arrival_times, total_travel_cost) or (None, None, None) if infeasible.
    """
    n = len(points)
    if n == 0:
        return [], [], 0

    # Earliest feasible arrival times DP:
    # dp[(mask, j)] = (arrival_time_at_j, prev_index)
    # mask includes j.
    dp: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # Initialize from depot -> j
    for j in range(n):
        t = manhattan(DEPOT, points[j])
        t = max(t + depot_earliest, earliest[j])  # waiting if early
        if t <= latest[j]:
            dp[(1 << j, j)] = (t, -1)

    # Transitions
    for m in range(2, n+1):
        for subset_tuple in combinations(range(n), m):
            mask = 0
            for i in subset_tuple: mask |= (1 << i)
            for j in subset_tuple:
                prev_mask = mask ^ (1 << j)
                best = (inf, -1)
                for k in subset_tuple:
                    if k == j: continue
                    if (prev_mask, k) not in dp: continue
                    arr_k, _ = dp[(prev_mask, k)]
                    t = arr_k + manhattan(points[k], points[j])
                    t = max(t, earliest[j])  # wait if early
                    if t <= latest[j] and t < best[0]:
                        best = (t, k)
                if best[0] < inf:
                    dp[(mask, j)] = best

    # Close tour to depot and choose best end
    full = (1 << n) - 1
    best_total = (inf, -1, -1)  # (arrival_time_at_j, j, final_time_at_depot)
    for j in range(n):
        if (full, j) not in dp: continue
        arr_j, _ = dp[(full, j)]
        t_back = arr_j + manhattan(points[j], DEPOT)
        if depot_earliest <= t_back <= depot_latest and t_back < best_total[0] + (t_back - arr_j):
            best_total = (arr_j, j, t_back)

    if best_total[1] == -1:
        return None, None, None  # infeasible

    # Reconstruct order
    order = []
    times = []
    mask = full
    j = best_total[1]
    while j != -1:
        arr_j, prev = dp[(mask, j)]
        order.append(j)
        times.append(arr_j)
        mask ^= (1 << j)
        j = prev
    order.reverse()
    times.reverse()
    total_cost = best_total[2]  # since starting at t=0 and travel time == distance
    return order, times, total_cost

# -----------------------------
# Sample input (from your prompt)
# -----------------------------
stops: List[Stop] = [
    ("L1",   5, -10, 3),
    ("L2", -12,   8, 5),
    ("L3",  15,  20, 2),
    ("L4",  -8, -15, 7),
    ("L5",  25,  -5, 4),
    ("L6", -20,  18, 6),
    ("L7",  10,  30, 1),
    ("L8", -30, -22, 8),
    ("L9",  18,  14, 2),
    ("L10", -7,  28, 5),
]

# Optional time info (starts taken from your table image)
start_times = {
    "L1":  "09:05",
    "L2":  "10:30",
    "L3":  "12:00",
    "L4":  "13:30",
    "L5":  "15:00",
    "L6":  "16:30",
    "L7":  "18:00",
    "L8":  "19:30",
    "L9":  "21:00",
    "L10": "22:30",
}

# Build arrays for algorithms
names = [s[0] for s in stops]
pts   = [(s[1], s[2]) for s in stops]

# -----------------------------
# Run: Base problem
# -----------------------------
order_idx, base_cost = tsp_held_karp(pts)
order_names = [names[i] for i in order_idx]

print("=== Base Problem (TSP with Manhattan distance) ===")
print("Tour (with depot):", ["DEPOT"] + order_names + ["DEPOT"])
print("Minimum total travel cost:", base_cost)
print()

# -----------------------------
# Run: Time-window problem
# -----------------------------
# If you only have starts, choose a window width (minutes). Change if needed.
DEFAULT_WINDOW_MINUTES = 120  # <- tweak here
earliest = [hm_to_min(start_times[n]) for n in names]
latest   = [e + DEFAULT_WINDOW_MINUTES for e in earliest]

tw_order_idx, tw_arrivals, tw_total = tsp_time_windows(
    pts, earliest, latest,
    depot_earliest=0, depot_latest=10**9
)

print("=== Time-Window Delivery (1 min per Manhattan unit) ===")
if tw_order_idx is None:
    print("No feasible route found with the given time windows.")
else:
    seq = [names[i] for i in tw_order_idx]
    arrivals_hm = [min_to_hm(t) for t in tw_arrivals]
    print("Route (with depot):", ["DEPOT"] + seq + ["DEPOT"])
    print("Arrival times at each stop:")
    for nm, tmin, thm in zip(seq, tw_arrivals, arrivals_hm):
        print(f"  {nm}: {thm}  (t = {tmin} min)")
    print("Total travel cost (distance) including return to depot:", tw_total)
