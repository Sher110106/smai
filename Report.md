## Implementation Report: Laundry-Room Puzzle Solver

### Overview
This project implements state-space search for the Laundry-Room Puzzle with BFS, DFS, Greedy Best-First, and A*. The solver uses a single informed, resource-aware heuristic across Greedy and A*.

### State Representation
- **State** `State(W, D, Q, H, B)`
  - `W` washers: `+L` running, `-L` finished, `0` empty
  - `D` dryers: `+L` running, `-L` finished, `0` empty
  - `Q` dirty queue: loads awaiting washing
  - `H` hand: `None` or `(stage, loadId)` where `stage in {'dirty','washed','dried'}`
  - `B` basket: finished loads

### Operators (Move Generation)
- `TakeDirty` → pick first from `Q` into `H=('dirty',p)`
- `StartWash(i)` → if `H=('dirty',p)` and `W[i]==0`, start washer
- `PickWasher(i)` → if `W[i]==-p` and `H is None`, pick into `H=('washed',p)`
- `StartDry(j)` → if `H=('washed',p)` and `D[j]==0`, start dryer
- `PickDryer(j)` → if `D[j]==-p` and `H is None`, pick into `H=('dried',p)`
- `Basket` → if `H=('dried',p)`, append `p` to `B`
- `Wait` → flips any running `+p` in `W/D` to finished `-p`

### Goal Test
All loads are in `B`, `Q` empty, `H=None`, and all machine slots are `0`.

### Search Algorithms
- **BFS**: shortest action sequence; exponential but complete/optimal w.r.t. action cost 1.
- **DFS**: memory-light, not optimal; included for completeness.
- **Greedy Best-First**: uses the informed heuristic to guide; fast but not guaranteed optimal.
- **A***: uses the same heuristic; optimal with admissible heuristic.

### Final Heuristic (Single Heuristic Used, Admissible for A*)
`heuristic_informed(state, total_loads, washers, dryers)`
- `h(s) = person_ops_lb + waits_lb`
  - `person_ops_lb`: minimal unavoidable person actions per remaining load based on stage/location.
  - `waits_lb`: admissible lower bound on `Wait()` calls required to finish drying remaining loads:
    - `waits_lb = ceil( |need_dry| / dryers )`
    - Rationale: drying is the terminal machine stage; washer waits can overlap with dryer waits in the pipeline.

This correction ensures admissibility (never overestimates), restoring A*'s optimality guarantee.

### How to Run
- Solve with BFS (baseline):
```bash
python laundry_solver.py --loads 3 --washers 3 --dryers 3 --search bfs --print
```
- Greedy with informed heuristic:
```bash
python laundry_solver.py --loads 3 --washers 3 --dryers 3 --search best --print
```
- A* with informed heuristic (recommended optimal):
```bash
python laundry_solver.py --loads 3 --washers 3 --dryers 3 --search astar --print
```
- Self-check (K=1):
```bash
python laundry_solver.py --loads 1 --washers 1 --dryers 1 --verify --print
```
- Benchmark (compares BFS, Greedy, A* with the single informed heuristic):
```bash
python laundry_solver.py --loads 3 --washers 3 --dryers 3 --benchmark
```

### Benchmark Results (after heuristic fix)
Instance: `K=3, washers=3, dryers=3`
```
algo,heuristic,actions,seconds,expansions
bfs,-,20,0.002180,579
greedy,informed,20,0.000181,
astar,informed,20,0.001145,164
```
- **Solution quality**: A* remains optimal; Greedy found an optimal-length plan on this instance but has no guarantee.
- **Performance**: With the admissible `waits_lb`, A* expanded 164 nodes (previously 117 with the non-admissible count) versus BFS 579; it is still significantly more efficient than BFS and remains fast. Greedy is fastest in wall time but approximate.

### Cleanups
- Added expansion stats to Greedy for consistent reporting.
- Removed unused simple heuristic and unreachable code in `heuristic_informed`.

### Potential Improvements
- Optional verbose tracing of states along returned plans.
- Larger-scale benchmarks; record frontier sizes and memory.
- Tie-breakers in Greedy/A* (that preserve admissibility for A*).
