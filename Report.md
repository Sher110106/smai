## Implementation Report: Laundry-Room Puzzle Solver

### Overview
This project implements state-space search for the Laundry-Room Puzzle with BFS, DFS, Greedy Best-First, and A*. The final solver uses a single informed, resource-aware heuristic across Greedy and A*.

### State Representation
- **State** `State(W, D, Q, H, B)`
  - `W` washers: `+L` running, `-L` finished, `0` empty
  - `D` dryers: `+L` running, `-L` finished, `0` empty
  - `Q` dirty queue: loads awaiting washing
  - `H` hand: `None` or `(stage, loadId)` where `stage in {'dirty','washed','dried'}`
  - `B` basket: finished loads

This encoding guarantees the strict sequence DIRTY → WASH → DRY → BASKET and avoids illegal states by generating only legal successors.

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
- **Greedy Best-First (informed)**: uses heuristic to guide; fast but not guaranteed optimal.
- **A***: uses the same heuristic; optimal with much smaller search than BFS.

### Final Heuristic (Single Heuristic Used)
The solver uses one informed, resource-aware heuristic `heuristic_informed(state, total_loads, washers, dryers)`:
- `h(s) = person_ops_lb + waits_lb`
  - `person_ops_lb`: sum of minimal unavoidable person actions per remaining load based on its stage/location
  - `waits_lb`: lower bound on unavoidable `Wait()` calls estimated via machine throughput: `ceil(need_wash/washers) + ceil(need_dry/dryers)`

This is conservative (lower bound) and thus suitable for A*. It captures both hand-operated actions and machine-parallelism constraints, steering search to unblock machines and maintain pipeline flow.

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

### Benchmark Results (Mac, Python 3)
Instance: `K=3, washers=3, dryers=3`
```
algo,heuristic,actions,seconds,expansions
bfs,-,20,0.002228,579
greedy,informed,20,0.000175,
astar,informed,20,0.000751,117
```
- **Solution quality**: Greedy found an optimal-length plan (20) on this instance but offers no guarantees. A* guarantees optimality by design.
- **Performance**: A* with the informed heuristic expanded ~117 nodes vs BFS 579 (≈5x reduction) and is still very fast. Greedy is the fastest but may yield suboptimal solutions on other instances.

### Discussion and Trade-offs
- The informed heuristic substantially shrinks the explored space versus uninformed BFS by aligning with unavoidable work (person ops + machine batches).
- Greedy is a pragmatic choice for large K when an approximate but often-good plan suffices.
- A* is the recommended default when optimality matters; it scales better than BFS due to the heuristic’s guidance.

### Potential Improvements
- Add optional verbose tracing of states along the returned plan to illuminate operator choices.
- Extend benchmarking to larger `K` and varied machine counts; log nodes expanded and peak frontier size.
- Explore slight tie-breakers in greedy/A* (e.g., prefer more items in `B`) to reduce wall time further while maintaining admissibility for A*.
