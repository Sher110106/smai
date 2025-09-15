from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterable, Dict, Callable, Set, Deque
from collections import deque
import heapq
import argparse
import sys
import math
import time


Stage = str  # 'dirty' | 'washed' | 'dried'
Hand = Optional[Tuple[Stage, int]]  # None or (stage, load-id)


@dataclass(frozen=True)
class State:
    W: Tuple[int, ...]  # washers: +L running, -L finished, 0 empty
    D: Tuple[int, ...]  # dryers : +L running, -L finished, 0 empty
    Q: Tuple[int, ...]  # dirty loads waiting
    H: Hand             # load in hand with stage tag
    B: Tuple[int, ...]  # loads in basket


def move_gen_actions(s: State) -> List[Tuple[str, State]]:
    """Generate (action_label, successor_state) pairs.

    Actions correspond to the Task.md operators:
      - TakeDirty
      - StartWash(i)
      - PickWasher(i)
      - StartDry(j)
      - PickDryer(j)
      - Basket
      - Wait
    Indices i/j are 1-based in labels for readability.
    """
    W, D, Q, H, B = s.W, s.D, s.Q, s.H, s.B
    successors: List[Tuple[str, State]] = []

    # TakeDirty  (only if hand empty)
    if H is None and Q:
        p = Q[0]
        successors.append((
            "TakeDirty",
            State(W, D, Q[1:], ("dirty", p), B),
        ))

    # StartWash
    if H and H[0] == "dirty" and 0 in W:
        i = W.index(0)
        p = H[1]
        W2 = list(W)
        W2[i] = +p
        successors.append((
            f"StartWash({i + 1})",
            State(tuple(W2), D, Q, None, B),
        ))

    # PickWasher
    if H is None:
        for i, slot in enumerate(W):
            if slot < 0:  # finished wash
                p = -slot
                W2 = list(W)
                W2[i] = 0
                successors.append((
                    f"PickWasher({i + 1})",
                    State(tuple(W2), D, Q, ("washed", p), B),
                ))

    # StartDry
    if H and H[0] == "washed" and 0 in D:
        j = D.index(0)
        p = H[1]
        D2 = list(D)
        D2[j] = +p
        successors.append((
            f"StartDry({j + 1})",
            State(W, tuple(D2), Q, None, B),
        ))

    # PickDryer
    if H is None:
        for j, slot in enumerate(D):
            if slot < 0:  # finished dry
                p = -slot
                D2 = list(D)
                D2[j] = 0
                successors.append((
                    f"PickDryer({j + 1})",
                    State(W, tuple(D2), Q, ("dried", p), B),
                ))

    # Basket
    if H and H[0] == "dried":
        p = H[1]
        successors.append((
            "Basket",
            State(W, D, Q, None, B + (p,)),
        ))

    # Wait  (only if at least one machine is still running)
    if any(x > 0 for x in W) or any(x > 0 for x in D):
        W2 = tuple(-x if x > 0 else x for x in W)
        D2 = tuple(-x if x > 0 else x for x in D)
        successors.append((
            "Wait",
            State(W2, D2, Q, H, B),
        ))

    return successors


def goal_test(s: State, total_loads: int) -> bool:
    """True iff every load is in the basket and nothing is left elsewhere."""
    return (
        len(s.B) == total_loads and
        s.H is None and
        not s.Q and
        all(x == 0 for x in s.W) and
        all(x == 0 for x in s.D)
    )


# -------------------------------
# Problem construction
# -------------------------------

def initial_state(num_loads: int, num_washers: int, num_dryers: int) -> State:
    return State(
        W=tuple(0 for _ in range(num_washers)),
        D=tuple(0 for _ in range(num_dryers)),
        Q=tuple(range(1, num_loads + 1)),
        H=None,
        B=tuple(),
    )


# -------------------------------
# Search utilities
# -------------------------------

def reconstruct_path(
    parent: Dict[State, Optional[State]],
    action_taken: Dict[State, Optional[str]],
    goal: State,
) -> List[str]:
    actions: List[str] = []
    s: Optional[State] = goal
    while s is not None and parent[s] is not None:
        act = action_taken[s]
        if act is not None:
            actions.append(act)
        s = parent[s]
    actions.reverse()
    return actions


# -------------------------------
# BFS (shortest actions)
# -------------------------------

def bfs(start: State, total_loads: int, node_limit: Optional[int] = None, stats: Optional[Dict[str, int]] = None) -> Optional[List[str]]:
    if goal_test(start, total_loads):
        return []

    frontier: Deque[State] = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    action_taken: Dict[State, Optional[str]] = {start: None}
    visited: Set[State] = {start}

    expansions = 0

    while frontier:
        if node_limit is not None and expansions >= node_limit:
            return None
        current = frontier.popleft()
        expansions += 1

        for action_label, nxt in move_gen_actions(current):
            if nxt in visited:
                continue
            parent[nxt] = current
            action_taken[nxt] = action_label
            if goal_test(nxt, total_loads):
                if stats is not None:
                    stats["expansions"] = expansions
                return reconstruct_path(parent, action_taken, nxt)
            visited.add(nxt)
            frontier.append(nxt)

    if stats is not None:
        stats["expansions"] = expansions
    return None


# -------------------------------
# DFS (graph search)
# -------------------------------

def dfs(start: State, total_loads: int, node_limit: Optional[int] = None, stats: Optional[Dict[str, int]] = None) -> Optional[List[str]]:
    if goal_test(start, total_loads):
        return []

    stack: List[State] = [start]
    parent: Dict[State, Optional[State]] = {start: None}
    action_taken: Dict[State, Optional[str]] = {start: None}
    visited: Set[State] = {start}

    expansions = 0

    while stack:
        if node_limit is not None and expansions >= node_limit:
            return None
        current = stack.pop()
        expansions += 1

        for action_label, nxt in move_gen_actions(current):
            if nxt in visited:
                continue
            parent[nxt] = current
            action_taken[nxt] = action_label
            if goal_test(nxt, total_loads):
                if stats is not None:
                    stats["expansions"] = expansions
                return reconstruct_path(parent, action_taken, nxt)
            visited.add(nxt)
            stack.append(nxt)

    if stats is not None:
        stats["expansions"] = expansions
    return None


# -------------------------------
# Greedy Best-First Search
# -------------------------------

def heuristic_remaining_work(s: State, total_loads: int) -> int:
    """A simple heuristic that estimates remaining work.

    Counts loads not yet in basket plus penalties for items mid-process.
    This is admissible-ish for guidance but not guaranteed optimal; used
    purely for greedy best-first per the assignment brief.
    """
    in_basket = len(s.B)
    remaining = total_loads - in_basket

    # Add small penalties to prefer states closer to freeing machines/hands
    finished_in_washers = sum(1 for x in s.W if x < 0)
    finished_in_dryers = sum(1 for x in s.D if x < 0)
    running_machines = sum(1 for x in s.W if x > 0) + sum(1 for x in s.D if x > 0)

    hand_penalty = 0
    if s.H is not None:
        stage, _ = s.H
        if stage == "dirty":
            hand_penalty = 2
        elif stage == "washed":
            hand_penalty = 1
        elif stage == "dried":
            hand_penalty = 0

    # Combine components
    estimate = (
        remaining
        + finished_in_washers  # need to be picked
        + finished_in_dryers   # need to be picked
        + running_machines     # need a Wait
        + hand_penalty
    )
    return estimate


# Informed heuristic from Heuristic.md
def heuristic_informed(s: State, total_loads: int, washers: int, dryers: int) -> int:
    """Conservative lower bound on remaining atomic actions.

    h(s) = person_ops_lb + waits_lb
    """
    Bset: Set[int] = set(s.B)
    W_running: Set[int] = {x for x in s.W if x > 0}
    W_finished: Set[int] = {-x for x in s.W if x < 0}
    D_running: Set[int] = {x for x in s.D if x > 0}
    D_finished: Set[int] = {-x for x in s.D if x < 0}
    Qset: Set[int] = set(s.Q)
    H = s.H

    person_ops_lb = 0
    for p in range(1, total_loads + 1):
        if p in Bset:
            continue
        if H is not None and H[1] == p:
            stage = H[0]
            if stage == 'dried':
                person_ops_lb += 1
            elif stage == 'washed':
                person_ops_lb += 3
            elif stage == 'dirty':
                person_ops_lb += 5
            else:
                person_ops_lb += 4
            continue
        if p in D_finished:
            person_ops_lb += 2
            continue
        if p in D_running:
            person_ops_lb += 2
            continue
        if p in W_finished or p in W_running:
            person_ops_lb += 4
            continue
        if p in Qset:
            person_ops_lb += 6
            continue
        person_ops_lb += 6

    washed_or_later = Bset.union(W_finished, D_finished, W_running, D_running)
    if H is not None and H[0] in ('washed', 'dried'):
        washed_or_later.add(H[1])
    need_wash = set(range(1, total_loads + 1)) - washed_or_later

    dry_done_or_beyond = Bset.union(D_finished)
    if H is not None and H[0] == 'dried':
        dry_done_or_beyond.add(H[1])
    need_dry = set(range(1, total_loads + 1)) - dry_done_or_beyond

    waits_wash_batches = math.ceil(len(need_wash) / max(1, washers))
    waits_dry_batches = math.ceil(len(need_dry) / max(1, dryers))
    waits_lb = waits_wash_batches + waits_dry_batches

    return person_ops_lb + waits_lb


def greedy_best_first(
    start: State,
    total_loads: int,
    washers: int,
    dryers: int,
    node_limit: Optional[int] = None,
) -> Optional[List[str]]:
    if goal_test(start, total_loads):
        return []

    counter = 0
    heap: List[Tuple[int, int, State]] = []  # (h, tie, state)
    parent: Dict[State, Optional[State]] = {start: None}
    action_taken: Dict[State, Optional[str]] = {start: None}
    visited: Set[State] = {start}

    def eval_h(s: State) -> int:
        return heuristic_informed(s, total_loads, washers, dryers)

    initial_h = eval_h(start)
    heapq.heappush(heap, (initial_h, counter, start))

    expansions = 0

    while heap:
        if node_limit is not None and expansions >= node_limit:
            return None
        _, _, current = heapq.heappop(heap)
        expansions += 1

        for action_label, nxt in move_gen_actions(current):
            if nxt in visited:
                continue
            parent[nxt] = current
            action_taken[nxt] = action_label
            if goal_test(nxt, total_loads):
                return reconstruct_path(parent, action_taken, nxt)
            visited.add(nxt)
            counter += 1
            heapq.heappush(heap, (eval_h(nxt), counter, nxt))

    return None


def astar(
    start: State,
    total_loads: int,
    washers: int,
    dryers: int,
    h: Optional[Callable[[State, int], int]] = None,
    node_limit: Optional[int] = None,
    stats: Optional[Dict[str, int]] = None,
) -> Optional[List[str]]:
    if goal_test(start, total_loads):
        return []

    def eval_h(s: State) -> int:
        if h is not None:
            return h(s, total_loads)
        return heuristic_informed(s, total_loads, washers, dryers)

    counter = 0
    # (f=g+h, tie, g, state)
    heap: List[Tuple[int, int, int, State]] = []
    parent: Dict[State, Optional[State]] = {start: None}
    action_taken: Dict[State, Optional[str]] = {start: None}
    g_cost: Dict[State, int] = {start: 0}

    heapq.heappush(heap, (eval_h(start), counter, 0, start))

    expansions = 0
    closed: Set[State] = set()

    while heap:
        if node_limit is not None and expansions >= node_limit:
            return None
        f, _, g, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1

        for action_label, nxt in move_gen_actions(current):
            tentative_g = g + 1
            if nxt in closed and tentative_g >= g_cost.get(nxt, 1 << 60):
                continue
            if tentative_g < g_cost.get(nxt, 1 << 60):
                parent[nxt] = current
                action_taken[nxt] = action_label
                g_cost[nxt] = tentative_g
                if goal_test(nxt, total_loads):
                    if stats is not None:
                        stats["expansions"] = expansions
                    return reconstruct_path(parent, action_taken, nxt)
                counter += 1
                heapq.heappush(heap, (tentative_g + eval_h(nxt), counter, tentative_g, nxt))

    if stats is not None:
        stats["expansions"] = expansions
    return None


# -------------------------------
# CLI
# -------------------------------

def solve(
    loads: int,
    washers: int,
    dryers: int,
    search: str,
    node_limit: Optional[int],
) -> Optional[List[str]]:
    s0 = initial_state(loads, washers, dryers)
    if search == "bfs":
        return bfs(s0, loads, node_limit=node_limit)
    if search == "dfs":
        return dfs(s0, loads, node_limit=node_limit)
    if search in {"best", "best-first", "greedy"}:
        return greedy_best_first(s0, loads, washers=washers, dryers=dryers, node_limit=node_limit)
    if search == "astar":
        return astar(s0, loads, washers, dryers, node_limit=node_limit)
    raise ValueError(f"Unknown search: {search}")


def format_solution(actions: List[str]) -> str:
    return "\n".join(f"{i+1:>3}. {a}" for i, a in enumerate(actions))


def verify_example() -> bool:
    # K = 1, single washer and dryer schematic should be solvable
    s0 = initial_state(1, 1, 1)
    path = bfs(s0, total_loads=1, node_limit=50_000)
    return path is not None and len(path) > 0


def run_benchmark(loads: int, washers: int, dryers: int) -> None:
    s0 = initial_state(loads, washers, dryers)
    rows: List[Tuple[str, str, int, float, Optional[int]]] = []  # (algo, heuristic, actions, seconds, expansions)

    # BFS baseline
    stats: Dict[str, int] = {}
    t0 = time.perf_counter()
    sol = bfs(s0, loads, stats=stats)
    t1 = time.perf_counter()
    rows.append(("bfs", "-", len(sol) if sol else -1, t1 - t0, stats.get("expansions")))

    # Greedy (informed)
    t0 = time.perf_counter()
    sol = greedy_best_first(s0, loads, washers=washers, dryers=dryers)
    t1 = time.perf_counter()
    rows.append(("greedy", "informed", len(sol) if sol else -1, t1 - t0, None))

    # A* (informed)
    stats = {}
    t0 = time.perf_counter()
    sol = astar(s0, loads, washers, dryers, stats=stats)
    t1 = time.perf_counter()
    rows.append(("astar", "informed", len(sol) if sol else -1, t1 - t0, stats.get("expansions")))

    # Print compact comparison
    print("algo,heuristic,actions,seconds,expansions")
    for algo, heur, acts, secs, exps in rows:
        print(f"{algo},{heur},{acts},{secs:.6f},{exps if exps is not None else ''}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Laundry-room puzzle solver")
    parser.add_argument("--loads", type=int, required=True, help="Number of dirty loads (K)")
    parser.add_argument("--washers", type=int, default=3, help="Number of washers (default 3)")
    parser.add_argument("--dryers", type=int, default=3, help="Number of dryers (default 3)")
    parser.add_argument("--search", type=str, default="bfs", choices=["bfs", "dfs", "best", "astar"], help="Search algorithm: bfs / dfs / best / astar (all use informed heuristic where applicable)")
    parser.add_argument("--node-limit", type=int, default=None, help="Optional max node expansions")
    parser.add_argument("--verify", action="store_true", help="Run a basic self-check (K=1)")
    parser.add_argument("--print", dest="do_print", action="store_true", help="Print action sequence if found")
    parser.add_argument("--benchmark", action="store_true", help="Run a quick benchmark across heuristics/algs")

    args = parser.parse_args(argv)

    if args.verify:
        ok = verify_example()
        print("verify:", "OK" if ok else "FAIL")
        if not ok:
            return 2

    if args.benchmark:
        run_benchmark(args.loads, args.washers, args.dryers)
        return 0

    try:
        actions = solve(args.loads, args.washers, args.dryers, args.search, args.node_limit)
    except ValueError as e:
        print(str(e))
        return 2

    if actions is None:
        print("No solution found or node limit reached.")
        return 1

    print(f"Solution length: {len(actions)} actions")
    if args.do_print:
        print(format_solution(actions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
