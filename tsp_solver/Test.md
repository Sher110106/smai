Below is a “check-list + tool-kit” you can run through in < ½ day to be confident the solver is (1) correct, (2) fast enough, and (3) robust against the judge’s quirks.


────────────────────────────────────────────
1.  Functional correctness
────────────────────────────────────────────
1.1  Unit tests (pytest or unittest)
- cost() returns 0 for tour length 1
- cost() is symmetric: cost(t) == cost(t[::-1])
- two_opt() does not change the multiset of cities
- double_bridge() and or_opt_kick() always produce permutations

1.2  Exact-answer sanity
For N ≤ 8-10 you can brute-force optimal cost.

```python
# oracle_bruteforce.py
import itertools, numpy as np, sys

D = np.loadtxt(sys.argv[1], skiprows=2)
n = D.shape[0]
best = 1e30
for perm in itertools.permutations(range(1, n)):
    tour = (0,) + perm
    cost = sum(D[tour[i], tour[(i+1)%n]] for i in range(n))
    best = min(best, cost)
print(best)
```

Run:

```bash
python oracle_bruteforce.py tests/data/tiny_sym_5.txt
python main.py tests/data/tiny_sym_5.txt out.txt
python oracle_bruteforce.py out.txt   # script that parses last line
```

Your last-line cost must equal the optimum for those tiny cases.

1.3  Invariants on every produced tour  
Add in‐code asserts (only under `if __debug__:`):
- len(tour) == N  
- len(set(tour)) == N  
- all(0 ≤ v < N for v in tour)

────────────────────────────────────────────
2.  Output-stream validation
────────────────────────────────────────────
Write a small checker:

```python
def check_output(input_path, output_path):
    D = load_matrix(input_path)
    tours = [list(map(int,l.split())) for l in open(output_path)]
    assert tours, "no tours produced"
    prev_cost = 1e100
    for t in tours:
        assert len(set(t)) == len(t) == len(D)
        c = tour_cost(t, D)
        assert c <= prev_cost + 1e-6, "cost must be non-increasing"
        prev_cost = c
```

Run it after every unit/integration test.

────────────────────────────────────────────
3.  Performance tests
────────────────────────────────────────────
3.1  Profile hot spots  
```bash
python -m cProfile -s cumtime main.py tests/data/random_euc_200.txt out.txt
```
Targets:
- ≤ 0.05 s per complete 2-opt sweep (200 nodes)  
- ≤ 295 s overall; typical run should finish ~260 s

3.2  Stress with random matrices  
```python
import numpy as np, tempfile, subprocess, time, os
for n in (50, 100, 200):
    D = np.random.rand(n, n)
    D = (D + D.T)/2         # symmetric
    path = f"rand_{n}.txt"
    write_instance(path, D)
    t0 = time.time()
    subprocess.run(["python3","main.py", path, "out.txt"], timeout=310)
    print(n, "t = %.1fs" % (time.time()-t0))
```

Look for crashes, assertions, or >300 s runtime.

3.3  SIGTERM / OOM resilience  
```bash
# sigterm_wrapper.sh
python3 main.py "$1" "$2" &
pid=$!
sleep 5          # kill after 5 s
kill -TERM $pid  # judge uses SIGTERM
wait $pid
```
Check that `out.txt` still contains at least one tour and passes the checker.

────────────────────────────────────────────
4.  Quality regression guard
────────────────────────────────────────────
Keep a CSV bench file:

| date | commit | n=50 gap | n=100 gap | run-time |
|------|--------|----------|-----------|----------|

Automatically append to it in CI (GitHub Actions / GitLab CI) so you immediately see if a code change worsens tour quality.

────────────────────────────────────────────
5.  Reproduce judge environment
────────────────────────────────────────────
Dockerfile snippet:

```dockerfile
FROM python:3.10-slim
RUN pip install numpy==1.26.4
COPY . /app
WORKDIR /app
ENTRYPOINT ["python3","main.py"]
```

Build & run:

```bash
docker build -t tsp-solver .
docker run --rm -v $PWD/tests/data:/data tsp-solver /data/random_euc_200.txt /tmp/out.txt
```

Gives you 99 % confidence the grader will not surprise you.

────────────────────────────────────────────
6.  Final dry-run script (to package with report)
────────────────────────────────────────────
`harness.py`

```python
import glob, os, subprocess, time, checker

for f in glob.glob("tests/data/*.txt"):
    t0=time.time()
    subprocess.run(["python3","main.py", f, "out.txt"], timeout=310, check=True)
    checker.check_output(f,"out.txt")
    print(f, "OK  (%.1fs)"%(time.time()-t0))
```

One command confirms functional, performance, and format for every sample.

