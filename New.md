Final TSP Implementation Plan  
=============================  
(incorporates every point raised in the two rounds of feedback)

------------------------------------------------------------
0 . What “success” looks like
------------------------------------------------------------
• Language Python 3.10 + NumPy only  
• Runtime ≤ 300 s (self-abort at ≈ 295 s)  
• Instances   50, 100, 200 nodes; Euclidean & Non-Euclidean  
• Output progressively better tours, last line evaluated  
• Target gap ≤ 12 % (200 nodes) ≤ 5 % (50 nodes)  
• Robustness handles N ≤ 3, malformed input, SIGTERM, FP noise  

------------------------------------------------------------
1 . Algorithm stack (unchanged core, tighter details)
------------------------------------------------------------
1. Diverse construction  
   a Nearest-Neighbour (three random starts)  
   b α-Random NN (α ≈ 0.2, 2 runs)  
   c Cheapest-Insertion (Christofides seed)     
   STOP when either 5 tours or 15 s spent (whichever first)

2. 2-Opt local search  
   • First-improvement, candidate lists, don’t-look bits  
   • Valid move `j > i+1  and  (i!=0 or j!=n-1)`  
   • Delta check with threshold –1 × 10⁻⁹  
   • Candidate size   
      `k = min(25, n//3) + (5 if euclid else 10)` (adaptive)

3. Perturbation  
   70 % double-bridge 30 % OR-opt-kick(k=3)  
   Gap constraint (b−a),(c−b),(d−c) ≥ max(2, n//10)

4. Iterated Local Search  
   Dynamic acceptance ε:  
      Euclid 0.0   (only better)  
      Non-Euclid  ε(t)=0.01·0.1^(t / maxIter)  (→0.1 %)  
   Restart if no better tour in 60 s (Non-Euclid only)

------------------------------------------------------------
2 . Timing policy  (295 s total wall-clock)
------------------------------------------------------------
```
construction      ≤  15 s   (max)
first 2-opt pass  ≤  45 s   (combined over starts)
ILS loop           ~235 s   (check clock every 100 iters)
buffer              ≥  5 s   (graceful exit & flush)
```

------------------------------------------------------------
3 . Data structures & speed tricks
------------------------------------------------------------
• Distance matrix D  
  –  `float64` by default; switch to `float32` if n ≥ 150  
  –  Store upper triangle only when symmetric and n ≥ 150  
   ```
   def dist(i,j):
       return D[i,j] if not half_store else D[min(i,j),max(i,j)]
   ```  
• Tour = Python list of length n (no repeat)  
• Cost   cache running sum; update with Δ in 2-opt  
• Don’t-look bit per node; reset to k after improvement  
• Vectorised candidate Δ test (batch 25 neighbours with NumPy)  

------------------------------------------------------------
4 . Edge-case & robustness checklist
------------------------------------------------------------
✓ N≤3 return trivial permutation  
✓ Symmetry test `|D[i,j]-D[j,i]|<1e-6`; else treat as asymmetric  
✓ Floating-point guard abs(Δ) > 1e-9  
✓ SIGTERM / Ctrl-C → flush best tour & exit(0)  
✓ Output validator (unit test) ensures strictly better costs line-by-line  

------------------------------------------------------------
5 . Code layout (root of ZIP)
------------------------------------------------------------
main.py             CLI, timer, orchestration  
io_utils.py         parse_input(), write_tour()  
construct.py        NN, α-NN, cheapest_insertion  
two_opt.py          candidate builder, don’t-look 2-opt  
perturb.py          double_bridge(), or_opt_kick()  
ils.py              meta-loop, adaptive ε, restarts  
utils.py            cost(), RNG seed, profiler helpers  

------------------------------------------------------------
6 . Development timeline (3 weeks, team of 3)
------------------------------------------------------------
Week 1   end-to-end skeleton  
  • I/O + cost + simple NN + naïve 2-opt (no candidates)  
  • Integration test on sample files  
Week 2   speed & quality  
  • Candidate lists, don’t-look bits, vectorised Δ  
  • Cheapest-Insertion, α-NN, double-bridge, OR-opt  
  • Unit tests + cProfile (goal: 2-opt sweep <0.05 s @ n=200)  
Week 3   tuning & hardening  
  • Adaptive ε, dynamic k, memory trim  
  • Timeout & signal tests, random seed toggle  
  • Handwritten report (2 pages) + ablation table  
  • Stretch (only if green): parallel ILS or shallow LK-move  

------------------------------------------------------------
7 . Report outline
------------------------------------------------------------
1. Problem & constraints (3 lines)  
2. Algorithm pipeline diagram  
3. Key engineering tricks (candidate lists, timing, flushing)  
4. Experimental results table (6 hidden + 3 public instances)  
5. Ablation study (–cands, –perturb, –ε)  
6. Lessons & future work (LK-opt, GPU, TSPLIB replay)

------------------------------------------------------------
8 . Submission-day checklist
------------------------------------------------------------
☐ `python3 main.py sample.txt out.txt` runs in clean Docker Ubuntu 22.04  
☐ Last line of out.txt has lowest cost seen  
☐ Runtime < 295 s on worst-case sample N=200  
☐ ZIP name `id1_id2_id3.zip`; contains only *.py + handwritten PDF/JPG  
☐ No external libs; NumPy only  
☐ Random seed removed or toggled via `--seed` flag  

------------------------------------------------------------
Stretch but safe
------------------------------------------------------------
• Parallel ILS using `concurrent.futures.ProcessPoolExecutor`, share best cost via `multiprocessing.Value` (try only if week-3 slack)  
• Depth-5 LK-chain every 30 s on current best (skip if runtime spikes)  

------------------------------------------------------------
Ready to implement 🚀
------------------------------------------------------------
This plan now integrates:  
– aggressive yet safe time split,  
– adaptive candidate sizing & ε,  
– memory downgrades for big N,  
– explicit profiling targets,  
– clear stop-the-scope guardrails.  

Follow the timeline, keep tests green, and you’re on track for a top-quartile score. Good coding & good luck!


Format:
AI3002 – Search Methods in AI
Travelling Salesman Problem
2025-09-26
Given a set of N cities and the distance between every pair of cities, find the shortest possible route
that starts from a city and visits every city exactly once and returns to the starting city. Write a
program to find as short a tour as possible of the N cities.
This assignment carries 7% of your course grade. The assignment has to be done in teams of 3. Only
one member from each team needs to submit the assignment.
1. Input Description
Your program must read from a file with the following structure:
• First line: either EUCLIDEAN or NON-EUCLIDEAN.
• Second line: an integer N, number of cities.
• Next N lines: each line contains N distance values, i.e., 𝑎ij is distance between 𝑖th and 𝑗th node.
The distance matrix contains floating point numbers. Use the distance matrix for computing tour
cost.
For example, a Euclidean TSP for 𝑁 = 5 might have the following input:
1 EUCLIDEAN
2 5
3 0.0 4.597411 9.110738 3.187861 4.302992
4 4.597411 0.0 13.285327 5.736168 4.420019
5 9.110738 13.285327 0.0 7.822640 10.096052
6 3.187861 5.736168 7.822640 0.0 2.419287
7 4.302992 4.420019 10.096052 2.419287 0.0
Similarly, a Non-Eucledian TSP for 𝑁 = 3 might have the following input:
1 NON-EUCLIDEAN
2 3
3 0.0 100.0 2.0
4 100.0 0.0 5.0
5 2.0 5.0 0.0
Note:
1. For 𝑁 nodes, the nodes are labelled from 0 to 𝑁 − 1.
2. Distance matrix represents an undirected fully connected graph. The diagonal elements are 0 and
every non-diagonal element has a finite positive value.
Sample input files are uploaded for your reference.
2. Output Description
Each line of your output should contain a valid tour in path representation. You should keep
writing to the output file as you discover better solutions (with lower costs). The last line of the
output will be used for evaluation. If you do not output any tours, it will lead to a score of 0 in that
test case.
1
AI3002 – Search Methods in AI Travelling Salesman Problem
For 𝑁 = 5, following is a valid output file -
1 0 2 1 3 4
2 1 2 4 3 0
3 4 3 1 2 0
In this case, the tour 4 3 1 2 0 will be used for evaluation.
3. Evaluation
There will be 6 test cases -
1. EUCLEDIAN (𝑁 = 50)
2. EUCLEDIAN (𝑁 = 100)
3. EUCLEDIAN (𝑁 = 200)
4. NON-EUCLEDIAN (𝑁 = 50)
5. NON-EUCLEDIAN (𝑁 = 100)
6. NON-EUCLEDIAN (𝑁 = 200)
Your program will be run for each test case with a timeout of 300𝑠. We will terminate your program
after this limit (if you do not exit early). The last line of your output file will be used for evaluation.
Grading is relative, the best tour will get the highest points. . The team with the highest cost
for a test case will get the lowest score.
4. Final Submission Details (6%)
4.1. Code Submission
You are expected to submit a .zip file with the main.py at the root. Auxiliary/helper files are allowed
and you should import them in your main.py. You are only allowed to use Python3.
The auto-grader will run your code with the following format -
1 python3 main.py /path/to/input.txt /path/to/output.txt
1. Make sure your code runs on a Linux environment.
2. You will be allowed to use NumPy for any matrix operations. You are not allowed to use any
other external libraries.
3. If your program has any execution or runtime errors for the aforementioned input/output
formats, zero marks will be awarded.
Your .zip file should be named id1_id2_id3.zip. Example - U20230001_U20230013_U20230134.zip
4.2. Report Submission
Your code submission should be accompanied with a handwritten report (upto 2 A4 pages) detailing
your methodology, experiments, and approach. This report is not graded but is mandatory for the
code submission. Failing to submit the report or writing the report in an incomprehensible manner
will lead to a 0 grade in the assignment.
2
AI3002 – Search Methods in AI Travelling Salesman Problem
5. Trial Submission Details (1%)
You can choose to make a trial code submission on or before October 6, 2025. Your code submission
should be in the same format as above, and should execute successfully (with at least 1 valid tour
within the time limit). This will allow you to check if your code runs successfully on our Linux
environment.