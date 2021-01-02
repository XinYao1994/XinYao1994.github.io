
1. Organisation
2. Distributed algorithms
   - In distributed algorithms, the problem is given by the network itself and solved by coordination protocols
3. Graphs and spanning trees revisited
   - Tree edge: an edge that belongs to the spanning tree
   - Frond edge: an edge that does not belong to the spanning tree
4. Basics
   - Synchronous models
      - nodes work totally synchronised – in lock-step
      - easiest to develop, but often unrealistic and less efficient
      - all node process (1) + all message (0)
      - all node process (0) + all message (1)
   - Asynchronous models
      - messages can take an arbitrary unbounded time to arrive
      - often unrealistic and sometimes impossible
      - all node process (0) + each message ([0, 1])
      - FIFO guarantee
      - async time complexity ≥ sync time complexity
   - Partially synchronous models
      - some time bounds or guarantees
      - more realistic, but most difficult
5. Echo algorithm
6. Echo algorithm revisited
   -  sync, BFS spanning tree -> SyncBFS
      - Time Units = 3 ≤ 2D + 1
      - Messages = 10 = 2|E|
   -  async, not a BFS spanning tree
      - Time Units on Broadcast = 1 ≤ D
      - Messages = 10
      - Time Units on Convergecast = 3 = |V | − 1
      - Total time is 4
```
let parent = null
let rec = 0

for q in Neigh do
    send tok to q
while rec < | Neigh | do
    receive tok
    rec += 1
decide

let parent = null
let rec = 0

receive tok from q // non-deterministic choice!
parent = q
rec += 1
for q in Neigh \ parent do
    send tok to q
while rec < | Neigh | do // count all received tokens
    receive tok // forward and return
    rec += 1
send tok to parent
```
7. Echo/size algorithm
   - The size of the network (number of nodes)
   - The number of nodes which have a given property
   - The maximum, minimum or sum of all values contained in nodes (assuming that each node contains a numerical value)
   - In general, functions which are associative and commutative, such as + (why?)
   - A “leader” (e.g. the node with the highest ID)
   - Non-determnistic but confluent evaluations
```
let parent = null
let rec = 0
let size = 1

for q in Neigh do
   send (tok ,0) to q
while rec < | Neigh | do
   receive (tok , s ) // order irrelevant; + commutative
   rec += 1
   size += s

decide size

let parent = null
let rec = 0
let size = 1

receive (tok , s ) from q // choice irrelevant; + associative
parent = q
rec += 1

for q in Neigh \ parent do
   send (tok ,0) to q

while rec < | Neigh | do
   receive (tok , s ) // fan-out tokens: s=0
   rec += 1 // fan-in tokens: s=subtree size
   size += s

send (tok , size ) to parent // only children really contribute
```
8. Further algorithms
    - Distributed MST (Minimal Spanning Tree): Dijkstra prize 2004
    - Byzantine agreement: ”the crown jewel of distributed algorithms” – Dijkstra prize 2005, 2001
    - The relation between Byzantine algorithms and blockchains
9. Project and practical work
    - Lynch, Nancy (1996). Distributed Algorithms. Morgan Kaufmann Publishers. ISBN 978-1-55860-348-6.
    - Tel, Gerard (2000), Introduction to Distributed Algorithms, Second Edition, Cambridge University Press, ISBN 978-0-52179-483-1.
    - Fokkink, Wan (2013), Distributed Algorithms - An Intuitive Approach, MIT Press (Second Edition 2018 also available)
10. Readings

## Search Fundamentals ##
- Classical DFS
   - 2|E|
   - Cidon DFS: 2|V|-2, 3|E|
- SyncBFS
   - 2D + 1, 2E
   - AsyncBFS: messages = O(D*|E|); time−pileups = O(D); time+pileups = O(D*|V|)
   - LayeredBFS: messages = O(D|V| + |E|); time−pileups = O(D 2 )
   - (P27)
- Bellman-Ford algorithm
   - Time complexity of Dijkstra = O((|E| + |V|)log|V|)
   -  Time complexity * Bellman-Ford = O(|V||E|)
   - Sync Bellman-Ford: O(|V|), O(|V||E|)
   - O(|V|), O(|V|^(|V|)), 
- Sync Echo (aka Sync BFS) : BFS ST, no link changes, fast
- Async Echo : arbitrary ST, no link changes, but not so fast
- Sync BF : shortest paths ST, many link changes, not so fast
   - emulating slow links by extra edges exponentially increases V!
   - the formula is still exponential on k, but not anymore on N = exp(k)!
- Async BF : shortest paths ST, exponential link changes
   - if FIFO: worst case exponential time
- Luby’s algorithm
   - stop with probability 1, expected O(logn) rounds
   - send, greater winner
   - notify neighbors
   - loser boardcase(disconnected)

## Minimum spanning trees ##
- min-height ST: here also BFS ST (cf. sync Echo)
- shortest paths ST (cf. sync/async Bellman-Ford)
- minimum ST (cf. sync/async GHS), here also DFS ST (cf. sync/async Cidon)
- arbitrary ST (cf. async Echo)
- Prim - one node - search for the shortest path, which does not contain repeated node
- Kruskal - 2-way merge - sorted path and find the path does not contain repeated node
- Bor˚uvka - component (subtree) + connected edge - multi-way merges
- distributed MST is based on Bor˚uvka
   - In any multi-way merge there is always one Common MWOE.
   - Level 0 trees, i.e. initial nodes (size 1 trees)
   - Level k trees, for any k ≥ 0
   - Sync
       - O(NlogN), O((N + M)logN)
       - Levels: O(logN)
 - async MST
    - GHS and variants
    - Two neighbours may pe part of the same component tree
    - Not all component trees may have a guaranteed siz
    - component trees may be at different levels

## Logical Clocks and Distributed Snapshots ##

p1 : a s1 r3 b
p2 : c r2 s3
p3 : r1 d s2 e

- Vector Logical Clock
 - V(P2) : (v1 ,v2 ,v3),(r1 ,r2 ,r3)
    - (max(v1 ,r1), max(v2 ,r2) + 1, max(v3 ,r3))


## The Byzantine Agreemen ##















