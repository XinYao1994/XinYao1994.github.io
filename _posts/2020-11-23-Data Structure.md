## Data Structures ##
• The Array.
• The List.
• The Stack.
• The Queue.
• The Record.

### heap ###
- (2A)A heap is an essentially complete binary tree with an additional property
- heapsort basic: 
- create a heap by siftdown -> [n ÷ 2 to 1]
- the first one is always the largest one -> [n -> 2]

- Each time we remove a customer from the heap we
   - Move the last entry to the top of the heap
   - Reduce the heap size by one
   - Sift down the top entry
- Each time we add a customer to the heap we:
   - Increase the heap size by one
   - Add the customer to the end of the heap
   - Sift up the last entry
## AVL trees

In the worst case, when using a Binary Search Tree, the data is adding in ascending order, giving the following tree:

`A`  
`.\`  
`..B`  
`...\`  
`....C`  


That is, the depth of the tree is the same as the number of elements.  What we ideally want is a balanced tree, where the depth of the tree is *log(N)* in the number of elements, N.

AVL trees rebalance the tree every time a node is inserted.  This is done by computing the *balance factor* of that node.  It is computed as:

`balanceFactor(node) = maxDepth(left node) - maxDepth(right node)`

If this balance factor is ever 2, or -2, the tree beneath that node needs to be rebalanced: it is much deeper on one side than the other.  For the following tree:


`A`  
`.\`  
`..B`  
`...\`  
`....C`  


...the balance factors are:


`-2`  
`.\`  
`..-1`  
`...\`  
`....0`  


...because looking at the root, the depth of the right subtree is 2; but the depth of the left subtree is 0.

To become rebalanced, an AVL tree performs one of four operations.  Perform these where needed in your implementation of `insert` in the BinarySearchTree class.  After inserting a node you will need to look at its parent, and its parent's parent; compute the balance factor of its parent's parent; and if the tree is then unbalanced, perform the appropriate operation.

(A full AVL tree implementation does a bit more than this, but implementing the cases described here is sufficient for this assignment.)

### Left rotation

If a node becomes unbalanced, when a node is inserted into the right subtree of its right child, then we perform a left rotation.  This is best shown with an example.  Suppose we had the subtree:

`A`  
`.\`  
`..B`  


...and added 'C', we would get:

`A`  
`.\`  
`..B`  
`...\`  
`....C`  

*C* is what we have just inserted; *B* is its parent; *A* is its parent's parent.

*A* now has a balance factor of -2, so we left rotate: we reorder the nodes so that B is the root of this subtree instead of A:

`..B`  
`./.\`  
`A...C`  


Each of these now has a balance factor of 0, so it is balanced again.

Note if A had a parent, B is attached to this, replacing A.

### Right rotation

If a node becomes unbalanced when a node is inserted into the left subtree of its left child, then we perform a right rotation.  Suppose we had the tree:

`....C`  
`.../ `  
`..B`  

...and added 'A', we would get:

`....C`  
`.../ `  
`..B`  
`./`  
`A`  

C is now unbalanced: its balance factor is 2, because its left child has depth 2, but its right child is empty (depth 0).  Thus, we right rotate: we reorder the nodes so that B is the root of this subtree instead of C:

`..B`  
`./.\`  
`A...C`  

Note if C had a parent, B is attached to this, replacing C.

### Left-Right rotation

If a node becomes unbalanced when a node is inserted into the right subtree of its left child, then we perform a left-right rotation.  If we had the tree:

`....C`  
`.../ `  
`..A`  

...and added B, we would get:

`....C`  
`.../ `  
`..A`  
`..\`  
`...B`  

C is now unbalanced.  This scenario is fixed by performing a left--right rotation.  First, we perform a left rotation on the subtree rooted at A, making B the root of this subtree:

`....C`  
`.../ `  
`..B`  
`./`  
`A`  

Then, we perform a right rotation on C, making B the root of this subtree:

`..B`  
`./.\`  
`A...C`  

Note if C had a parent, B is attached to this, replacing C.

### Right-left rotation

One scenario left: a node becomes unbalanced when a node is inserted into the left subtree of its right child, then we perform a right-left rotation.  If we had the tree:

`....A`  
`.....\ `  
`......C`  

... and added B, we would get:

`....A`  
`.....\ `  
`......C`  
`...../`  
`....B`  

A is now unbalanced.  A right-left rotation fixes this in two stages.  First, we perform a right rotation on the subtree rooted at C:

`....A`  
`.....\ `  
`......B`  
`.......\`  
`........C`  

Then, we perform a left rotation on A, making B the root of this subtree:

`..B`  
`./.\`  
`A...C`  

Note if A had a parent, B is attached to this, replacing A.


## 2‐4 Trees ##
- B-tree and 2-4 tree

## dir ##
- The Direct Access Table
   -  Hashing With Chaining
-  (6A)Table doubling minimizes the cost associated with dynamic data structures.
- Open Addressing

- Open Adressing:
   -  Uses less memory—no need for pointers;
   -  Is faster—provided  is kept below 0.5;
   -  Is a little harder to implement and understand.
   -  Is clean—one data structure, the array.
- Chaining:
   -  Uses more memory;
   -  Is faster—if we are not careful with open addressing.
   -  Is a little easier to implement and understand.
   -  Is a bit messy—arrays of linked lists.

- Linear Time Search 
   - We compare the hash of string s with the hash of each substring of t with the same length
- 257^3 x104+257^2 x97+257^1 x114+257^0 x114

- Big Numbers
 - add: write them in base x, and add each pair
 - a^p mod p = a
- (7A) for RAS


## Graph
- Adjacency List
- Adjacency Matrix
BFS and DFS (8A)
Articulation Points (8B)
- Dijkstra;
- Bellman Ford.












