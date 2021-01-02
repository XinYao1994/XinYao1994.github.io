
## Scope and Scoping
- Run-time:
   - refers to the time when an application actually executes.
- Compile-time:
   - everything before run-time, that is, compilation, linking, and loading.
- For a program, its static structure is the structure of the source program, how it is organized.
- The dynamic structure is the structure that evolves during run-time.

- Static area: compile time
- Stack area: last-in, first-out, function call
   - wasteful
   - implement recursion statically
   - dynamic data structures
- Heap: Allocation & deallocation
   - External fragmentation
   - Internal fragmentation
- Scope and Referencing

## Haskell

```
main = do putStrLn "What is 4 * 5?"
          x <- readLn
          if x == 20
             then putStrLn "You're right!"
             else putStrLn "You're wrong!"


x = 2 -- Two hyphens introduce a comment
y = 3 -- ...that continues to end of line.
main = let z = x + y  -- let introduces local bindings
       in print z
```
- curried functions

```
safeDiv x y = let q = div x y -- safe as q never evaluated if y == 0
              in if y == 0 then 0 else q
main = print (safeDiv 1 0)
```
```
a:[]
[]++[]
[]!!0
head []
tail []
length []
elem  a [] -- existing or not
maximum []
minimum []
sum []
product []
```
```
signum x | x < 0 = -1
         | x == 0 = 0
         | x > 0 = 1
```
```
signum x | x < 0 = -1
         | x == 0 = 0
         | otherwise = 1
```
```
(x:ys) = mylist
_:xs
y:_
((_,y),_) = nestedTupl
```
```
toUpper c
isDigit c
maxBound
```
```
reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = let snoc x xs = xs ++ [x]
                  in x `snoc` (reverse' xs)

reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = x `snoc` (reverse' xs)
                  where snoc x xs = xs ++ [x]
```


- A virtual function uses dynamic dispatch
- A non-virtual function uses static dispatch.

```
sumOfSquareRoots xs = sum $ map sqrt $ filter (\x -> x>0) xs
sumOfSquareRoots = sum $ map sqrt $ filter (\x -> x>0)
```

```
type Point = (Float, Float)
type Line = (Point, Point)
-- polymorphic type
type Node a = (a,a)
type Edge a = (Node a, Node a)


data Days = Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday deriving (Show,Eq)

isWeekday :: Days -> Bool
isWeekday day = not (day `elem` [Saturday, Sunday])
```
```
amount :: Fractional p => Money -> p
amount (NONE) = fromIntegral(0)
amount (COIN x) = fromIntegral(x)/100.0
amount (BILL x) = fromIntegral(x)
```
- Data type acts like enum in C.
- Data type with parameterized constructors acts like Union in C.

```
data Maybe a = Just a | Nothing deriving (Show, Eq, Ord)
head' :: [a] -> Maybe a
head' [] = Nothing
head' (x:xs) = (Just x)
```

