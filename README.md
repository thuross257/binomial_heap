# binomial_heap

A small pure-Python binomial heap: a min-priority queue whose signature feature is fast `union` of two heaps in O(log n) time.

```python
from binomial_heap import BinomialHeap

a = BinomialHeap([4, 1, 7])
b = BinomialHeap([3, 9, 2])
a.union(b)
print(a.pop())   # 1
print(a.pop())   # 2

records = BinomialHeap([("a", 3), ("b", 1)], key=lambda r: r[1])
print(records.pop())   # ('b', 1)
```

## Why this exists

A plain binary heap melds two heaps in O(n) because it must splice and re-heapify an array. A binomial heap keeps its elements in a forest of binomial trees (one per bit set in the size), so a meld is binary addition over the forest — link equal-order trees and carry — giving O(log n) worst-case `union`. The trade-off: constant factors are higher than an array heap for plain push/pop, and the structure is pointer-based, so it is heavier per element. You'd reach for this when merging priority queues is a hot operation, not when you only ever push and pop one queue.

## Edge cases worth knowing

- **Empty-heap operations.** `peek` and `pop` raise `IndexError`; `len` and `bool` behave normally. `union` accepts an empty heap on either side.
- **Duplicate keys.** Handled; the heap property is `<=`, so ties are stable only by accident of insertion order — we do not guarantee stability among equal keys.
- **`key` functions.** When you construct with `key`, the heap orders by `key(item)` but stores and returns the original items. The key is computed once per `push` and cached on the node, so `pop`/`union` never re-call your function. Two heaps may be `union`ed only if they were constructed with the **same** `key` callable (compared by identity); mismatched regimes raise `ValueError` rather than silently comparing apples to oranges.
- **Iteration.** `for x in heap` yields items in arbitrary forest order, not sorted order, and does not consume the heap. If you need sorted output, `pop` until empty.

## Exported names

- `BinomialHeap(items=None, *, key=None)` — constructor; `items` is any iterable.
  - `.push(item)`
  - `.peek()` — returns smallest item without removing; raises `IndexError` if empty.
  - `.pop()` — removes and returns smallest; raises `IndexError` if empty.
  - `.union(other)` — melds `other` into `self` and empties `other`; raises `ValueError` if the `key` functions differ, `TypeError` if `other` is not a `BinomialHeap`.
  - `.__len__`, `.__bool__`, `.__iter__`
- `BinomialTree` — the underlying node type, exposed for inspection/testing.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
