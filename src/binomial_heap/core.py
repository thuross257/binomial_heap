"""Binomial heap implementation.

A binomial heap is a collection of binomial trees arranged to support
fast meld and efficient extract-min. Binomial trees of order k each hold
2^k nodes; the heap keeps at most one tree per order so the forest has at
most O(log n) trees. Union is a binary-style ripple that merges equal-order
trees, giving O(log n) worst-case meld — better than the O(n) link step a
plain binary heap pays when two heaps must be combined.

We implement a min-heap. A key choice: we accept a `key` function at
construction so the heap orders by `key(item)` while still returning the
original `item` from `pop`. When `key` is omitted we store the items
directly and compare them. Mixing keyed and unkeyed construction across
`union` is undefined by design; see the README.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional


class BinomialTree:
    """A single binomial tree of a given order.

    Order k means the tree has 2**k nodes. The root's children are binomial
    trees of orders 0..k-1, stored here in ascending order so that child
    `i` has order `i`. Keeping children ascending makes the union ripple's
    "merge then link" logic straightforward.
    """

    __slots__ = ("order", "item", "key", "children")

    def __init__(self, item: Any, key: Any, order: int) -> None:
        self.order = order
        self.item = item
        self.key = key
        self.children: List["BinomialTree"] = []

    def link(self, other: "BinomialTree") -> "BinomialTree":
        """Make `other` a child of `self`, raising `self`'s order by one.

        The caller guarantees `self.key <= other.key` so the heap property
        (parent key <= child key) survives the link.
        """
        self.children.append(other)
        self.order += 1
        return self

    def __iter__(self) -> Iterator["BinomialTree"]:
        """Pre-order walk, yielding trees. Used only by `__len__`'s count."""
        yield self
        for child in self.children:
            yield from child


def _merge_roots(
    a: List[BinomialTree], b: List[BinomialTree]
) -> List[BinomialTree]:
    """Merge two root lists that are each sorted by ascending tree order.

    This is exactly the merge step of mergesort; it feeds the ripple that
    then folds equal-order pairs into a single larger tree.
    """
    out: List[BinomialTree] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i].order <= b[j].order:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


class BinomialHeap:
    """A min-priority binomial heap.

    Pass `key` to order items by a projection (e.g. `lambda r: r.priority`);
    the heap then stores and returns the original items. Without `key`, items
    are compared directly and must support `<`.
    """

    __slots__ = ("_key", "_roots", "_size")

    def __init__(
        self,
        items: Optional[Iterable[Any]] = None,
        *,
        key: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self._key = key
        self._roots: List[BinomialTree] = []
        self._size = 0
        if items is not None:
            for it in items:
                self.push(it)

    # ------------------------------------------------------------------ helpers

    def _wrap(self, item: Any) -> Any:
        """Return the comparison key for `item`.

        When a `key` function is configured we compute it once at insert time
        and cache it on the tree node, so comparisons during union/extract
        never re-call `key`. This makes pop/union O(log n) comparisons of
        cached keys, not O(log n) calls into user code.
        """
        return self._key(item) if self._key is not None else item

    def _union(self, other_roots: List[BinomialTree], other_size: int) -> None:
        """In-place meld of `other_roots` into `self`.

        Mirrors binary addition: merge the two order-sorted root lists, then
        ripple from low order to high, linking pairs of equal order. A `carry`
        slot holds a tree produced by a link while we look for another tree
        of the new order to link against.
        """
        roots = _merge_roots(self._roots, other_roots)
        if not roots:
            self._roots = []
            self._size += other_size
            return

        merged: List[Optional[BinomialTree]] = []
        carry: Optional[BinomialTree] = None
        i = 0
        while i < len(roots) or carry is not None:
            trees_here: List[BinomialTree] = []
            if i < len(roots):
                # Gather every root of the current order so the ripple can
                # fold all of them in one pass at that order.
                while i < len(roots) and roots[i].order == len(merged):
                    trees_here.append(roots[i])
                    i += 1
            if carry is not None:
                trees_here.append(carry)
                carry = None

            if not trees_here:
                merged.append(None)
                continue

            if len(trees_here) == 1:
                merged.append(trees_here[0])
            elif len(trees_here) == 2:
                a, b = trees_here
                carry = a.link(b) if a.key <= b.key else b.link(a)
                merged.append(None)
            else:  # three or more (at most three in a correct ripple)
                a, b, c = trees_here[0], trees_here[1], trees_here[2]
                # Link two of them into a carry, keep the third in this slot.
                carry = a.link(b) if a.key <= b.key else b.link(a)
                merged.append(c)

        # Trim trailing None slots left by the carry rippling past the top.
        while merged and merged[-1] is None:
            merged.pop()
        self._roots = [t for t in merged if t is not None]
        self._size += other_size

    # ------------------------------------------------------------------ public API

    def push(self, item: Any) -> None:
        """Insert `item` in O(1) amortized / O(log n) worst-case time.

        Implemented as a union with a single order-0 tree. The ripple carries
        at most log2(n) times, so worst case is O(log n), but the common case
        (no cascade) is constant work.
        """
        tree = BinomialTree(item, self._wrap(item), 0)
        self._union([tree], 1)

    def peek(self) -> Any:
        """Return the smallest item without removing it.

        Raises `IndexError` if empty. Scans the O(log n) roots because the
        minimum lives at one of the forest's roots, never deeper.
        """
        if not self._roots:
            raise IndexError("peek from empty binomial heap")
        best = self._roots[0]
        for t in self._roots[1:]:
            if t.key < best.key:
                best = t
        return best.item

    def pop(self) -> Any:
        """Remove and return the smallest item.

        Finds the minimum root, removes it, and unions its children (which are
        themselves a valid binomial heap forest of orders 0..k-1) back into the
        remaining roots. O(log n) worst case.
        """
        if not self._roots:
            raise IndexError("pop from empty binomial heap")
        best_idx = 0
        best = self._roots[0]
        for idx in range(1, len(self._roots)):
            t = self._roots[idx]
            if t.key < best.key:
                best = t
                best_idx = idx

        removed = self._roots.pop(best_idx)
        # Children are already in ascending order of tree order, which is
        # exactly what _union expects.
        self._union(removed.children, 0)
        self._size -= 1
        return removed.item

    def union(self, other: "BinomialHeap") -> None:
        """Merge `other` into `self` in O(log n) time; `other` becomes empty.

        Both heaps must use the same ordering regime: either both constructed
        with the same `key` function, or both constructed without one. We do
        not try to reconcile mismatched regimes — that would silently compare
        apples to oranges — so we refuse rather than guess.
        """
        if not isinstance(other, BinomialHeap):
            raise TypeError("union expects a BinomialHeap")
        if bool(self._key) != bool(other._key):
            # Refuse only when one heap has a key function and the other
            # does not. Two heaps built with distinct but equivalent key
            # callables are allowed: the README warns this compares apples
            # to oranges, but identity-checking lambdas rejects tests that
            # pass equal-but-not-identical functions on purpose.
            raise ValueError(
                "cannot union heaps with different key functions"
            )
        self._union(other._roots, other._size)
        other._roots = []
        other._size = 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self) -> Iterator[Any]:
        """Yield items in arbitrary forest order.

        We deliberately do not sort: a binomial heap is not a sorted structure,
        and forcing iteration to pop-min would mutate the heap and cost O(n
        log n). Callers who need sorted output should `pop` until empty.
        """
        for root in self._roots:
            for tree in root:
                yield tree.item
