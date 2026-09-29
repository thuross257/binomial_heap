import unittest

from binomial_heap import BinomialHeap, BinomialTree


class TestPushPopOrdering(unittest.TestCase):
    def test_single_element(self):
        h = BinomialHeap()
        h.push(42)
        self.assertEqual(h.peek(), 42)
        self.assertEqual(h.pop(), 42)
        self.assertEqual(len(h), 0)

    def test_pop_returns_smallest(self):
        h = BinomialHeap([5, 3, 8, 1, 9, 2, 7])
        out = [h.pop() for _ in range(len(h))]
        self.assertEqual(out, [1, 2, 3, 5, 7, 8, 9])

    def test_ascending_insertion_still_sorts(self):
        # Forces the union ripple to carry repeatedly; a common bug spot.
        h = BinomialHeap(list(range(32)))
        out = [h.pop() for _ in range(len(h))]
        self.assertEqual(out, list(range(32)))

    def test_descending_insertion_still_sorts(self):
        h = BinomialHeap(list(range(31, -1, -1)))
        out = [h.pop() for _ in range(len(h))]
        self.assertEqual(out, list(range(32)))

    def test_duplicates(self):
        h = BinomialHeap([4, 4, 4, 1, 1, 9])
        out = [h.pop() for _ in range(len(h))]
        self.assertEqual(out, [1, 1, 4, 4, 4, 9])

    def test_negative_and_zero(self):
        h = BinomialHeap([0, -5, 3, -1, -5, 2])
        out = [h.pop() for _ in range(len(h))]
        self.assertEqual(out, [-5, -5, -1, 0, 2, 3])


class TestPeekAndEmpty(unittest.TestCase):
    def test_peek_does_not_remove(self):
        h = BinomialHeap([3, 1, 2])
        self.assertEqual(h.peek(), 1)
        self.assertEqual(len(h), 3)
        self.assertEqual(h.peek(), 1)

    def test_peek_empty_raises(self):
        h = BinomialHeap()
        with self.assertRaises(IndexError):
            h.peek()

    def test_pop_empty_raises(self):
        h = BinomialHeap()
        with self.assertRaises(IndexError):
            h.pop()

    def test_bool(self):
        h = BinomialHeap()
        self.assertFalse(h)
        h.push(1)
        self.assertTrue(h)
        h.pop()
        self.assertFalse(h)


class TestKeyFunction(unittest.TestCase):
    def test_key_orders_by_projection_returns_original(self):
        records = [("a", 3), ("b", 1), ("c", 2)]
        h = BinomialHeap(records, key=lambda r: r[1])
        first = h.pop()
        self.assertEqual(first, ("b", 1))
        self.assertEqual(first[1], 1)
        second = h.pop()
        self.assertEqual(second, ("c", 2))

    def test_key_with_ties_keeps_original_items(self):
        h = BinomialHeap([("x", 1), ("y", 1), ("z", 1)], key=lambda r: r[1])
        out = [h.pop() for _ in range(len(h))]
        self.assertEqual(set(out), {("x", 1), ("y", 1), ("z", 1)})
        self.assertEqual(len(out), 3)

    def test_key_called_once_per_push_not_per_comparison(self):
        calls = []

        def k(v):
            calls.append(v)
            return v

        h = BinomialHeap([5, 3, 8, 1], key=k)
        self.assertEqual(len(calls), 4)  # one per push, not per comparison
        before = len(calls)
        h.pop()
        h.pop()
        self.assertEqual(len(calls), before)  # no extra key calls on pop


class TestUnion(unittest.TestCase):
    def test_union_then_pop_all(self):
        a = BinomialHeap([4, 1, 7])
        b = BinomialHeap([3, 9, 2, 6])
        a.union(b)
        self.assertEqual(len(a), 7)
        self.assertEqual(len(b), 0)
        out = [a.pop() for _ in range(len(a))]
        self.assertEqual(out, [1, 2, 3, 4, 6, 7, 9])

    def test_union_into_empty(self):
        a = BinomialHeap()
        b = BinomialHeap([2, 1, 3])
        a.union(b)
        self.assertEqual([a.pop() for _ in range(len(a))], [1, 2, 3])
        self.assertEqual(len(b), 0)

    def test_union_empty_into_nonempty(self):
        a = BinomialHeap([2, 1, 3])
        b = BinomialHeap()
        a.union(b)
        self.assertEqual([a.pop() for _ in range(len(a))], [1, 2, 3])

    def test_union_drains_other(self):
        a = BinomialHeap([5, 1])
        b = BinomialHeap([3, 9])
        a.union(b)
        with self.assertRaises(IndexError):
            b.pop()
        self.assertFalse(b)

    def test_union_three_heaps(self):
        a = BinomialHeap([10, 2, 8])
        b = BinomialHeap([5, 1, 7])
        c = BinomialHeap([4, 9, 3, 6])
        a.union(b)
        a.union(c)
        out = [a.pop() for _ in range(len(a))]
        self.assertEqual(out, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_union_mismatched_key_raises(self):
        a = BinomialHeap([1, 2], key=lambda x: x)
        b = BinomialHeap([3, 4])
        with self.assertRaises(ValueError):
            a.union(b)

    def test_union_matching_key_succeeds(self):
        a = BinomialHeap([2, 5], key=lambda x: x)
        b = BinomialHeap([1, 3], key=lambda x: x)
        a.union(b)
        self.assertEqual([a.pop() for _ in range(len(a))], [1, 2, 3, 5])

    def test_union_non_heap_raises(self):
        a = BinomialHeap([1])
        with self.assertRaises(TypeError):
            a.union([2, 3])  # type: ignore[arg-type]


class TestIteration(unittest.TestCase):
    def test_iter_yields_all_items(self):
        h = BinomialHeap([4, 1, 3, 2])
        self.assertEqual(sorted(h), [1, 2, 3, 4])
        self.assertEqual(len(h), 4)

    def test_iter_does_not_mutate(self):
        h = BinomialHeap([4, 1, 3, 2])
        list(h)
        self.assertEqual(len(h), 4)
        self.assertEqual(h.peek(), 1)

    def test_iter_empty(self):
        self.assertEqual(list(BinomialHeap()), [])


class TestConstruction(unittest.TestCase):
    def test_construct_from_iterable(self):
        h = BinomialHeap([9, 4, 7, 1, 6])
        self.assertEqual(len(h), 5)
        self.assertEqual(h.pop(), 1)

    def test_construct_from_empty_iterable(self):
        h = BinomialHeap([])
        self.assertEqual(len(h), 0)
        self.assertFalse(h)

    def test_construct_from_generator(self):
        h = BinomialHeap((x for x in [3, 1, 2]))
        self.assertEqual([h.pop() for _ in range(len(h))], [1, 2, 3])

    def test_default_items_compared_directly(self):
        # No key function: items compared with <, returned as-is.
        h = BinomialHeap(["banana", "apple", "cherry"])
        self.assertEqual(h.pop(), "apple")


class TestInternalStructure(unittest.TestCase):
    def test_tree_link_increases_order(self):
        a = BinomialTree("x", 1, 0)
        b = BinomialTree("y", 2, 0)
        linked = a.link(b)
        self.assertIs(linked, a)
        self.assertEqual(a.order, 1)
        self.assertEqual(len(a.children), 1)
        self.assertIs(a.children[0], b)

    def test_tree_iter_pre_order(self):
        root = BinomialTree("r", 0, 0)
        c1 = BinomialTree("c1", 1, 0)
        c2 = BinomialTree("c2", 2, 0)
        root.link(c1)
        root.link(c2)
        items = [t.item for t in root]
        self.assertEqual(items, ["r", "c1", "c2"])

    def test_forest_has_at_most_one_tree_per_order(self):
        # After any sequence of pushes, no two roots share an order.
        h = BinomialHeap()
        for i in range(64):
            h.push(i)
        orders = [t.order for t in h._roots]
        self.assertEqual(len(orders), len(set(orders)))
        self.assertEqual(sum(2 ** o for o in orders), 64)


class TestStress(unittest.TestCase):
    def test_random_insert_pop_matches_sorted(self):
        import random

        rng = random.Random(1234)
        data = [rng.randint(-1000, 1000) for _ in range(500)]
        h = BinomialHeap(data)
        self.assertEqual(len(h), 500)
        out = [h.pop() for _ in range(len(h))]
        self.assertEqual(out, sorted(data))

    def test_interleaved_push_pop(self):
        import random

        rng = random.Random(99)
        h = BinomialHeap()
        expected = []
        for _ in range(300):
            v = rng.randint(0, 1000)
            h.push(v)
            expected.append(v)
            if rng.random() < 0.5 and h:
                expected.sort()
                self.assertEqual(h.pop(), expected.pop(0))
        expected.sort()
        rest = [h.pop() for _ in range(len(h))]
        self.assertEqual(rest, expected)


if __name__ == "__main__":
    unittest.main()
