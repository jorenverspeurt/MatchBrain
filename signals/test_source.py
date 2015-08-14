from signals.primitive import *
import unittest
from unittest import TestCase
from hypothesis import given
import hypothesis.strategies as st
from itertools import compress

__author__ = 'joren'


class TestSource(TestCase):
    @given(st.integers())
    def test_what_goes_in_comes_out(self, some_input):
        s = Source(lambda: some_input)
        self.assertIsNone(s.pull())
        s.push(self)
        self.assertEqual(s.pull(), some_input)

    @given(st.integers())
    def test_one_subscriber_gets_stuff(self, some_input):
        so = Source(lambda: some_input)
        si = Sink([so],lambda x: self.assertEqual(some_input, x))
        so.push(self)

    @given(st.integers())
    def test_all_subscribers_get_stuff(self, some_input):
        so = Source(lambda: some_input)
        class Box:
            def __init__(self):
                self.value = None
            def set(self, value):
                self.value = value
            def get(self):
                return self.value
        box1, box2, box3 = Box(), Box(), Box()
        si1 = Sink([so],lambda x: box1.set(x))
        si2 = Sink([so],lambda x: box2.set(x))
        si3 = Sink([so],lambda x: box3.set(x))
        so.push(self)
        map(lambda x: self.assertEqual(some_input, x.get()), [box1, box2, box3])

    @given(st.lists(st.tuples(st.integers(),st.booleans())))
    def test_only_pull_when_necessary(self, some_list):
        the_ints = (li[0] for li in some_list)
        the_bools = (li[1] for li  in some_list)
        so = Source(the_ints.next, the_bools.next)
        fil = list(compress(the_ints, the_bools))
        class ListM:
            def __init__(self):

        self.assertEqual(results, fil)

if __name__ == '__main__':
    unittest.main()
