import itertools
import numpy as np
import pandas as pd

from ..utils import testing
from ..utils.maths import split_contiguous


class TestSplitContigous(testing.TestCase):

    def setUp(self):
        self.chunks = [ [3, 4, 5, 6],       # step: 2
                        [8],                # step: 5
                        [13, 14, 15],
                        [25, 26, 27],       # step: 10
                        [29]                # step: 2
                      ]
        lens = [len(c) for c in self.chunks]

        self.x = list(itertools.chain(*self.chunks))
        self.stops = list(itertools.accumulate(lens))
        self.starts = [0] + self.stops[:-1]
        self.indices = list(zip(self.starts, self.stops))

    def test_basic(self):
        ret = split_contiguous(arr=self.x, tol=1,
                               inclusive=False,
                               indices=True)
        self.assertArrayEqual(ret, self.indices)

        ret = split_contiguous(arr=self.x, tol=1,
                               inclusive=False,
                               indices=False)
        self.assertNestedEqual(ret, self.chunks)

        ret = split_contiguous(arr=self.x, tol=1,
                               inclusive=True,
                               indices=False)
        self.assertArrayEqual(ret, [[x] for x in self.x])


    def test_pandas(self):
        s = pd.Series(self.x, index=100*np.arange(len(self.x)))
        ret = split_contiguous(arr=s, tol=1,
                               inclusive=False,
                               indices=True)
        self.assertArrayEqual(ret, list(zip(self.starts, self.stops)))

        ret = split_contiguous(arr=s, tol=1,
                               inclusive=False,
                               indices=False)
        self.assertNestedEqual(ret, self.chunks)
        self.assertIsInstance(ret[0], pd.Series)


    def test_corner_cases(self):
        # None
        with self.assertRaises(TypeError):
            ret = split_contiguous(arr=None, tol=1)
        # Scalar
        with self.assertRaises(TypeError):
            ret = split_contiguous(arr=123, tol=1)
        # Empty list
        ret = split_contiguous(arr=[], tol=1, indices=True)
        self.assertArrayEqual(ret, [])
        ret = split_contiguous(arr=[], tol=1, indices=False)
        self.assertArrayEqual(ret, [])
        # No splits
        ret = split_contiguous(arr=self.x, tol=100,
                               inclusive=False,
                               indices=False)
        self.assertNestedEqual(ret, [self.x])
        ret = split_contiguous(arr=self.x, tol=100,
                               inclusive=False,
                               indices=True)
        self.assertNestedEqual(ret, [(0, len(self.x))])

