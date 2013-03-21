import unittest as unittest
import numpy as np
import scipy.sparse as sparse
import rftk.asserts
import rftk.buffers as buffers


# Generic type-free testing code.
#
# Use self.create_matrix to construct a SparseMatrixBuffer from a scipy.sparse matrix.
# Remember to specify (..., dtype=self.dtype) when creating the scipy.sparse matrix.
#
# Concrete realizations of this class are mixed with a unittest.TestCase so you
# can use all of it's fanciness here.

class TestSparseBuffers(object):
    def sparse_matrix_compare_expected(self, expected, result):
        self.assertEqual(expected.shape[0], result.GetM())
        self.assertEqual(expected.shape[1], result.GetN())

        for i in xrange(expected.shape[0]):
            for j in xrange(expected.shape[1]):
                self.assertEqual(expected[i,j], result.Get(i,j))

    def test_create_dense(self):
        with self.assertRaises(TypeError):
            self.sparse_matrix(np.zeros(5), dtype=self.dtype)

    def test_create_from_csr(self):
        expected = sparse.csr_matrix(10*np.eye(5,5), dtype=self.dtype)
        result = self.sparse_matrix(expected)
        self.sparse_matrix_compare_expected(expected, result)

    def test_create_from_non_csr(self):
        expected = sparse.lil_matrix(10*np.eye(5,5), dtype=self.dtype)
        result = self.sparse_matrix(expected)
        self.sparse_matrix_compare_expected(expected, result)

    def test_zero(self):
        result = self.sparse_matrix(sparse.csr_matrix(np.eye(5, 5), dtype=self.dtype))
        result.Zero()
        expected = np.zeros((5,5))
        self.sparse_matrix_compare_expected(expected, result)



# Concrete test case classes which realize the TestSparseBuffers class for
# different data types.

class TestFloat64SparseBuffers(TestSparseBuffers, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.sparse_matrix = buffers.Float64SparseMatrix
        self.dtype = np.float64

class TestFloat32SparseBuffers(TestSparseBuffers, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.sparse_matrix = buffers.Float32SparseMatrix
        self.dtype = np.float32

class TestInt32SparseBuffers(TestSparseBuffers, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.sparse_matrix = buffers.Int32SparseMatrix
        self.dtype = np.int32

class TestInt64SparseBuffers(TestSparseBuffers, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.sparse_matrix = buffers.Int64SparseMatrix
        self.dtype = np.int64


if __name__ == '__main__':
    unittest.main()
