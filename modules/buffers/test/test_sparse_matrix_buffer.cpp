#include <boost/test/unit_test.hpp>

#include <string>

#include "SparseMatrixBuffer.h"
#include "MatrixBuffer.h"

BOOST_AUTO_TEST_SUITE( SparseMatrixTests )

template<typename T, size_t N>
size_t length_of(T (&a)[N]) {
    return N;
}

template<typename T>
SparseMatrixBufferTemplate<T> CreateExampleSparseMatrix()
{
    // .. .. .. .. .. .. .. ..
    // 11 12 .. 14 .. .. .. ..
    // .. 22 23 .. 25 .. .. ..
    // 31 .. 33 34 .. .. .. ..
    // .. 42 .. .. 45 46 .. ..
    // .. .. .. .. 55 .. .. ..
    // .. .. .. .. 65 66 67 ..
    // .. .. .. .. 75 .. 77 78
    // .. .. .. .. .. .. 87 88

    int rowPtr[] = {0,0,3,6,9,12,13,16,19,21};
    int col[] = {0,1,3,1,2,4,0,2,3,1,4,5,4,4,5,6,4,6,7,6,7};
    T val[] = {11,12,14,22,23,25,31,33,34,42,45,46,55,65,66,67,75,77,78,87,88};

    SparseMatrixBufferTemplate<T>
        smb(&val[0], length_of(val), &col[0], length_of(col),
            &rowPtr[0], length_of(rowPtr), 9, 8);

    return smb;
}

// -----------

BOOST_AUTO_TEST_CASE(test_get_zero)
{
    SparseMatrixBufferTemplate<double> smb(3,3);
    BOOST_CHECK(smb.Get(1,1) == 0.0);
}

BOOST_AUTO_TEST_CASE(test_get_with_empty_row)
{

    // .. .. ..
    // .. 11 ..
    // .. .. ..

    int rowPtr[] = {0, 0, 1, 1};
    int col[] = {1};
    int val[] = {11};

    SparseMatrixBufferTemplate<int>
        smb(&val[0], length_of(val), &col[0], length_of(col),
            &rowPtr[0], length_of(rowPtr), 3, 3);


    bool all_match = true;
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            if (i==1 && j == 1) {
                all_match &= (smb.Get(i,j) == 11);
            }
            else {
                all_match &= (smb.Get(i,j) == 0);
            }
        }
    }

    BOOST_CHECK(all_match);
}

BOOST_AUTO_TEST_CASE(test_ToString)
{

    SparseMatrixBufferTemplate<long> smb = CreateExampleSparseMatrix<long>();

    std::string result = smb.ToString();

    std::string expected =
        ".. .. .. .. .. .. .. .. \n"
        "11 12 .. 14 .. .. .. .. \n"
        ".. 22 23 .. 25 .. .. .. \n"
        "31 .. 33 34 .. .. .. .. \n"
        ".. 42 .. .. 45 46 .. .. \n"
        ".. .. .. .. 55 .. .. .. \n"
        ".. .. .. .. 65 66 67 .. \n"
        ".. .. .. .. 75 .. 77 78 \n"
        ".. .. .. .. .. .. 87 88 \n"
        ;

    BOOST_CHECK(result == expected);

}

BOOST_AUTO_TEST_CASE(test_SumRow)
{
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();

    bool correct = true;
    correct &= smb.SumRow(0) == 0;
    correct &= smb.SumRow(1) == (11 + 12 + 14);
    correct &= smb.SumRow(2) == (22 + 23 + 25);
    correct &= smb.SumRow(3) == (31 + 33 + 34);
    correct &= smb.SumRow(4) == (42 + 45 + 46);
    correct &= smb.SumRow(5) == 55;
    correct &= smb.SumRow(6) == (65 + 66 + 67);
    correct &= smb.SumRow(7) == (75 + 77 + 78);
    correct &= smb.SumRow(8) == (87 + 88);
    BOOST_CHECK(correct);
}

BOOST_AUTO_TEST_CASE(test_NormalizeRow)
{
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();

    bool correct = true;
    for (int i=0; i<smb.GetM(); ++i) {
        smb.NormalizeRow(i);
        double sum = smb.SumRow(i);
        if (sum != 0.0) {
            correct &= abs(sum - 1.0) < 1e-10;
        }
    }
    BOOST_CHECK(correct);
}

BOOST_AUTO_TEST_CASE(test_GetMax_withpositive)
{
    // positive entries
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();
    BOOST_CHECK(smb.GetMax() == 88.0);
}

BOOST_AUTO_TEST_CASE(test_GetMax_nonpositive)
{
    // includes only negative entries and zeros
    // .. .. ..
    // .. -1 -3
    // .. .. ..

    int rowPtr[] = {0, 0, 1, 2};
    int col[] = {1, 2};
    double val[] = {-1, -3};

    SparseMatrixBufferTemplate<double>
        smb(&val[0], length_of(val), &col[0], length_of(col),
            &rowPtr[0], length_of(rowPtr), 3, 3);

    BOOST_CHECK(smb.GetMax() == 0.0);
}

BOOST_AUTO_TEST_CASE(test_GetMin_nonnegative)
{
    // only positive entries
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();
    BOOST_CHECK(smb.GetMin() == 0.0);
}

BOOST_AUTO_TEST_CASE(test_GetMin_withnegative)
{
    // includes a negative entry
    // .. .. ..
    // .. 11 -3
    // .. .. ..

    int rowPtr[] = {0, 0, 1, 2};
    int col[] = {1, 2};
    int val[] = {11, -3};

    SparseMatrixBufferTemplate<int>
        smb(&val[0], length_of(val), &col[0], length_of(col),
            &rowPtr[0], length_of(rowPtr), 3, 3);

     BOOST_CHECK(smb.GetMin() == -3);
}

BOOST_AUTO_TEST_CASE(test_GetM_GetN)
{
    SparseMatrixBufferTemplate<int> smb(4,22);
    BOOST_CHECK(smb.GetM() == 4 && smb.GetN() == 22);
}


BOOST_AUTO_TEST_CASE(test_Append)
{

    SparseMatrixBufferTemplate<long> smb1 = CreateExampleSparseMatrix<long>();
    SparseMatrixBufferTemplate<long> smb2 = CreateExampleSparseMatrix<long>();

    smb1.Append(smb2);

    std::string result = smb1.ToString();

    std::string expected =
        ".. .. .. .. .. .. .. .. \n"
        "11 12 .. 14 .. .. .. .. \n"
        ".. 22 23 .. 25 .. .. .. \n"
        "31 .. 33 34 .. .. .. .. \n"
        ".. 42 .. .. 45 46 .. .. \n"
        ".. .. .. .. 55 .. .. .. \n"
        ".. .. .. .. 65 66 67 .. \n"
        ".. .. .. .. 75 .. 77 78 \n"
        ".. .. .. .. .. .. 87 88 \n"
        ;
    expected = expected + expected;

    BOOST_CHECK(result == expected);

}

BOOST_AUTO_TEST_CASE(test_Append_append_to_empty)
{
    SparseMatrixBufferTemplate<double> smb;

    double data[] = {0,0,0, 1, 0, 1, 0, 0, 1};
    SparseMatrixBufferTemplate<double> expected(&data[0], 1, length_of(data));

    smb.Append(expected);

    BOOST_CHECK(smb == expected);
}

BOOST_AUTO_TEST_CASE(test_SliceRow)
{
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();
    SparseMatrixBufferTemplate<double> row = smb.SliceRow(3);

    double data[] = { 31, 0, 33, 34, 0, 0, 0, 0 };
    SparseMatrixBufferTemplate<double> expectedRow(&data[0], 1, 8);

    BOOST_CHECK(row == expectedRow);
}

BOOST_AUTO_TEST_CASE(test_ConstructFromDense)
{
    double values[] = {
        0, 0, 0,
        1, 0, 3,
        0, 5, 0,
        7, 0, 9,
        0, 0, 0
    };
    MatrixBufferTemplate<double> mb(&values[0], 5, 3);
    SparseMatrixBufferTemplate<double> smb(mb);
    std::string result = smb.ToString();

    std::string expected =
        ".. .. .. \n"
        "1 .. 3 \n"
        ".. 5 .. \n"
        "7 .. 9 \n"
        ".. .. .. \n"
        ;

    BOOST_CHECK(result == expected);
}

BOOST_AUTO_TEST_CASE(test_ColumnIndexOnePastTheEndBug)
{
    double values[] = {
        1, 0,
        0, 1,
    };
    MatrixBufferTemplate<double> mb(&values[0], 2, 2);
    SparseMatrixBufferTemplate<double> result(mb);

    std::string expected =
        "1 .. \n"
        ".. 1 \n"
        ;

    BOOST_CHECK(result.ToString() == expected);
}


BOOST_AUTO_TEST_CASE(test_equals)
{
    double data1[] = {
        1, 2, 3,
        0, 1, 2,
        3, 5, 1
    };
    double data2[] = {
        0, 0, 1,
        0, 2, 0,
        3, 0, 0
    };
    SparseMatrixBufferTemplate<double> smb1(&data1[0], 3, 3);
    SparseMatrixBufferTemplate<double> smb2(&data2[0], 3, 3);

    bool pass = true;
    pass &= (smb1 == smb1);
    pass &= (smb1 != smb2);
    pass &= (smb2 != smb1);
    pass &= (smb2 == smb2);
    BOOST_CHECK(pass);
}

BOOST_AUTO_TEST_CASE(test_Slice)
{
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();

    double indexData[] = { 1, 1, 2, 0, 4, 2, 4, 8 };
    VectorBufferTemplate<int> indices(&indexData[0], length_of(indexData));
    SparseMatrixBufferTemplate<double> sliced = smb.Slice(indices);

    SparseMatrixBufferTemplate<double> expectedSliced;
    for (int i=0; i<indices.GetN(); ++i) {
        expectedSliced.Append(smb.SliceRow(indices.Get(i)));
    }

    BOOST_CHECK(sliced == expectedSliced);
}

BOOST_AUTO_TEST_SUITE_END()
