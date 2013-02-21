#include <string>

#include "SparseMatrixBuffer.h"

template<typename T, size_t N>
size_t length_of(T (&a)[N]) {
    return N;
}

template<typename T>
SparseMatrixBufferTemplate<T> CreateExampleSparseMatrix()
{
    // Example from here:
    // http://www.cs.indiana.edu/classes/p573/notes/sparse/sparsemat.html
    // Their example uses 1 based indexing.
    //
    // 11 12 .. 14 .. .. .. .. 
    // .. 22 23 .. 25 .. .. .. 
    // 31 .. 33 34 .. .. .. .. 
    // .. 42 .. .. 45 46 .. .. 
    // .. .. .. .. 55 .. .. .. 
    // .. .. .. .. 65 66 67 .. 
    // .. .. .. .. 75 .. 77 78 
    // .. .. .. .. .. .. 87 88 

    size_t rowPtr[] = {0,3,6,9,12,13,16,19,21};
    size_t col[] = {0,1,3,1,2,4,0,2,3,1,4,5,4,4,5,6,4,6,7,6,7};
    T val[] = {11,12,14,22,23,25,31,33,34,42,45,46,55,65,66,67,75,77,78,87,88};

    SparseMatrixBufferTemplate<T>
        smb(&val[0], length_of(val), &col[0], length_of(col),
            &rowPtr[0], length_of(rowPtr), 8, 8);
    
    return smb;
}

// -----------

bool test_get_zero() {
    SparseMatrixBufferTemplate<double> smb(3,3);
    return smb.Get(1,1) == 0.0;
}

bool test_get_with_empty_row() {

    // .. .. ..
    // .. 11 ..
    // .. .. ..

    size_t rowPtr[] = {0, 0, 1, 1};
    size_t col[] = {1};
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
    
    return all_match;
}

bool test_ToString() {

    SparseMatrixBufferTemplate<long> smb = CreateExampleSparseMatrix<long>();

    std::string result = smb.ToString();

    std::string expected = 
        "11 12 .. 14 .. .. .. .. \n"
        ".. 22 23 .. 25 .. .. .. \n"
        "31 .. 33 34 .. .. .. .. \n"
        ".. 42 .. .. 45 46 .. .. \n"
        ".. .. .. .. 55 .. .. .. \n"
        ".. .. .. .. 65 66 67 .. \n"
        ".. .. .. .. 75 .. 77 78 \n"
        ".. .. .. .. .. .. 87 88 \n"
        ;

    return result == expected;

}

bool test_SumRow() {
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();

    bool correct = true;
    correct &= smb.SumRow(0) == (11 + 12 + 14);
    correct &= smb.SumRow(1) == (22 + 23 + 25);
    correct &= smb.SumRow(2) == (31 + 33 + 34);
    correct &= smb.SumRow(3) == (42 + 45 + 46);
    correct &= smb.SumRow(4) == 55;
    correct &= smb.SumRow(5) == (65 + 66 + 67);
    correct &= smb.SumRow(6) == (75 + 77 + 78);
    correct &= smb.SumRow(7) == (87 + 88);
    return correct;
}

bool test_NormalizeRow() {
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();

    bool correct = true;
    for (int i=0; i<smb.GetM(); ++i) {
        smb.NormalizeRow(i);
        correct &= abs(smb.SumRow(i) - 1.0) < 1e-10;
    }
    return correct;
}

bool test_GetMax_withpositive() {
    // positive entries
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();
    return smb.GetMax() == 88.0;
}

bool test_GetMax_nonpositive() {
    // includes only negative entries and zeros
    // .. .. ..
    // .. -1 -3
    // .. .. ..

    size_t rowPtr[] = {0, 0, 1, 2};
    size_t col[] = {1, 2};
    double val[] = {-1, -3};

    SparseMatrixBufferTemplate<double>
        smb(&val[0], length_of(val), &col[0], length_of(col),
            &rowPtr[0], length_of(rowPtr), 3, 3);

    return smb.GetMax() == 0.0;
}

bool test_GetMin_nonnegative() {
    // only positive entries
    SparseMatrixBufferTemplate<double> smb = CreateExampleSparseMatrix<double>();
    return smb.GetMin() == 0.0;
}

bool test_GetMin_withnegative() {
    // includes a negative entry
    // .. .. ..
    // .. 11 -3
    // .. .. ..

    size_t rowPtr[] = {0, 0, 1, 2};
    size_t col[] = {1, 2};
    int val[] = {11, -3};

    SparseMatrixBufferTemplate<int>
        smb(&val[0], length_of(val), &col[0], length_of(col),
            &rowPtr[0], length_of(rowPtr), 3, 3);

    return smb.GetMin() == -3;
}

bool test_GetM_GetN() {
    SparseMatrixBufferTemplate<int> smb(4,22);
    return smb.GetM() == 4 && smb.GetN() == 22;
}

bool test_Resize() {
    SparseMatrixBufferTemplate<float> smb(3,3);
    smb.Resize(10, 12);
    return smb.GetM() == 10 && smb.GetN() == 12;
}

// ------------


bool report_test(std::string const& test_name, bool result) {
    if (result) {
        std::cout << "SUCCESS: " << test_name << std::endl;
        return true;
    } else {
        std::cerr << "FAILURE: " << test_name << std::endl;
        return false;
    }
}
#define RUN_TEST(test_name) all_success &= report_test(#test_name, test_name());

int main() {
    bool all_success = true;

    RUN_TEST(test_get_zero);
    RUN_TEST(test_get_with_empty_row);
    RUN_TEST(test_ToString);
    RUN_TEST(test_SumRow);
    RUN_TEST(test_NormalizeRow);
    RUN_TEST(test_GetMax_withpositive);
    RUN_TEST(test_GetMax_nonpositive);
    RUN_TEST(test_GetMin_nonnegative);
    RUN_TEST(test_GetMin_withnegative);
    RUN_TEST(test_GetM_GetN);
    RUN_TEST(test_Resize);

    return !all_success;
}
