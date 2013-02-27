/* -*- C -*- */

%apply (double* IN_ARRAY1, int DIM1) {
  (double* values, int nV),
    (int* col, int nC),
    (int* rowPtr, int nRowPtr)
 }
