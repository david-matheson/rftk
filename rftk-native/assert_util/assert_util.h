#pragma once

// Turning off exceptions will break some tests
// #define ENABLE_EXCEPTIONS 1 (defined in SConscript)

#if ENABLE_EXCEPTIONS

#include <string>
void AssertArgDimension1d(int n_lhs, int n_rhs,
                            const std::string& file,
                            const int line);

void AssertArgDimension2d(  int m_lhs, int n_lhs,
                            int m_rhs, int n_rhs,
                            const std::string& file,
                            const int line);

void AssertArgDimension3d(  int l_lhs, int m_lhs, int n_lhs,
                            int l_rhs, int m_rhs, int n_rhs,
                            const std::string& file,
                            const int line);

void AssertValidRange( int index, int range_low, int range_high,
                        const std::string& file,
                        const int line);

void Assert(    bool test,
                const std::string& file,
                const int line);

#define ASSERT_ARG_DIM_1D(n_lhs, n_rhs) \
                        AssertArgDimension1d(n_lhs, n_rhs,  __FILE__, __LINE__);

#define ASSERT_ARG_DIM_2D(m_lhs, n_lhs, m_rhs, n_rhs) \
                        AssertArgDimension2d(m_lhs, n_lhs, m_rhs, n_rhs,  __FILE__, __LINE__);

#define ASSERT_ARG_DIM_3D(l_lhs, m_lhs, n_lhs, l_rhs, m_rhs, n_rhs) \
                        AssertArgDimension3d(l_lhs, m_lhs, n_lhs, l_rhs, m_rhs, n_rhs, __FILE__, __LINE__);

#define ASSERT_VALID_RANGE(index, range_low, range_high) \
                        AssertValidRange(index, range_low, range_high, __FILE__, __LINE__);

#define ASSERT(test) Assert(test, __FILE__, __LINE__);

#else

#define ASSERT_ARG_DIM_1D(n_lhs, n_rhs)
#define ASSERT_ARG_DIM_2D(m_lhs, n_lhs, m_rhs, n_rhs)
#define ASSERT_ARG_DIM_3D(l_lhs, m_lhs, n_lhs, l_rhs, m_rhs, n_rhs)
#define ASSERT_VALID_RANGE(index, range_low, range_high)
#define ASSERT(test)

#endif

