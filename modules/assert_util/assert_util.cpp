#include <stdexcept>
#include <string>
#include <sstream>

#include "assert_util.h"

#if ENABLE_EXCEPTIONS

void AssertArgDimension1d(int n_lhs, int n_rhs,
                            const std::string& file,
                            const int line)
{
    if(n_lhs != n_rhs)
    {
        std::stringstream ss;
        ss << "Dimensions not equal "
                << n_lhs << " != " << n_rhs
                << " - file:" << file << ", line:" << line;
        throw std::length_error(ss.str());
    }
}

void AssertArgDimension2d(  int m_lhs, int n_lhs,
                            int m_rhs, int n_rhs,
                            const std::string& file,
                            const int line)
{
    if(m_lhs != m_rhs || n_lhs != n_rhs)
    {
        std::stringstream ss;
        ss << "Dimensions not equal "
                << "(" << m_lhs << "," << n_lhs << ") != ("
                << m_rhs << "," << n_rhs << ")"
                << " - file:" << file << ", line:" << line;
        throw std::length_error(ss.str());
    }
}

void AssertArgDimension3d(  int l_lhs, int m_lhs, int n_lhs,
                            int l_rhs, int m_rhs, int n_rhs,
                            const std::string& file,
                            const int line)
{
    if(l_lhs != l_rhs || m_lhs != m_rhs || n_lhs != n_rhs)
    {
        std::stringstream ss;
        ss << "Dimensions not equal "
                << "(" << l_lhs << "," << m_lhs << "," << n_lhs << ") != ("
                << l_rhs << "," << m_rhs << "," << n_rhs << ")"
                << " - file:" << file << ", line:" << line;
        throw std::length_error(ss.str());
    }
}

void AssertValidRange( int index, int range_low, int range_high,
                        const std::string& file,
                        const int line)
{
    if(index < range_low || index >= range_high)
    {
        std::stringstream ss;
        ss << "Index " << index << " not in range (" << range_low << "," << range_high << ")"
                << " - file:" << file << ", line:" << line;
        throw std::out_of_range(ss.str());
    }
}

void Assert(    bool test,
                const std::string& file,
                const int line)
{
    if(!test)
    {
        std::stringstream ss;
        ss << "Assert " << " - file:" << file << ", line:" << line;
        throw std::out_of_range(ss.str());
    }
}
#endif