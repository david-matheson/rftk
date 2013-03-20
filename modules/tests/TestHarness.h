#include <iostream>

bool report_test(std::string const& test_name, bool result) {
    if (result) {
        std::cout << "SUCCESS: " << test_name << std::endl;
        return true;
    } else {
        std::cout << "FAILURE: " << test_name << std::endl;
        return false;
    }
}
#define RUN_TEST(test_name) all_success &= report_test(#test_name, test_name());

