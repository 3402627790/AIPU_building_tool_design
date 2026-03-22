#include "test_utils.h"
#include "utils/json.h"
#include <fstream>
#include <sstream>
#include <iostream>
void test_json()
{
    std::ifstream ifile("test.json");
    std::ostringstream tmp;
    tmp << ifile.rdbuf();
    auto str = tmp.str();
    std::cout << str;
    auto ptr = utils::JsonObject::parse(str);
    std::cout << ptr;
    return;
}