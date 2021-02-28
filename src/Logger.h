#pragma once

#include <iostream>

#define LOG_ENDL do { \
    std::cout << std::endl; \
} while(0);

#define LOG(x) do { \
    std::cout << x << std::endl; \
} while(0);
