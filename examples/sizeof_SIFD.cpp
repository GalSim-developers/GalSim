/// @file sizeof_SIFD.cpp @brief Prints short, int, float and double sizes to stdout. 
#include <cstdio>

int main(){

    printf("Sizeof short (bits) = %zu\n", sizeof(short) * 8);
    printf("Sizeof int (bits) = %zu\n", sizeof(int) * 8);
    printf("Sizeof long (bits) = %zu\n", sizeof(long) * 8);
    printf("Sizeof float (bits) = %zu\n", sizeof(float) * 8);
    printf("Sizeof double (bits) = %zu\n", sizeof(double) * 8);

}
