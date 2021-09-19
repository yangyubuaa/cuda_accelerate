#include <iostream>

int main(){
    int *ptr;
    int a = 10;
    ptr = &a;
    std::cout << ptr << std::endl;
    std::cout << *((int*)ptr) << std::endl;
    return 0;
}