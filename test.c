#include <stdio.h>
#include <stdint.h>

int main()
{
    uint8_t my_counter = 254;
    my_counter = my_counter + 1;
    // statt “int my_counter“
    // alle bekannten Operatoren
    // funktionieren: +, -, *, /, %
    // gibt 255 aus
    printf("%d\n", my_counter);
    my_counter = my_counter + 1;
    printf("%d\n", my_counter);
    return 0;
}