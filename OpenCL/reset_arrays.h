#ifndef RESET_ARRAYS
#define RESET_ARRAYS


#include"precision.h"

/** @brief Initialize or reset an example for a simple linear A.x=b
*  @param x array of unknowns [N]
*  @param b array of constant terms [N]
*  @param A array of coefficients [N*N]
*  @param N size of arrays
*  @return void
*/
void reset_arrays(real* A, real* x, real* b, int N);


#endif