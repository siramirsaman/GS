#include "reset_arrays.h"

#define A(ix,j) A[ix+j*N]


/** @brief Initialize or reset an example for a simple linear A.x=b
*  @param x array of unknowns [N]
*  @param b array of constant terms [N]
*  @param A array of coefficients [N*N]
*  @param N size of arrays
*  @return void
*/
void reset_arrays(real* A, real* x, real* b, int N)
{
	b[0] = 1100;
	for (int i = 1; i < N; i++)
		b[i] = 100;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A(i, j) = 0.0;

	for (int i = 0; i < N; i++)
		A(i, i) = 15.0;

	for (int i = 0; i < N; i++)
	{
		if (i + 1 < N)
			A(i, (i + 1)) = -5.0;
		if (i - 1 >= 0)
			A(i, (i - 1)) = -5.0;
	}

	A(0, 0) = 20;
	A((N - 1), (N - 1)) = 10;

	for (int i = 0; i < N; i++)
		x[i] = 0;
}