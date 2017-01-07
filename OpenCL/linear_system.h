#ifndef linear_system
#define linear_system

/*! Define a linear system in the form of: A[N*N] . x[N] = b[N] */
class Linear_System
{
public:
	real * A; /**< array of coefficients [N*N] */
	real * x; /**< array of unknowns [N] */
	real * b; /**< array of constant terms [N] */
	Linear_System(int N)
	{
		A = (real *)malloc(sizeof(real) *N*N);
		x = (real *)malloc(sizeof(real) *N);
		b = (real *)malloc(sizeof(real) *N);
	}
	~Linear_System()
	{
		free(A);
		free(x);
		free(b);
	}
};


#endif