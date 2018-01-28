#include "sse.h"
#include <stdint.h>

#define INP(x,y) inp[(x)*ncols/8 + (y)/8]
#define OUT(x,y) out[(y)*nrows/8 + (x)/8]

// LSBit-first
void
ssebmx(char const *inp, char *out, int nrows, int ncols)
{
#define II (i)
#include "ssebmx.src"
}

// MSBit-first
void
ssebmx_m(char const *inp, char *out, int nrows, int ncols)
{
#undef II
#define II (i ^ 7)
#include "ssebmx.src"
}
