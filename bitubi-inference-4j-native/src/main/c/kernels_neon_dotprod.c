/* The NEON kernels compiled with dotprod (kernels_neon.c picks that path through __ARM_FEATURE_DOTPROD). */
#define BI4J_NEON_TABLE bi4j_kernels_neon_dotprod
#include "kernels_neon.c"
