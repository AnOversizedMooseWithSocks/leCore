#include <float.h>
#include <fenv.h>
#include <math.h>

int main(void)
{
    volatile double one = 1.0;
    volatile double zero = 0.0;
    volatile float large = 16777216.0F;
    double infinity = one / zero;
    double not_a_number = zero / zero;
    float rounded = large + 1.0F;

    if (FLT_EVAL_METHOD != 0 || fegetround() != FE_TONEAREST) {
        return 1;
    }
    if (!isinf(infinity) || signbit(infinity) ||
        !isinf(-infinity) || !signbit(-infinity)) {
        return 2;
    }
    if (!isnan(not_a_number) || !isnan(not_a_number + one)) {
        return 3;
    }
    if (rounded != 16777216.0F) {
        return 4;
    }
    return 0;
}
