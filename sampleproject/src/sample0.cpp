/** \file
* A brief overview of the file.
* A more elaborate description.
* @author jonathan.byrne@intel.com
* @copyright Intel Internal License (see LICENSE file).
**/
#include <math.h>

/// Computes the square root.
/// @param a a double.
double squareRoot(const double a) {

    double b = sqrt(a);
    if(b != b) { // nan check
        return -1.0;
    }else{
        return sqrt(a);
    }
}
