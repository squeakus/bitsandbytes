/**
* A completely useless Counter class.
* A class which makes you wonder why it exists.
* @author jonathan.byrne@intel.com
* @copyright Intel Internal License (see LICENSE file).
**/
#include <stdio.h>

#include "sample2.h"

/// Increments the current counter
int Counter::Increment() {
  return counter_++;
}

/// Prints the current counter value to STDOUT.
void Counter::Print() const {
  printf("%d", counter_);
}
