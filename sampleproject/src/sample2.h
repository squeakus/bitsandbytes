/**
* A completely useless Counter class.
* A class which makes you wonder why it exists.
* @author jonathan.byrne@intel.com
* @copyright Intel Internal License (see LICENSE file).
**/

class Counter {
 private:
  int counter_;

 public:
  ///counter constructor
  Counter() : counter_(0) {}

  /// @return the counter value and increments it.
  int Increment();

  /// @return counter value to STDOUT.
  void Print() const;
};
