# include <cstdlib>
# include <iostream>

using namespace std;

void junk_data ( );
int main ( );

//****************************************************************************80

int main ( )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for TEST01.
//
//  Discussion:
//
//    TEST02 has some uninitialized data.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 May 2011
//
{
  cout << "\n";
  cout << "TEST02:\n";
  cout << "  C++ version\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  junk_data ( );
//
//  Terminate.
//
  cout << "\n";
  cout << "TEST02\n";
  cout << "  Normal end of execution.\n";

  return 0;
}
//****************************************************************************80

void junk_data ( )

//****************************************************************************80
//
//  Purpose:
//
//    JUNK_DATA has some uninitialized variables.
//
//  Discussion:
//
//    VALGRIND's MEMCHECK program monitors uninitialized variables, but does
//    not complain unless such a variable is used in a way that means its
//    value affects the program's results, that is, the value is printed,
//    or computed with.  Simply copying the unitialized data to another variable
//    is of no concern.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 May 2011
//

// Original code:
// ERROR SUMMARY: 24 errors from 4 contexts (suppressed: 0 from 0)
// Fixed version:
// ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
{
  int i;
  int *x;

  // Error 1. The original code does not initialize the dynamic array x.
  // note: use valgrind memcheck --track-original=yes to trace the original problem
  // While using dynamic arrays, if we use "new int[n]", then it will generate an
  // array with n unintialized int.
  // However, if we use "new int[n]()" instead, then it will generate an array
  // with n intialized int.
  x = new int[10]();
//
//  X = { 0, 1, 2, 3, 4, ?a, ?b, ?c, ?d, ?e }.
//
  for ( i = 0; i < 5; i++ )
  {
    x[i] = i;
  }
//
//  Copy some values.
//  X = { 0, 1, ?c, 3, 4, ?b, ?b, ?c, ?d, ?e }.
//
  x[2] = x[7];
  x[5] = x[6];
//
//  Modify some uninitialized entries.
//  Memcheck doesn't seem to care about this.
//
  for ( i = 0; i < 10; i++ )
  {
    x[i] = 2 * x[i];
  }
//
//  Print X.
//
  for ( i = 0; i < 10; i++ )
  {
    cout << "  " << i << "  " << x[i] << "\n";
  }

  delete [] x;

  return;
}
