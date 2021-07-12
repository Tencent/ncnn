// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp target update to(x)
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // Aexpected-note {{declared here}}

template <class T, class S> // Aexpected-note {{declared here}}
int tmain(T argc, S **argv) {
  int n;
  return 0;
}

int main(int argc, char **argv) {
  int m;
  #pragma omp target update // expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(m) { // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ( // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) [ // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ] // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}

  #pragma omp target update from(m) allocate(m) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp target update'}}
  {
    foo();
  }
  return tmain(argc, argv);
}
