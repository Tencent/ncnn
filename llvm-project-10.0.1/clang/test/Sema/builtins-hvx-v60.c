// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon %s -target-feature +hvx-length128b -target-feature +hvxv60 -target-cpu hexagonv60 -fsyntax-only -verify

typedef long Vect1024 __attribute__((__vector_size__(128)))
    __attribute__((aligned(128)));
typedef long Vect2048 __attribute__((__vector_size__(256)))
    __attribute__((aligned(128)));

typedef Vect1024 HVX_Vector;
typedef Vect2048 HVX_VectorPair;


HVX_Vector builtin_needs_v60(HVX_VectorPair a) {
  return __builtin_HEXAGON_V6_hi_128B(a);
}

HVX_Vector builtin_needs_v62(char a) {
  // expected-error@+1 {{builtin is not supported on this version of HVX}}
  return __builtin_HEXAGON_V6_lvsplatb_128B(a);
}

HVX_VectorPair builtin_needs_v65() {
  // expected-error@+1 {{builtin is not supported on this version of HVX}}
  return __builtin_HEXAGON_V6_vdd0_128B();
}
