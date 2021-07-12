// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

struct A {
  // This conversion operator allows implicit conversion to bool but not to other integer types.
  typedef A * (A::*MemberPointer);
  operator MemberPointer() const { return m_ptr ? &A::m_ptr : 0; }

  A *m_ptr;

  A *getPtr();
  typedef A * (A::*MemberFnPointer)(void);
};

void testConditionalUse() {
  A obj;

  obj.m_ptr = &obj;
  clang_analyzer_eval(obj.m_ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(&A::m_ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(obj); // expected-warning{{TRUE}}

  obj.m_ptr = 0;
  clang_analyzer_eval(obj.m_ptr); // expected-warning{{FALSE}}
  clang_analyzer_eval(A::MemberPointer(0)); // expected-warning{{FALSE}}
  clang_analyzer_eval(obj); // expected-warning{{FALSE}}

  clang_analyzer_eval(&A::getPtr); // expected-warning{{TRUE}}
  clang_analyzer_eval(A::MemberFnPointer(0)); // expected-warning{{FALSE}}
}


void testComparison() {
  clang_analyzer_eval(&A::getPtr == &A::getPtr); // expected-warning{{TRUE}}
  clang_analyzer_eval(&A::getPtr == 0); // expected-warning{{FALSE}}

  clang_analyzer_eval(&A::m_ptr == &A::m_ptr); // expected-warning{{TRUE}}
}

namespace PR15742 {
  template <class _T1, class _T2> struct A {
    A (const _T1 &, const _T2 &);
  };
  
  typedef void *NPIdentifier;

  template <class T> class B {
  public:
    typedef A<NPIdentifier, bool (T::*) (const NPIdentifier *, unsigned,
                                         NPIdentifier *)> MethodMapMember;
  };

  class C : public B<C> {
  public:
    bool Find(const NPIdentifier *, unsigned, NPIdentifier *);
  };

  void InitStaticData () {
    C::MethodMapMember(0, &C::Find); // don't crash
  }
}

bool testDereferencing() {
  A obj;
  obj.m_ptr = 0;

  A::MemberPointer member = &A::m_ptr;

  clang_analyzer_eval(obj.*member == 0); // expected-warning{{TRUE}}

  member = 0;

  return obj.*member; // expected-warning{{The result of the '.*' expression is undefined}}
}

namespace testPointerToMemberFunction {
  struct A {
    virtual int foo() { return 1; }
    int bar() { return 2; }
    int static staticMemberFunction(int p) { return p + 1; };
  };

  struct B : public A {
    virtual int foo() { return 3; }
  };

  typedef int (A::*AFnPointer)();
  typedef int (B::*BFnPointer)();

  void testPointerToMemberCasts() {
    AFnPointer AFP = &A::bar;
    BFnPointer StaticCastedBase2Derived = static_cast<BFnPointer>(&A::bar),
               CCastedBase2Derived = (BFnPointer) (&A::bar);
    A a;
    B b;

    clang_analyzer_eval((a.*AFP)() == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval((b.*StaticCastedBase2Derived)() == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(((b.*CCastedBase2Derived)() == 2)); // expected-warning{{TRUE}}
  }

  void testPointerToMemberVirtualCall() {
    A a;
    B b;
    A *APtr = &a;
    AFnPointer AFP = &A::foo;

    clang_analyzer_eval((APtr->*AFP)() == 1); // expected-warning{{TRUE}}

    APtr = &b;

    clang_analyzer_eval((APtr->*AFP)() == 3); // expected-warning{{TRUE}}
  }

  void testPointerToStaticMemberCall() {
    int (*fPtr)(int) = &A::staticMemberFunction;
    if (fPtr != 0) { // no-crash
      clang_analyzer_eval(fPtr(2) == 3); // expected-warning{{TRUE}}
    }
  }
} // end of testPointerToMemberFunction namespace

namespace testPointerToMemberData {
  struct A {
    int i;
    static int j;
  };

  void testPointerToMemberData() {
    int A::*AMdPointer = &A::i;
    A a;

    a.i = 42;
    a.*AMdPointer += 1;

    clang_analyzer_eval(a.i == 43); // expected-warning{{TRUE}}

    int *ptrToStaticField = &A::j;
    if (ptrToStaticField != 0) {
      *ptrToStaticField = 7;
      clang_analyzer_eval(*ptrToStaticField == 7); // expected-warning{{TRUE}}
      clang_analyzer_eval(A::j == 7); // expected-warning{{TRUE}}
    }
  }
} // end of testPointerToMemberData namespace

namespace testPointerToMemberMiscCasts {
struct B {
  int f;
};

struct D : public B {
  int g;
};

void foo() {
  D d;
  d.f = 7;

  int B::* pfb = &B::f;
  int D::* pfd = pfb;
  int v = d.*pfd;

  clang_analyzer_eval(v == 7); // expected-warning{{TRUE}}
}
} // end of testPointerToMemberMiscCasts namespace

namespace testPointerToMemberMiscCasts2 {
struct B {
  int f;
};
struct L : public B { };
struct R : public B { };
struct D : public L, R { };

void foo() {
  D d;

  int B::* pb = &B::f;
  int L::* pl = pb;
  int R::* pr = pb;

  int D::* pdl = pl;
  int D::* pdr = pr;

  clang_analyzer_eval(pdl == pdr); // expected-warning{{FALSE}}
  clang_analyzer_eval(pb == pl); // expected-warning{{TRUE}}
}
} // end of testPointerToMemberMiscCasts2 namespace

namespace testPointerToMemberDiamond {
struct B {
  int f;
};
struct L1 : public B { };
struct R1 : public B { };
struct M : public L1, R1 { };
struct L2 : public M { };
struct R2 : public M { };
struct D2 : public L2, R2 { };

void diamond() {
  M m;

  static_cast<L1 *>(&m)->f = 7;
  static_cast<R1 *>(&m)->f = 16;

  int L1::* pl1 = &B::f;
  int M::* pm_via_l1 = pl1;

  int R1::* pr1 = &B::f;
  int M::* pm_via_r1 = pr1;

  clang_analyzer_eval(m.*(pm_via_l1) == 7); // expected-warning {{TRUE}}
  clang_analyzer_eval(m.*(pm_via_r1) == 16); // expected-warning {{TRUE}}
}

void double_diamond() {
  D2 d2;

  static_cast<L1 *>(static_cast<L2 *>(&d2))->f = 1;
  static_cast<L1 *>(static_cast<R2 *>(&d2))->f = 2;
  static_cast<R1 *>(static_cast<L2 *>(&d2))->f = 3;
  static_cast<R1 *>(static_cast<R2 *>(&d2))->f = 4;

  clang_analyzer_eval(d2.*(static_cast<int D2::*>(static_cast<int L2::*>(static_cast<int L1::*>(&B::f)))) == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(d2.*(static_cast<int D2::*>(static_cast<int R2::*>(static_cast<int L1::*>(&B::f)))) == 2); // expected-warning {{TRUE}}
  clang_analyzer_eval(d2.*(static_cast<int D2::*>(static_cast<int L2::*>(static_cast<int R1::*>(&B::f)))) == 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(d2.*(static_cast<int D2::*>(static_cast<int R2::*>(static_cast<int R1::*>(&B::f)))) == 4); // expected-warning {{TRUE}}
}
} // end of testPointerToMemberDiamond namespace

namespace testAnonymousMember {
struct A {
  struct {
    int x;
  };
  struct {
    struct {
      int y;
    };
  };
  struct {
    union {
      int z;
    };
  };
};

void test() {
  clang_analyzer_eval(&A::x); // expected-warning{{TRUE}}
  clang_analyzer_eval(&A::y); // expected-warning{{TRUE}}
  clang_analyzer_eval(&A::z); // expected-warning{{TRUE}}

  // FIXME: These should be true.
  int A::*l = &A::x, A::*m = &A::y, A::*n = &A::z;
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(m); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(n); // expected-warning{{UNKNOWN}}

  // FIXME: These should be true as well.
  A a;
  a.x = 1;
  clang_analyzer_eval(a.*l == 1); // expected-warning{{UNKNOWN}}
  a.y = 2;
  clang_analyzer_eval(a.*m == 2); // expected-warning{{UNKNOWN}}
  a.z = 3;
  clang_analyzer_eval(a.*n == 3); // expected-warning{{UNKNOWN}}
}
} // end of testAnonymousMember namespace
