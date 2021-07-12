// RUN: %clang -target le64-unknown-unknown -### %s -emit-llvm-only -c 2>&1 | FileCheck %s -check-prefix=ECHO
// RUN: %clang -target le64-unknown-unknown %s -emit-llvm -S -c -o - | FileCheck %s

// ECHO: {{.*}} "-cc1" {{.*}}le64-unknown-unknown.c

typedef __builtin_va_list va_list;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

extern "C" {

// CHECK: @align_c = dso_local global i32 1
int align_c = __alignof(char);

// CHECK: @align_s = dso_local global i32 2
int align_s = __alignof(short);

// CHECK: @align_i = dso_local global i32 4
int align_i = __alignof(int);

// CHECK: @align_l = dso_local global i32 8
int align_l = __alignof(long);

// CHECK: @align_ll = dso_local global i32 8
int align_ll = __alignof(long long);

// CHECK: @align_p = dso_local global i32 8
int align_p = __alignof(void*);

// CHECK: @align_f = dso_local global i32 4
int align_f = __alignof(float);

// CHECK: @align_d = dso_local global i32 8
int align_d = __alignof(double);

// CHECK: @align_ld = dso_local global i32 8
int align_ld = __alignof(long double);

// CHECK: @align_vl = dso_local global i32 4
int align_vl = __alignof(va_list);

// CHECK: __LITTLE_ENDIAN__defined
#ifdef __LITTLE_ENDIAN__
void __LITTLE_ENDIAN__defined() {}
#endif

// CHECK: __le64defined
#ifdef __le64
void __le64defined() {}
#endif

// CHECK: __le64__defined
#ifdef __le64__
void __le64__defined() {}
#endif

// CHECK: unixdefined
#ifdef unix
void unixdefined() {}
#endif

// CHECK: __unixdefined
#ifdef __unix
void __unixdefined() {}
#endif

// CHECK: __unix__defined
#ifdef __unix__
void __unix__defined() {}
#endif

// CHECK: __ELF__defined
#ifdef __ELF__
void __ELF__defined() {}
#endif

// Check types

// CHECK: signext i8 @check_char()
char check_char() { return 0; }

// CHECK: signext i16 @check_short()
short check_short() { return 0; }

// CHECK: i32 @check_int()
int check_int() { return 0; }

// CHECK: i64 @check_long()
long check_long() { return 0; }

// CHECK: i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK: zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK: zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK: i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK: i64 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK: i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK: i64 @check_size_t()
size_t check_size_t() { return 0; }

// CHECK: i64 @check_ptrdiff_t()
ptrdiff_t check_ptrdiff_t() { return 0; }

// CHECK: float @check_float()
float check_float() { return 0; }

// CHECK: double @check_double()
double check_double() { return 0; }

// CHECK: double @check_longdouble()
long double check_longdouble() { return 0; }

}

template<int> void Switch();
template<> void Switch<4>();
template<> void Switch<8>();
template<> void Switch<16>();

void check_pointer_size() {
  // CHECK: SwitchILi8
  Switch<sizeof(void*)>();

  // CHECK: SwitchILi16
  Switch<sizeof(va_list)>();
}
