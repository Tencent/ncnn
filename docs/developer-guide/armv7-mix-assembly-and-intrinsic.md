```c
// d寄存器全部使用 %P
// d reg matches %P
// a += b * c
float32x2_t _a = vld1_f32(a);
float32x2_t _b = vld1_f32(b);
float32x2_t _c = vld1_f32(c);
asm volatile(
    "vmla.f32  %P0, %P2, %P3"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// q寄存器全部使用 %q
// q reg matches %q
// a += b * c
float32x4_t _a = vld1q_f32(a);
float32x4_t _b = vld1q_f32(b);
float32x4_t _c = vld1q_f32(c);
asm volatile(
    "vmla.f32  %q0, %q2, %q3"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// d寄存器单路使用 %P[0] %P[1]
// 32bit d reg matches %P[0]
// a += b * c[0]
// a += b * c[1]
float32x2_t _a = vld1_f32(a);
float32x2_t _b = vld1_f32(b);
float32x2_t _c = vld1_f32(c);
asm volatile(
    "vmla.f32  %P0, %P2, %P3[0]"
    "vmla.f32  %P0, %P2, %P3[1]"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// q寄存器单路使用 %e[0] %e[1] %f[0] %f[1]
// 32-bit q reg matches %e[0]
// a += b * c[0]
// a += b * c[1]
// a += b * c[2]
// a += b * c[3]
float32x4_t _a = vld1q_f32(a);
float32x4_t _b = vld1q_f32(b);
float32x4_t _c = vld1q_f32(c);
asm volatile(
    "vmla.f32  %q0, %q2, %e3[0]"
    "vmla.f32  %q0, %q2, %e3[1]"
    "vmla.f32  %q0, %q2, %f3[0]"
    "vmla.f32  %q0, %q2, %f3[1]"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// q寄存器拆分d寄存器使用 %e %f
// use %e %f to split q reg into two d regs
// a += b * c[0]c[1]
// a += b * c[2]c[3]
float32x2_t _a = vldq_f32(a);
float32x2_t _b = vldq_f32(b);
float32x4_t _c = vld1q_f32(c);
asm volatile(
    "vmla.f32  %P0, %P2, %e3"
    "vmla.f32  %P0, %P2, %f3"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// d寄存器声明绑定
// specify concrete d reg which want to save
// vmla.f32  d0, d2, d4
register float32x2_t _a asm("d0") = vld1_f32(a);
register float32x2_t _b asm("d2") = vld1_f32(b);
register float32x2_t _c asm("d4") = vld1_f32(c);

asm volatile(
    "vmla.f32  %P0, %P2, %P3"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// q寄存器声明绑定
// bind q reg with data
// vmla.f32  q0, q1, q2
register float32x4_t _a asm("q0") = vld1q_f32(a);
register float32x4_t _b asm("q1") = vld1q_f32(b);
register float32x4_t _c asm("q2") = vld1q_f32(c);

asm volatile(
    "vmla.f32  %q0, %q2, %q3"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```

如果不是因为编译器的bug，寄存器绑定是用不着的，然而。。。

https://gcc.gnu.org/bugzilla/show_bug.cgi?id=41538

qwq
