```c
// v寄存器全部使用 %.4s
// 128-bit vreg matches %.4s
// a += b * c
float32x4_t _a = vld1q_f32(a);
float32x4_t _b = vld1q_f32(b);
float32x4_t _c = vld1q_f32(c);
asm volatile(
    "fmla  %0.4s, %2.4s, %3.4s"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// v寄存器使用低64位  %.2s
// low 64-bit vreg matches %.2s
// a += b * c
float32x2_t _a = vld1_f32(a);
float32x2_t _b = vld1_f32(b);
float32x2_t _c = vld1_f32(c);
asm volatile(
    "fmla  %0.2s, %2.2s, %3.2s"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```
```c
// v寄存器单路使用 %.s[0] %.s[1] %.s[2] %.s[3]
// 32-bit register matches %.s[0]
// a += b * c[0]
// a += b * c[1]
// a += b * c[2]
// a += b * c[3]
float32x4_t _a = vld1_f32(a);
float32x4_t _b = vld1_f32(b);
float32x4_t _c = vld1_f32(c);
asm volatile(
    "fmla  %0.4s, %2.4s, %3.s[0]"
    "fmla  %0.4s, %2.4s, %3.s[1]"
    "fmla  %0.4s, %2.4s, %3.s[2]"
    "fmla  %0.4s, %2.4s, %3.s[3]"
    : "=w"(_a) // %0
    : "0"(_a),
      "w"(_b), // %2
      "w"(_c)  // %3
    :
);
```


qwq
