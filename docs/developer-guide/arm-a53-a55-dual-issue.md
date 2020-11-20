## natural assembly
* no register dependency, no penalty
```
ld1     {v0.4s}, [r0], #16
fmla    v10.4s, v16.4s, v24.s[0]
fmla    v11.4s, v16.4s, v24.s[1]
fmla    v12.4s, v16.4s, v24.s[2]
fmla    v13.4s, v16.4s, v24.s[3]
```

## A53
* 128bit vector load cannot be dual issued with fmla, wait 2 cycles
* 64bit vector load cannot be dual issued with fmla, wait 1 cycle
* 64bit integer load can be dual issued with fmla, no penalty
* pointer update can be dual issued with fmla, no penalty
* 64bit vector load and 64bit vector insert can be dual issued, no penalty
* any vector load cannot be issued on the 4th cycle of each fmla (enters the accumulator pipeline)

### practical guide
* use 64bit vector load only
* issue vector load every three fmla
* 1 cycle to load 64bit, dual issue with the previous interleaved 64bit insert
* load the remaining 64bit into integer register, dual issue with fmla
* update pointer, dual issue with fmla
* insert 64bit into vector from integer register, dual issue with the next interleaved 64bit load
* add nop every three fmla if no load, seems to be faster
```
ldr     d0, [r0] // 1 cycle, v0 first 64bit
fmla
ldr     x23, [r0, #8] // 0 cycle, v0 second 64bit to temp register
fmla
add     r0, r0, #16 // 0 cycle, update pointer
fmla
ldr     d1, [r0] // 1 cycle, v1 first 64bit
ins     v0.d[1], x23 // 0 cycle, v0 second 64bit complete
fmla
ldr     x23, [r0, #8] // 0 cycle, v1 second 64bit to temp register
fmla
add     r0, r0, #16 // 0 cycle, update pointer
fmla
ins     v1.d[1], x23 // 1 cycle, v1 second 64bit complete
nop
fmla
fmla
fmla
nop
nop
fmla
fmla
fmla
```

## A55
* 128bit vector load cannot be dual issued with fmla, wait 2 cycles
* 64bit vector load can be dual issued with fmla, no penalty
* 64bit integer load can be dual issued with fmla, no penalty
* pointer update can be dual issued with fmla, no penalty
* 64bit vector insert can be dual issued with fmla, no penalty

### practical guide
* use 64bit vector load only
* load 64bit, dual issue with fmla
* load the remaining 64bit into integer register, dual issue with fmla
* update pointer, dual issue with fmla
* insert 64bit into vector from integer register, dual issue with fmla
* interleaved load loose register dependency
* nop trick is not needed
```
ldr     d0, [r0] // 0 cycle, v0 first 64bit
fmla
ldr     x23, [r0, #8] // 0 cycle, v0 second 64bit to temp register
fmla
add     r0, r0, #16 // 0 cycle, update pointer
fmla
ldr     d1, [r0] // 0 cycle, v1 first 64bit
fmla
ins     v0.d[1], x23 // 0 cycle, v0 second 64bit complete
fmla
ldr     x23, [r0, #8] // 0 cycle, v1 second 64bit to temp register
fmla
add     r0, r0, #16 // 0 cycle, update pointer
fmla
ins     v1.d[1], x23 // 0 cycle, v1 second 64bit complete
fmla
```
