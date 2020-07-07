### what is packing and why

packing is the form of storing multiple short-sized values as one long-sized value.

element packing is well mapped with the underlying simd register, which usually use one very wide register to store different types of values.

|C|elemsize|elempack|
|---|---|---|
|double|8|1|
|float|4|1|
|int|4|1|
|short|2|1|
|signed char|1|1|

|arm neon|elemsize|elempack|
|---|---|---|
|float64x2_t|16|2|
|float32x4_t|16|4|
|int32x4_t|16|4|
|float16x4_t|8|4|
|int8x8_t|8|8|

Though the real count of values doubles when elempack is two, the wide-sized value is still treated as one value in the view of Mat structure. For example, we want to store 40 float values in Mat object, if elempack 1 is used, Mat width is then 40, while 10 if elempack 4 is used.

|dims|w|h|c|cstep|elemsize|elempack|
|---|---|---|---|---|---|---|
|1|40|1|1|40|4|1|
|1|10|1|1|10|16|4|

### packing style convention

In practise, elempack 1, 4, 8 are the most common cases. It is possible to use any other packing style in theory.

The following table show the packing axis used in ncnn for different dimension.

|dims|packing axis|shape before packing|shape after packing|
|---|---|---|---|
|1|w|w|w/elempack|
|2|h|w, h|w, h/elempack|
|3|c|w, h, c|w, h, c/elempack|

If the packing axis dim is not evenly divisible by elempack, zero padding may be used.

```
outw = (w + elempack - 1) / elempack;
```

The following snippet shows the memory layout after elempack=4 on 3-dim Mat

```
// w=2 h=3 c=4 elempack=1
0 1
2 3
4 5

6 7
8 9
10 11

12 13
14 15
16 17

18 19
20 21
22 23

// w=2 h=3 c=1 elempack=4
(0,6,12,18) (1,7,13,19)
(2,8,14,20) (3,9,15,21)
(4,10,16,22) (5,11,17,23)
```

### how to convert elempack

There is a convenient wrapper function provided
```
// convert to elempack 4 if packing axis dim is evenly divisible by elempack
// return the identity Mat otherwise
ncnn::Mat a;
ncnn::Mat a_packed;
ncnn::convert_packing(a, a_packed, 4);
if (a_packed.elempack == 4)
{
    // check if packing is successful
}

// convert to packing 1, aka unpacking, shall be always successful
ncnn::Mat b;
ncnn::Mat b_unpacked;
ncnn::convert_packing(b, b_unpacked, 1);
```

### handle general interleaved data

Here is an example of using convert packing to convert RGB interleaved data to planar

**NOTE:** The following code is just presented to explain what packing is and the conversion process. Do not use it in production due to its poor performance. Do use ncnn::Mat::from_pixels()

```cpp
// rgb_interleaved_u8 is RGB RGB RGB ...
// rgb_interleaved_u8.w = w;
// rgb_interleaved_u8.h = h;
// rgb_interleaved_u8.c = 1;
// rgb_interleaved_u8.elemsize = 3;
// rgb_interleaved_u8.elempack = 3;

ncnn::Mat rgb_interleaved_u8(w, h, 1, 3, 3);
ncnn::Mat rgb_planar_u8;

ncnn::convert_packing(rgb_interleaved_u8, rgb_planar_u8, 1);

// rgb_planar_u8 is now RRR ... GGG ... BBB ...
// rgb_planar_u8.w = w;
// rgb_planar_u8.h = h;
// rgb_planar_u8.c = 3;
// rgb_planar_u8.elemsize = 1;
// rgb_planar_u8.elempack = 1;
```
