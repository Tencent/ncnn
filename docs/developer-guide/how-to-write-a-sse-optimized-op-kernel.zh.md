# 如何使用SSE来优化算子核心

## 一：准备

### 1.背景资料

​	SSE 全称Intel® Streaming SIMD Extensions (Intel® SSE),本质是Intel公司封装汇编语句提供的底层操作指令函数集。同样属于底层操作指令集的还有著名的Intel® AVX(Advanced Vector Extensions),  及 Intel® AVX2(Intel® Advanced Vector Extensions 2)。基于同样原理封装的还有Arm 对应Arm Intrinsic，MIPS中对应MIPS Intrinsic。

​	SSE的版本包含：SSE/SSE2/SSE3/SSE4.1/SSE4.2。下文中在描述CPU特性上统称为SSE系列指令集。在描述具体使用指令函数中的CPUID Flags，才会具体区分SSE不同版本。

​	自从MSVC不再支持x64的汇编指令后（虽然可以强制使用，但不推荐不安全）。SSE，AVX等成为MSVC 支持的最佳底层优化方法。

​	本文将从SSE的使用出发，以ncnn实现为例，展示如何使用SSE优化深度学习中算子。

​	优化算子工作需要三方面的准备事项：

- 测试正确的原生代码
- 快速测试验证环境
- 基准统计程序

### 2.确认硬件是否支持SSE

​	在开始SSE优化之前，首先请确保您硬件支持SSE指令集，对于大多数Intel CPU都支持SSE指令集。但在各种系统环境下，查看方式不同。我们有：

#### 1.windows环境

​	windows环境下推荐简单使用GPU-Z来检测当前处理器是否支持SSE扩展。在GPU-Z官网下载后，运行，在“处理器”-“CPU支持的特性”项目下，若包含SSE系列指令集，即当前CPU支持SSE。

#### 2.Linux环境和类Unix环境

​	Linux环境和类Unix环境下，使用查看cpuinfo文件来确认CPU特性；

```shell
cat  /proc/cpuinfo
...
flags: *** sse sse2 ***	#在cpu flags中即可检查是否支持sse扩展
```

#### 2.macOS环境：

​	macOS本质是像Unix环境，所以同样使用sysctl 来查看CPU特性.(注意Mac的 M1 M2系列芯片是arm架构，不支持SSE)

```shell
sysctl machdep.cpu		# 结果同Linux环境
```

## 二：编写原生代码

​	使用SSE来优化算法的过程本质就是代码重构的一种情况。代码重构的首要条件是完成完备的代码行为测试集合。所以，这部分将从测试代码的编写开始。

​	其次优化过程的目标是调优某些性能指标的过程。所以第二部分将讨论性能指标的选定和优先级；

### 1.编写测试代码

​	在大多数情况下，看到这篇文章的人肯定是比笔者更会写算法，所以我在这里只谈一些编写测试的注意事项（这里的测试指验证你算法满足你的要求所编写的代码行为，跟其他人无关）。

​	编写测试代码主要注意事项：

- 思考如何构造基础数据结构才能满足算法行为的输入要求。举例来说，如果你准备为ncnn贡献算法，请阅读ncnn中关于Mat结构的函数。最好编写相关测试来验证该数据结构满足你的需要。（笔者的建议是可以先从简单结构来验证，比如需要做一个支持f32任意大小的矩阵加法算子，可以先从支持固定矩阵int8类型的加法开始编写测试代码）。
- 保持结果的正确性。首先考虑，你所编写的原生代码行为上，是否满足你所需要的结果（不论这个结果是手算的，numpy算的或者pytorch算的）。其次要考虑，结果在内存上结构如何排布。以ncnn为例，思考你的结果该如何放入到一个Mat中，Mat的size该如何设定。（在后续SSE优化中，我们将多次以原生代码结果作为target结果，验证每次优化后的正确性，原生代码能够稳定输出正确结果非常重要）
- 不用过早考虑算法的完备性，应该随着每次测试结果的正确来迭代重构算法和测试代码。二者同样重要。如果能够自动化测试，请尽量让一个简单的脚本执行来完成所有你慢吞吞的命令行。

### 2.考虑性能指标

​	性能指标的主要作用是随着每次优化的迭代，告诉我们所采取的措施在什么方面取得效果，是正面优化还是负面优化。

​	性能指标很多，包括吞吐量，还有类似计算稳定度，时间延迟，视频方面还有fps 等等。无法确认有效的性能指标也是大多数优化算法的困难点之一。	

​	随着简单粗暴地叠晶体管数量来解决电脑运行问题，性能指标似乎变得越来越不重要。这是一种错误观念，如果在单核上编写非常烂的代码，增加N个核心只是把烂代码重复N次而已。另一方面，性能指标有着客观性，在开发板上和集群设备上运行同样的算法，性能指标的优先级也不一样。但是，我认为应当满足最基础性能指标有这两个：

- **吞吐量**：算法在单位时间内执行的次数，用Gflops表示，该值越大越好（也可以认为执行同样算法所平均占用的时间，时间越短越好）；
- **性能衰退**：即随着数据规模的增加，Gflops在不同数据规模下的波动情况。更低的离散程度意味着吞吐量保持在一定范围内不发生变化。

​	其余有效的性能指标应当由业务环境和任务需求决定。负责技术基础设施建设的算法工程师，一方面应该理解业务所需求中的最高优先级，另一方面也应该追求做到更好。

​	以SSE优化算法，本质上是重构的迭代过程。不用在初期就考虑如何达到最大性能指标，而是应该考虑每次迭代中带来一定量的性能优化。

## 三：理解SSE

​	SSE主要由SSE基础数据类型 及 针对性的SSE操作函数构成。前文提到，SSE是针对汇编语句的封装，所以本身不具备错误检查和错误处理（错误检查和错误处理一般由编译器完成）。使用不当的话，诸如segmentation fault之内指针指向不存在的内存错误非常常见。我在此处建议：<u>使用SSE优化之前，确保理解代码指针位置和移动原理，原生代码已经完成测试，输出结果正确。</u>

### 1.SSE数据类型

​	SSE数据类型形如：

```c++
__m<bit><type>			 //__m适用代表申请mm寄存器
    					// bit 代表数据类型的字节长度，在SSE中为128 或 64
    					// 默认type为单精度浮点（f32），其余为int 或double
// 另外要注意所有SSE的类型除__m128和__m64外，随着版本更新有不同的类型，建议根据需要且确定硬件性能后选择合适的类型
// 举例如下：
__m128					//4xf32 含有4个单精度浮点数；SSE
__m64    				//4xf32 含有2个单精度浮点数；SSE
__m128i   				//8个int类型（8x16)		 ；SSE3
__m128d					//2个double类型(2x64)
```

### 2.SSE内联函数结构

​	SSE内联函数在线查询：[Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2) 在此 单个指令的结构如下：

- Synopsis ：摘要。描述指令的接口定义，需要引入的头文件，对应的指令，CPU必须支持的标志；
- Description：描述该指令的行为；
- Operation：逻辑层面描述指令行为；
- Performance：在不同架构中所需要的延迟和执行所需要的时钟周期数（CPI）。

​	值得指出的的是此处默认使用小端存储，即左边为高位，右边为低位。

​	相似的内联函数有很多，在使用时候一定要注意Operation中的逻辑满足您的要求。

​	另外，在ncnn中，ncnn已经将部分SSE内联函数以NCNN内联的方式封装。在为NCNN添加SSE优化的算法的过程中，请首先考虑搜索“NCNNINLINE”宏封装的SSE函数。

## 四：样例

### 1.一个简单的样例：4x4矩阵乘法

​	矩阵乘法方面，已经有很多出色的成果。值得一读的比如[how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm)，及 [以Arm Intrinsic优化矩阵乘法](https://github.com/tpoisonooo/how-to-optimize-gemm)。我建议感兴趣同学参考和学习这两份项目，来探究如何从0到1优化一份算法；

​	矩阵乘法原理很简单：

​	假设有A，B两个矩阵，如下：
$$
A_{[4][4]} =  
\begin{bmatrix}
	a_0 & a_4 & a_8 & a_{12} \\
	a_1 & a_5 & a_9 & a_{13} \\
	a_2 & a_6 & a_{10} & a_{14} \\
	a_3 & a_7 & a_{11} & a_{15} 
\end{bmatrix}
~~
B_{[4][4]} =  
\begin{bmatrix}
	b_0 & b_4 & b_8 & b_{12} \\
	b_1 & b_5 & b_9 & b_{13} \\
	b_2 & b_6 & b_{10} & b_{14} \\
	b_3 & b_7 & b_{11} & b_{15} 
\end{bmatrix}
~~
C_{[4][4]} =  
\begin{bmatrix}
	c_0 & c_4 & c_8 & c_{12} \\
	c_1 & c_5 & c_9 & c_{13} \\
	c_2 & c_6 & c_{10} & c_{14} \\
	c_3 & c_7 & c_{11} & c_{15} 
\end{bmatrix}
$$
​	对于C 矩阵的第一列，我们有：
$$
c_0 = a_0b_0 + a_4b_1 + a_8b_2 + a_{12}b_3 \\
	c_1 = a_1b_0 + a_5b_1 + a_9b_2 + a_{13}b_3 \\
	c_2 = a_2b_0 + a_6b_1 + a_{10}b_2 + a_{14}b_3 \\
	c_3 = a_3b_0 + a_7b_1 + a_{11}b_2 + a_{15}b_3
$$


#### 1.编写测试代码和基准测量程序

​	在该样例中，测试代码很容易编写出来，我们只需要初始化4x4的二维数组，并返回指针即可。此时，可以不考虑泛用性，初始化为固定值即可。

```c
// <代码片段>
...
float A[16] = {0.0f};			// 此处已经将输入和输出的矩阵默认展开成im2col 后的单行（inch = 1） 宽度为h*w = 16的矩阵
float B[16] = {0.0f};
float C[16] = {0.0f};
matrix_init_rand(A, 4, 4);		// 随机初始化A数组
matrix_init_rand(B, 4, 4);		// 随机初始化B数组
```

​	编写验证正确性的测试代码。

```c
// <代码片段>
...
float T[16] = {...};			// Target即为预测的C的结果数组，可用numpy或者纸笔计算
...
float error = 0.0001;
bool CheckAuc(T, C, error);		
// 注意：float在计算机中不能完全表示，只能使用绝对误差的判别方法。gtest等测试框架的EXCEPT宏无法处理1.234e5这样结构的float数的对比。
```

​	同样，编写计算耗时的基准测量代码，此处使用1000次操作所占的平均时间来作为基准。

```c
// <代码片段>
...
const int loop = 1000;
clock_gettime_(CLOCK_REALTIME, &time_start);
for(init i = 0; i < loop; i++)
{
	matirx_mult_native(C, A, B);
}
clock_gettime_(CLOCK_REALTIME, &time_end);
clocks_c = (time_end.tv_sec - time_start.tv_sec) * 1000000 +  (time_end.tv_sec - time_start.tv_sec) /1000;
```

#### 2.编写原生代码

​	编写原生代码，使得正确性测试能够通过。

```c
// <代码片段>
static void matirx_mult_native(float *C, float *A, float *B)
{
    for(int i_idx = 0; i_idx < 4; i_idx++)
    {
        for(int j_idx = 0; j_idx < 4; j_idx++)
        {
           for(int k_idx = 0; k_idx < 4; k_idx++)
           {
               C[4*j_idx + i_idx] += A[4*k_idx + i_idx] * B[4*j_idx + k_idx];
           }
        }
    }
}
```

#### 3.优化原生代码

​	注意到上述代码中，先取c0 - c3 的计算作为样例考虑：
$$
	c_0 = a_0b_0 + a_4b_1 + a_8b_2 + a_{12}b_3   \\
	c_1 = a_1b_0 + a_5b_1 + a_9b_2 + a_{13}b_3    \\
	c_2 = a_2b_0 + a_6b_1 + a_{10}b_2 + a_{14}b_3 \\
	c_3 = a_3b_0 + a_7b_1 + a_{11}b_2 + a_{15}b_3
$$

##### 1.装载寄存器

- 考虑竖排a0-a1-a2-a3 为4个f32 数据，又因为SSE可以申请mm寄存器，单次保存128bit，那么不妨把a0-a4保存在寄存器中，

- 对于b0-b3 则是，单次读取一个值，能够重复用4次，不妨考虑b0 重复4次，排满单个128bit的mm寄存器；

- 同理把c0-c3也放入寄存器，从列方向上考虑，取名为_c0 

  ```c++
  _m128 _a0 = _mm_load_ps(a_ptr);			//a0 -a1 -a2 -a3
  _m128 _a1 = _mm_load_ps(a_ptr + 4);		//a4 -a5 -a6 -a7
  _m128 _a2 = _mm_load_ps(a_ptr + 8);		//a8 -a9 -a10-a11
  _m128 _a3 = _mm_load_ps(a_ptr + 12);	//a12-a13-a14-a15
  
  _m128 _b0 = _mm_load_ps1(b_ptr);		// b0 - b0 - b0 - b0
  _m128 _b1 = _mm_load_ps1(b_ptr + 4);	// b1 - b1 - b1 - b1
  _m128 _b2 = _mm_load_ps1(b_ptr + 8);	// b2 - b2 - b2 - b2
  _m128 _b3 = _mm_load_ps1(b_ptr + 12);	// b3 - b3 - b3 - b3
  ```

##### 2.编写第一列的计算结果

​	对于_a0 -\_a3 数据与\_b0 数据相乘 ，有：

```c++
// 保存结果新建一个_c0 作为临时变量
_m128 _c0 = _mm_set_ps1(0.0f);
_c0 = _mm_mul_ps(_a0, _b0);
_c0 = _mm_add_ps(_mm_mul_ps(_a1, _b1),_c0);
_c0 = _mm_add_ps(_mm_mul_ps(_a2, _b2),_c0);
_c0 = _mm_add_ps(_mm_mul_ps(_a3, _b3),_c0);
// 把 _sum0存会以c指针开头的内存中，完美！
_mm_store_ps(c_ptr, _c0);
```

##### 3.将单列输出扩展到所有列：

​	我们针对剩下的c中的c1 列也做相同的操作： 对于C1 列 有：
$$
	c_4 = a_0b_4 + a_4b_5 + a_8b_6 + a_{12}b_7 \\
	c_5 = a_1b_4 + a_5b_5 + a_9b_6 + a_{13}b_7 \\
	c_6 = a_2b_4 + a_6b_5 + a_{10}b_6 + a_{14}b_7 \\
	c_7 = a_3b_4 + a_7b_5 + a_{11}b_6 + a_{15}b_7
$$


```c++
// a 系列不变 b系列指针+1
_m128 _b4 = _mm_load_ps1(b_ptr + 1);		// b4 - b4 - b4 - b4
_m128 _b5 = _mm_load_ps1(b_ptr + 4 + 1);	// b5 - b5 - b5 - b5
_m128 _b6 = _mm_load_ps1(b_ptr + 8 + 1);	// b6 - b6 - b6 - b6
_m128 _b7 = _mm_load_ps1(b_ptr + 12+ 1);	// b7 - b7 - b7 - b7

// 保存结果新建一个_c0 作为临时变量
_m128 _c1 = _mm_set_ps1(0.0f);
_c1 = _mm_mul_ps(_a0, _b4);
_c1 = _mm_add_ps(_mm_mul_ps(_a1, _b5),_c1);
_c1 = _mm_add_ps(_mm_mul_ps(_a2, _b6),_c1);
_c1 = _mm_add_ps(_mm_mul_ps(_a3, _b7),_c1);
// 把 _sum0存会以c指针开头的内存中，完美！
_mm_store_ps(c_ptr, _c1);
```

​	此时我们发现，对于C1列的操作与C0列及其相似，只不过是b_ptr的指针发生移动，不妨将其放到同一个循环中，有：

```C++
// a 系列不变
_m128 _a0 = _mm_load_ps(a_ptr);			//a0 -a1 -a2 -a3
_m128 _a1 = _mm_load_ps(a_ptr + 4);		//a4 -a5 -a6 -a7
_m128 _a2 = _mm_load_ps(a_ptr + 8);		//a8 -a9 -a10-a11
_m128 _a3 = _mm_load_ps(a_ptr + 12);	//a12-a13-a14-a15

for(int i = 0; i < 4; i++)
{
    _m128 _b0 = _mm_load_ps1(b_ptr);		// b0 - b0 - b0 - b0
    _m128 _b1 = _mm_load_ps1(b_ptr + 4);	// b1 - b1 - b1 - b1
    _m128 _b2 = _mm_load_ps1(b_ptr + 8);	// b2 - b2 - b2 - b2
    _m128 _b3 = _mm_load_ps1(b_ptr + 12);	// b3 - b3 - b3 - b3
    
    _m128 _ci = _mm_set_ps1(0.0f);
    _ci = _mm_mul_ps(_a0, _b0);
    _ci = _mm_add_ps(_mm_mul_ps(_a1, _b1),_ci);
    _ci = _mm_add_ps(_mm_mul_ps(_a2, _b2),_ci);
    _ci = _mm_add_ps(_mm_mul_ps(_a3, _b3),_ci);
    
    _mm_store_ps(c_ptr, _ci);
    
    b_ptr += 1;				// 移动b_ptr
    c_ptr += 4;				// 移动保存内存的c_ptr
}
```

### 2.NCNN中以SSE优化算子的注意事项

#### 1.线程与openmp

​	以上计算Benchmark 和 SSE优化的方法大多集中在单个核心中，但是在实际使用ncnn中，ncnn使用Option opt 中提供的num_threads 给openmp赋值，以实现多线程并行化，同时运行在多个核心上。

```c++
#pragma omp parallel for num_threads(opt.num_threads)
```

​	在优化成SSE代码的初期，可以考虑锁定为单线程，或者直接不用考虑线程的影响，仅对单核以SSE优化，保证单核的结果正确后，再加上opt的多线程进行结果测试。

#### 2.展开循环

​	在实际ncnn实现的原生代码的算法中，循环是非常常见的。针对以SSE优化这类循环，遵循非常简单的原则：循环中，迭代器等于零时刻，整个输出的结果也是正确的。

​	那么，在我们使用SSE优化过程中，不妨以迭代器等于零的时刻，函数计算结果作为此时目标结果。在此基础上再利用SSE优化代码。与目标结果核对正确以后，再进一步去考虑迭代器等于1的情况（重复这个过程直到迭代器达到最大值）。在迭代器的每个元素下，SSE优化出的代码都与结果相等，那么我们可以说，该次优化是正确性，且完全覆盖了需执行代码。（一般来说不用考虑到最大值，根据数学归纳法，n有效，n+1有效，那么n的序列都是有效的）

## 五：总结

​	本文描述SSE的使用及以4x4矩阵乘法的样例来优化SSE代码。

​	值得注意的是，SSE只是128bit数据宽度的指令集，但是也可以用来模拟256bit 和 512bit数据宽度，来实现以pack4拼接成pack8，甚至pack16的做法，只不过在输出结果管理上更加繁琐而已。感兴趣的同学可以尝试一下。

## 六：引用

1. [SSE指令扩展快查](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2)；
2. 浮点性能基准计算-[浮点峰值那些事儿](https://zhuanlan.zhihu.com/p/28226956)
3. 硬件性能基准测试计算样例：[M1芯片搞数据科学好使吗？5种基准测试给你答案](https://mp.weixin.qq.com/s/2N5cl_Z1MRF8dfbRo-sb4A)
4. 讨论矩阵乘法如何优化的系列论文：[how-to-optimized-gemm](https://github.com/flame/how-to-optimize-gemm/wiki)
5. 讨论以Arm Intrinsic 优化gemm的系列文章：[OpenBLAS gemm从零入门](https://zhuanlan.zhihu.com/p/65436463)
