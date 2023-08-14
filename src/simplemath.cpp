#include "platform.h"
#ifdef NCNN_SIMPLEMATH
#include <stdlib.h>
#include<stdio.h>
float absolute(float x)//绝对值
{
	if (x < 0)x = -x;
	return x;
}

float Factorial(int x)//阶乘
{
	if (x == 1 || x == 0)return 1;
	else
		return 1.0 * x * Factorial(x - 1);
}


float nth(float x, int n)//n次方
{
	if (n > 0)
	{
		return x * nth(x, n - 1);
	}
	if (n == 0)
	{
		return 1;
	}
	if (n < 0)
	{
		return (1 / x) * nth(x, n + 1);
	}
}
float Bernoulli(int x)//伯努利数
{
	int k = x;
	float B = 0;
	if (x == 0)
		return 1;
	else
		if (x > 1 && x % 2 == 1)
		{
			return 0;
		}
		else
		{
			while (k)
			{
				k--;
				B += -1.0 * (Factorial(x) * Bernoulli(k)) / (Factorial(x - k) * Factorial(k) * (x - k + 1));
			}
			return B;
		}
}
float tan(float x)//tan(x)  精度保持0.000001 计算速度太慢
{
	int i = 1;
	float e = 1, sum = 0;
	while (x < (-pai / 2))x += pai;
	while (x > (pai / 2))x -= pai;
	if (x == (pai / 2))
	{
		printf("\tNaN\n");
		return 0;
	}
	while (absolute(e) > accuracy && i <= 24)//考虑计算速度，控制Bernoulli()
	{
		e = 1.0 * (nth(-1, i - 1) * nth(2, 2 * i) * (nth(2, 2 * i) - 1.0) * Bernoulli(2 * i) * nth(x, 2 * i - 1)) / (Factorial(2 * i));
		sum += e;
		i++;
	}
	return sum;
}
#endif



