// Copyright 2009-2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Floating-point mod function.
*/

// Mod returns the floating-point remainder of x/y.
// The magnitude of the result is less than y and its
// sign agrees with that of x.
//
// Special cases are:
//	Mod(±Inf, y) = NaN
//	Mod(NaN, y) = NaN
//	Mod(x, 0) = NaN
//	Mod(x, ±Inf) = x
//	Mod(x, NaN) = NaN

//extern fmod
func libc_fmod(float64, float64) float64

func Mod(x, y float64) float64 {
	return libc_fmod(x, y)
}

func mod(x, y float64) float64 {
	if y == 0 || IsInf(x, 0) || IsNaN(x) || IsNaN(y) {
		return NaN()
	}
	if y < 0 {
		y = -y
	}

	yfr, yexp := Frexp(y)
	sign := false
	r := x
	if x < 0 {
		r = -x
		sign = true
	}

	for r >= y {
		rfr, rexp := Frexp(r)
		if rfr < yfr {
			rexp = rexp - 1
		}
		r = r - Ldexp(y, rexp-yexp)
	}
	if sign {
		r = -r
	}
	return r
}
