// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package color

// RGBToYCbCr converts an RGB triple to a Y'CbCr triple.
func RGBToYCbCr(r, g, b uint8) (uint8, uint8, uint8) {
	// The JFIF specification says:
	//	Y' =  0.2990*R + 0.5870*G + 0.1140*B
	//	Cb = -0.1687*R - 0.3313*G + 0.5000*B + 128
	//	Cr =  0.5000*R - 0.4187*G - 0.0813*B + 128
	// http://www.w3.org/Graphics/JPEG/jfif3.pdf says Y but means Y'.

	r1 := int32(r)
	g1 := int32(g)
	b1 := int32(b)
	yy := (19595*r1 + 38470*g1 + 7471*b1 + 1<<15) >> 16
	cb := (-11056*r1 - 21712*g1 + 32768*b1 + 257<<15) >> 16
	cr := (32768*r1 - 27440*g1 - 5328*b1 + 257<<15) >> 16
	if yy < 0 {
		yy = 0
	} else if yy > 0xff {
		yy = 0xff
	}
	if cb < 0 {
		cb = 0
	} else if cb > 0xff {
		cb = 0xff
	}
	if cr < 0 {
		cr = 0
	} else if cr > 0xff {
		cr = 0xff
	}
	return uint8(yy), uint8(cb), uint8(cr)
}

// YCbCrToRGB converts a Y'CbCr triple to an RGB triple.
func YCbCrToRGB(y, cb, cr uint8) (uint8, uint8, uint8) {
	// The JFIF specification says:
	//	R = Y' + 1.40200*(Cr-128)
	//	G = Y' - 0.34414*(Cb-128) - 0.71414*(Cr-128)
	//	B = Y' + 1.77200*(Cb-128)
	// http://www.w3.org/Graphics/JPEG/jfif3.pdf says Y but means Y'.

	yy1 := int32(y) * 0x10100 // Convert 0x12 to 0x121200.
	cb1 := int32(cb) - 128
	cr1 := int32(cr) - 128
	r := (yy1 + 91881*cr1) >> 16
	g := (yy1 - 22554*cb1 - 46802*cr1) >> 16
	b := (yy1 + 116130*cb1) >> 16
	if r < 0 {
		r = 0
	} else if r > 0xff {
		r = 0xff
	}
	if g < 0 {
		g = 0
	} else if g > 0xff {
		g = 0xff
	}
	if b < 0 {
		b = 0
	} else if b > 0xff {
		b = 0xff
	}
	return uint8(r), uint8(g), uint8(b)
}

// YCbCr represents a fully opaque 24-bit Y'CbCr color, having 8 bits each for
// one luma and two chroma components.
//
// JPEG, VP8, the MPEG family and other codecs use this color model. Such
// codecs often use the terms YUV and Y'CbCr interchangeably, but strictly
// speaking, the term YUV applies only to analog video signals, and Y' (luma)
// is Y (luminance) after applying gamma correction.
//
// Conversion between RGB and Y'CbCr is lossy and there are multiple, slightly
// different formulae for converting between the two. This package follows
// the JFIF specification at http://www.w3.org/Graphics/JPEG/jfif3.pdf.
type YCbCr struct {
	Y, Cb, Cr uint8
}

func (c YCbCr) RGBA() (uint32, uint32, uint32, uint32) {
	// This code is a copy of the YCbCrToRGB function above, except that it
	// returns values in the range [0, 0xffff] instead of [0, 0xff]. There is a
	// subtle difference between doing this and having YCbCr satisfy the Color
	// interface by first converting to an RGBA. The latter loses some
	// information by going to and from 8 bits per channel.
	//
	// For example, this code:
	//	const y, cb, cr = 0x7f, 0x7f, 0x7f
	//	r, g, b := color.YCbCrToRGB(y, cb, cr)
	//	r0, g0, b0, _ := color.YCbCr{y, cb, cr}.RGBA()
	//	r1, g1, b1, _ := color.RGBA{r, g, b, 0xff}.RGBA()
	//	fmt.Printf("0x%04x 0x%04x 0x%04x\n", r0, g0, b0)
	//	fmt.Printf("0x%04x 0x%04x 0x%04x\n", r1, g1, b1)
	// prints:
	//	0x7e18 0x808e 0x7db9
	//	0x7e7e 0x8080 0x7d7d

	yy1 := int32(c.Y) * 0x10100 // Convert 0x12 to 0x121200.
	cb1 := int32(c.Cb) - 128
	cr1 := int32(c.Cr) - 128
	r := (yy1 + 91881*cr1) >> 8
	g := (yy1 - 22554*cb1 - 46802*cr1) >> 8
	b := (yy1 + 116130*cb1) >> 8
	if r < 0 {
		r = 0
	} else if r > 0xffff {
		r = 0xffff
	}
	if g < 0 {
		g = 0
	} else if g > 0xffff {
		g = 0xffff
	}
	if b < 0 {
		b = 0
	} else if b > 0xffff {
		b = 0xffff
	}
	return uint32(r), uint32(g), uint32(b), 0xffff
}

// YCbCrModel is the Model for Y'CbCr colors.
var YCbCrModel Model = ModelFunc(yCbCrModel)

func yCbCrModel(c Color) Color {
	if _, ok := c.(YCbCr); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	y, u, v := RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	return YCbCr{y, u, v}
}

// RGBToCMYK converts an RGB triple to a CMYK quadruple.
func RGBToCMYK(r, g, b uint8) (uint8, uint8, uint8, uint8) {
	rr := uint32(r)
	gg := uint32(g)
	bb := uint32(b)
	w := rr
	if w < gg {
		w = gg
	}
	if w < bb {
		w = bb
	}
	if w == 0 {
		return 0, 0, 0, 0xff
	}
	c := (w - rr) * 0xff / w
	m := (w - gg) * 0xff / w
	y := (w - bb) * 0xff / w
	return uint8(c), uint8(m), uint8(y), uint8(0xff - w)
}

// CMYKToRGB converts a CMYK quadruple to an RGB triple.
func CMYKToRGB(c, m, y, k uint8) (uint8, uint8, uint8) {
	w := uint32(0xffff - uint32(k)*0x101)
	r := uint32(0xffff-uint32(c)*0x101) * w / 0xffff
	g := uint32(0xffff-uint32(m)*0x101) * w / 0xffff
	b := uint32(0xffff-uint32(y)*0x101) * w / 0xffff
	return uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)
}

// CMYK represents a fully opaque CMYK color, having 8 bits for each of cyan,
// magenta, yellow and black.
//
// It is not associated with any particular color profile.
type CMYK struct {
	C, M, Y, K uint8
}

func (c CMYK) RGBA() (uint32, uint32, uint32, uint32) {
	// This code is a copy of the CMYKToRGB function above, except that it
	// returns values in the range [0, 0xffff] instead of [0, 0xff].

	w := uint32(0xffff - uint32(c.K)*0x101)
	r := uint32(0xffff-uint32(c.C)*0x101) * w / 0xffff
	g := uint32(0xffff-uint32(c.M)*0x101) * w / 0xffff
	b := uint32(0xffff-uint32(c.Y)*0x101) * w / 0xffff
	return uint32(r), uint32(g), uint32(b), 0xffff
}

// CMYKModel is the Model for CMYK colors.
var CMYKModel Model = ModelFunc(cmykModel)

func cmykModel(c Color) Color {
	if _, ok := c.(CMYK); ok {
		return c
	}
	r, g, b, _ := c.RGBA()
	cc, mm, yy, kk := RGBToCMYK(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	return CMYK{cc, mm, yy, kk}
}
