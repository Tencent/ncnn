// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bufio"
	"compress/zlib"
	"hash/crc32"
	"image"
	"image/color"
	"io"
	"strconv"
)

// Encoder configures encoding PNG images.
type Encoder struct {
	CompressionLevel CompressionLevel
}

type encoder struct {
	enc    *Encoder
	w      io.Writer
	m      image.Image
	cb     int
	err    error
	header [8]byte
	footer [4]byte
	tmp    [4 * 256]byte
}

type CompressionLevel int

const (
	DefaultCompression CompressionLevel = 0
	NoCompression      CompressionLevel = -1
	BestSpeed          CompressionLevel = -2
	BestCompression    CompressionLevel = -3

	// Positive CompressionLevel values are reserved to mean a numeric zlib
	// compression level, although that is not implemented yet.
)

// Big-endian.
func writeUint32(b []uint8, u uint32) {
	b[0] = uint8(u >> 24)
	b[1] = uint8(u >> 16)
	b[2] = uint8(u >> 8)
	b[3] = uint8(u >> 0)
}

type opaquer interface {
	Opaque() bool
}

// Returns whether or not the image is fully opaque.
func opaque(m image.Image) bool {
	if o, ok := m.(opaquer); ok {
		return o.Opaque()
	}
	b := m.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			_, _, _, a := m.At(x, y).RGBA()
			if a != 0xffff {
				return false
			}
		}
	}
	return true
}

// The absolute value of a byte interpreted as a signed int8.
func abs8(d uint8) int {
	if d < 128 {
		return int(d)
	}
	return 256 - int(d)
}

func (e *encoder) writeChunk(b []byte, name string) {
	if e.err != nil {
		return
	}
	n := uint32(len(b))
	if int(n) != len(b) {
		e.err = UnsupportedError(name + " chunk is too large: " + strconv.Itoa(len(b)))
		return
	}
	writeUint32(e.header[:4], n)
	e.header[4] = name[0]
	e.header[5] = name[1]
	e.header[6] = name[2]
	e.header[7] = name[3]
	crc := crc32.NewIEEE()
	crc.Write(e.header[4:8])
	crc.Write(b)
	writeUint32(e.footer[:4], crc.Sum32())

	_, e.err = e.w.Write(e.header[:8])
	if e.err != nil {
		return
	}
	_, e.err = e.w.Write(b)
	if e.err != nil {
		return
	}
	_, e.err = e.w.Write(e.footer[:4])
}

func (e *encoder) writeIHDR() {
	b := e.m.Bounds()
	writeUint32(e.tmp[0:4], uint32(b.Dx()))
	writeUint32(e.tmp[4:8], uint32(b.Dy()))
	// Set bit depth and color type.
	switch e.cb {
	case cbG8:
		e.tmp[8] = 8
		e.tmp[9] = ctGrayscale
	case cbTC8:
		e.tmp[8] = 8
		e.tmp[9] = ctTrueColor
	case cbP8:
		e.tmp[8] = 8
		e.tmp[9] = ctPaletted
	case cbTCA8:
		e.tmp[8] = 8
		e.tmp[9] = ctTrueColorAlpha
	case cbG16:
		e.tmp[8] = 16
		e.tmp[9] = ctGrayscale
	case cbTC16:
		e.tmp[8] = 16
		e.tmp[9] = ctTrueColor
	case cbTCA16:
		e.tmp[8] = 16
		e.tmp[9] = ctTrueColorAlpha
	}
	e.tmp[10] = 0 // default compression method
	e.tmp[11] = 0 // default filter method
	e.tmp[12] = 0 // non-interlaced
	e.writeChunk(e.tmp[:13], "IHDR")
}

func (e *encoder) writePLTEAndTRNS(p color.Palette) {
	if len(p) < 1 || len(p) > 256 {
		e.err = FormatError("bad palette length: " + strconv.Itoa(len(p)))
		return
	}
	last := -1
	for i, c := range p {
		c1 := color.NRGBAModel.Convert(c).(color.NRGBA)
		e.tmp[3*i+0] = c1.R
		e.tmp[3*i+1] = c1.G
		e.tmp[3*i+2] = c1.B
		if c1.A != 0xff {
			last = i
		}
		e.tmp[3*256+i] = c1.A
	}
	e.writeChunk(e.tmp[:3*len(p)], "PLTE")
	if last != -1 {
		e.writeChunk(e.tmp[3*256:3*256+1+last], "tRNS")
	}
}

// An encoder is an io.Writer that satisfies writes by writing PNG IDAT chunks,
// including an 8-byte header and 4-byte CRC checksum per Write call. Such calls
// should be relatively infrequent, since writeIDATs uses a bufio.Writer.
//
// This method should only be called from writeIDATs (via writeImage).
// No other code should treat an encoder as an io.Writer.
func (e *encoder) Write(b []byte) (int, error) {
	e.writeChunk(b, "IDAT")
	if e.err != nil {
		return 0, e.err
	}
	return len(b), nil
}

// Chooses the filter to use for encoding the current row, and applies it.
// The return value is the index of the filter and also of the row in cr that has had it applied.
func filter(cr *[nFilter][]byte, pr []byte, bpp int) int {
	// We try all five filter types, and pick the one that minimizes the sum of absolute differences.
	// This is the same heuristic that libpng uses, although the filters are attempted in order of
	// estimated most likely to be minimal (ftUp, ftPaeth, ftNone, ftSub, ftAverage), rather than
	// in their enumeration order (ftNone, ftSub, ftUp, ftAverage, ftPaeth).
	cdat0 := cr[0][1:]
	cdat1 := cr[1][1:]
	cdat2 := cr[2][1:]
	cdat3 := cr[3][1:]
	cdat4 := cr[4][1:]
	pdat := pr[1:]
	n := len(cdat0)

	// The up filter.
	sum := 0
	for i := 0; i < n; i++ {
		cdat2[i] = cdat0[i] - pdat[i]
		sum += abs8(cdat2[i])
	}
	best := sum
	filter := ftUp

	// The Paeth filter.
	sum = 0
	for i := 0; i < bpp; i++ {
		cdat4[i] = cdat0[i] - pdat[i]
		sum += abs8(cdat4[i])
	}
	for i := bpp; i < n; i++ {
		cdat4[i] = cdat0[i] - paeth(cdat0[i-bpp], pdat[i], pdat[i-bpp])
		sum += abs8(cdat4[i])
		if sum >= best {
			break
		}
	}
	if sum < best {
		best = sum
		filter = ftPaeth
	}

	// The none filter.
	sum = 0
	for i := 0; i < n; i++ {
		sum += abs8(cdat0[i])
		if sum >= best {
			break
		}
	}
	if sum < best {
		best = sum
		filter = ftNone
	}

	// The sub filter.
	sum = 0
	for i := 0; i < bpp; i++ {
		cdat1[i] = cdat0[i]
		sum += abs8(cdat1[i])
	}
	for i := bpp; i < n; i++ {
		cdat1[i] = cdat0[i] - cdat0[i-bpp]
		sum += abs8(cdat1[i])
		if sum >= best {
			break
		}
	}
	if sum < best {
		best = sum
		filter = ftSub
	}

	// The average filter.
	sum = 0
	for i := 0; i < bpp; i++ {
		cdat3[i] = cdat0[i] - pdat[i]/2
		sum += abs8(cdat3[i])
	}
	for i := bpp; i < n; i++ {
		cdat3[i] = cdat0[i] - uint8((int(cdat0[i-bpp])+int(pdat[i]))/2)
		sum += abs8(cdat3[i])
		if sum >= best {
			break
		}
	}
	if sum < best {
		best = sum
		filter = ftAverage
	}

	return filter
}

func writeImage(w io.Writer, m image.Image, cb int, level int) error {
	zw, err := zlib.NewWriterLevel(w, level)
	if err != nil {
		return err
	}
	defer zw.Close()

	bpp := 0 // Bytes per pixel.

	switch cb {
	case cbG8:
		bpp = 1
	case cbTC8:
		bpp = 3
	case cbP8:
		bpp = 1
	case cbTCA8:
		bpp = 4
	case cbTC16:
		bpp = 6
	case cbTCA16:
		bpp = 8
	case cbG16:
		bpp = 2
	}
	// cr[*] and pr are the bytes for the current and previous row.
	// cr[0] is unfiltered (or equivalently, filtered with the ftNone filter).
	// cr[ft], for non-zero filter types ft, are buffers for transforming cr[0] under the
	// other PNG filter types. These buffers are allocated once and re-used for each row.
	// The +1 is for the per-row filter type, which is at cr[*][0].
	b := m.Bounds()
	var cr [nFilter][]uint8
	for i := range cr {
		cr[i] = make([]uint8, 1+bpp*b.Dx())
		cr[i][0] = uint8(i)
	}
	pr := make([]uint8, 1+bpp*b.Dx())

	gray, _ := m.(*image.Gray)
	rgba, _ := m.(*image.RGBA)
	paletted, _ := m.(*image.Paletted)
	nrgba, _ := m.(*image.NRGBA)

	for y := b.Min.Y; y < b.Max.Y; y++ {
		// Convert from colors to bytes.
		i := 1
		switch cb {
		case cbG8:
			if gray != nil {
				offset := (y - b.Min.Y) * gray.Stride
				copy(cr[0][1:], gray.Pix[offset:offset+b.Dx()])
			} else {
				for x := b.Min.X; x < b.Max.X; x++ {
					c := color.GrayModel.Convert(m.At(x, y)).(color.Gray)
					cr[0][i] = c.Y
					i++
				}
			}
		case cbTC8:
			// We have previously verified that the alpha value is fully opaque.
			cr0 := cr[0]
			stride, pix := 0, []byte(nil)
			if rgba != nil {
				stride, pix = rgba.Stride, rgba.Pix
			} else if nrgba != nil {
				stride, pix = nrgba.Stride, nrgba.Pix
			}
			if stride != 0 {
				j0 := (y - b.Min.Y) * stride
				j1 := j0 + b.Dx()*4
				for j := j0; j < j1; j += 4 {
					cr0[i+0] = pix[j+0]
					cr0[i+1] = pix[j+1]
					cr0[i+2] = pix[j+2]
					i += 3
				}
			} else {
				for x := b.Min.X; x < b.Max.X; x++ {
					r, g, b, _ := m.At(x, y).RGBA()
					cr0[i+0] = uint8(r >> 8)
					cr0[i+1] = uint8(g >> 8)
					cr0[i+2] = uint8(b >> 8)
					i += 3
				}
			}
		case cbP8:
			if paletted != nil {
				offset := (y - b.Min.Y) * paletted.Stride
				copy(cr[0][1:], paletted.Pix[offset:offset+b.Dx()])
			} else {
				pi := m.(image.PalettedImage)
				for x := b.Min.X; x < b.Max.X; x++ {
					cr[0][i] = pi.ColorIndexAt(x, y)
					i += 1
				}
			}
		case cbTCA8:
			if nrgba != nil {
				offset := (y - b.Min.Y) * nrgba.Stride
				copy(cr[0][1:], nrgba.Pix[offset:offset+b.Dx()*4])
			} else {
				// Convert from image.Image (which is alpha-premultiplied) to PNG's non-alpha-premultiplied.
				for x := b.Min.X; x < b.Max.X; x++ {
					c := color.NRGBAModel.Convert(m.At(x, y)).(color.NRGBA)
					cr[0][i+0] = c.R
					cr[0][i+1] = c.G
					cr[0][i+2] = c.B
					cr[0][i+3] = c.A
					i += 4
				}
			}
		case cbG16:
			for x := b.Min.X; x < b.Max.X; x++ {
				c := color.Gray16Model.Convert(m.At(x, y)).(color.Gray16)
				cr[0][i+0] = uint8(c.Y >> 8)
				cr[0][i+1] = uint8(c.Y)
				i += 2
			}
		case cbTC16:
			// We have previously verified that the alpha value is fully opaque.
			for x := b.Min.X; x < b.Max.X; x++ {
				r, g, b, _ := m.At(x, y).RGBA()
				cr[0][i+0] = uint8(r >> 8)
				cr[0][i+1] = uint8(r)
				cr[0][i+2] = uint8(g >> 8)
				cr[0][i+3] = uint8(g)
				cr[0][i+4] = uint8(b >> 8)
				cr[0][i+5] = uint8(b)
				i += 6
			}
		case cbTCA16:
			// Convert from image.Image (which is alpha-premultiplied) to PNG's non-alpha-premultiplied.
			for x := b.Min.X; x < b.Max.X; x++ {
				c := color.NRGBA64Model.Convert(m.At(x, y)).(color.NRGBA64)
				cr[0][i+0] = uint8(c.R >> 8)
				cr[0][i+1] = uint8(c.R)
				cr[0][i+2] = uint8(c.G >> 8)
				cr[0][i+3] = uint8(c.G)
				cr[0][i+4] = uint8(c.B >> 8)
				cr[0][i+5] = uint8(c.B)
				cr[0][i+6] = uint8(c.A >> 8)
				cr[0][i+7] = uint8(c.A)
				i += 8
			}
		}

		// Apply the filter.
		f := ftNone
		if level != zlib.NoCompression {
			f = filter(&cr, pr, bpp)
		}

		// Write the compressed bytes.
		if _, err := zw.Write(cr[f]); err != nil {
			return err
		}

		// The current row for y is the previous row for y+1.
		pr, cr[0] = cr[0], pr
	}
	return nil
}

// Write the actual image data to one or more IDAT chunks.
func (e *encoder) writeIDATs() {
	if e.err != nil {
		return
	}
	var bw *bufio.Writer
	bw = bufio.NewWriterSize(e, 1<<15)
	e.err = writeImage(bw, e.m, e.cb, levelToZlib(e.enc.CompressionLevel))
	if e.err != nil {
		return
	}
	e.err = bw.Flush()
}

// This function is required because we want the zero value of
// Encoder.CompressionLevel to map to zlib.DefaultCompression.
func levelToZlib(l CompressionLevel) int {
	switch l {
	case DefaultCompression:
		return zlib.DefaultCompression
	case NoCompression:
		return zlib.NoCompression
	case BestSpeed:
		return zlib.BestSpeed
	case BestCompression:
		return zlib.BestCompression
	default:
		return zlib.DefaultCompression
	}
}

func (e *encoder) writeIEND() { e.writeChunk(nil, "IEND") }

// Encode writes the Image m to w in PNG format. Any Image may be
// encoded, but images that are not image.NRGBA might be encoded lossily.
func Encode(w io.Writer, m image.Image) error {
	var e Encoder
	return e.Encode(w, m)
}

// Encode writes the Image m to w in PNG format.
func (enc *Encoder) Encode(w io.Writer, m image.Image) error {
	// Obviously, negative widths and heights are invalid. Furthermore, the PNG
	// spec section 11.2.2 says that zero is invalid. Excessively large images are
	// also rejected.
	mw, mh := int64(m.Bounds().Dx()), int64(m.Bounds().Dy())
	if mw <= 0 || mh <= 0 || mw >= 1<<32 || mh >= 1<<32 {
		return FormatError("invalid image size: " + strconv.FormatInt(mw, 10) + "x" + strconv.FormatInt(mh, 10))
	}

	var e encoder
	e.enc = enc
	e.w = w
	e.m = m

	var pal color.Palette
	// cbP8 encoding needs PalettedImage's ColorIndexAt method.
	if _, ok := m.(image.PalettedImage); ok {
		pal, _ = m.ColorModel().(color.Palette)
	}
	if pal != nil {
		e.cb = cbP8
	} else {
		switch m.ColorModel() {
		case color.GrayModel:
			e.cb = cbG8
		case color.Gray16Model:
			e.cb = cbG16
		case color.RGBAModel, color.NRGBAModel, color.AlphaModel:
			if opaque(m) {
				e.cb = cbTC8
			} else {
				e.cb = cbTCA8
			}
		default:
			if opaque(m) {
				e.cb = cbTC16
			} else {
				e.cb = cbTCA16
			}
		}
	}

	_, e.err = io.WriteString(w, pngHeader)
	e.writeIHDR()
	if pal != nil {
		e.writePLTEAndTRNS(pal)
	}
	e.writeIDATs()
	e.writeIEND()
	return e.err
}
