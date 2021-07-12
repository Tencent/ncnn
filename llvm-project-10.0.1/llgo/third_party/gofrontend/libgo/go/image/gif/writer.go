// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gif

import (
	"bufio"
	"bytes"
	"compress/lzw"
	"errors"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"io"
)

// Graphic control extension fields.
const (
	gcLabel     = 0xF9
	gcBlockSize = 0x04
)

var log2Lookup = [8]int{2, 4, 8, 16, 32, 64, 128, 256}

func log2(x int) int {
	for i, v := range log2Lookup {
		if x <= v {
			return i
		}
	}
	return -1
}

// Little-endian.
func writeUint16(b []uint8, u uint16) {
	b[0] = uint8(u)
	b[1] = uint8(u >> 8)
}

// writer is a buffered writer.
type writer interface {
	Flush() error
	io.Writer
	io.ByteWriter
}

// encoder encodes an image to the GIF format.
type encoder struct {
	// w is the writer to write to. err is the first error encountered during
	// writing. All attempted writes after the first error become no-ops.
	w   writer
	err error
	// g is a reference to the data that is being encoded.
	g GIF
	// globalCT is the size in bytes of the global color table.
	globalCT int
	// buf is a scratch buffer. It must be at least 256 for the blockWriter.
	buf              [256]byte
	globalColorTable [3 * 256]byte
	localColorTable  [3 * 256]byte
}

// blockWriter writes the block structure of GIF image data, which
// comprises (n, (n bytes)) blocks, with 1 <= n <= 255. It is the
// writer given to the LZW encoder, which is thus immune to the
// blocking.
type blockWriter struct {
	e *encoder
}

func (b blockWriter) Write(data []byte) (int, error) {
	if b.e.err != nil {
		return 0, b.e.err
	}
	if len(data) == 0 {
		return 0, nil
	}
	total := 0
	for total < len(data) {
		n := copy(b.e.buf[1:256], data[total:])
		total += n
		b.e.buf[0] = uint8(n)

		n, b.e.err = b.e.w.Write(b.e.buf[:n+1])
		if b.e.err != nil {
			return 0, b.e.err
		}
	}
	return total, b.e.err
}

func (e *encoder) flush() {
	if e.err != nil {
		return
	}
	e.err = e.w.Flush()
}

func (e *encoder) write(p []byte) {
	if e.err != nil {
		return
	}
	_, e.err = e.w.Write(p)
}

func (e *encoder) writeByte(b byte) {
	if e.err != nil {
		return
	}
	e.err = e.w.WriteByte(b)
}

func (e *encoder) writeHeader() {
	if e.err != nil {
		return
	}
	_, e.err = io.WriteString(e.w, "GIF89a")
	if e.err != nil {
		return
	}

	// Logical screen width and height.
	writeUint16(e.buf[0:2], uint16(e.g.Config.Width))
	writeUint16(e.buf[2:4], uint16(e.g.Config.Height))
	e.write(e.buf[:4])

	if p, ok := e.g.Config.ColorModel.(color.Palette); ok && len(p) > 0 {
		paddedSize := log2(len(p)) // Size of Global Color Table: 2^(1+n).
		e.buf[0] = fColorTable | uint8(paddedSize)
		e.buf[1] = e.g.BackgroundIndex
		e.buf[2] = 0x00 // Pixel Aspect Ratio.
		e.write(e.buf[:3])
		e.globalCT = encodeColorTable(e.globalColorTable[:], p, paddedSize)
		e.write(e.globalColorTable[:e.globalCT])
	} else {
		// All frames have a local color table, so a global color table
		// is not needed.
		e.buf[0] = 0x00
		e.buf[1] = 0x00 // Background Color Index.
		e.buf[2] = 0x00 // Pixel Aspect Ratio.
		e.write(e.buf[:3])
	}

	// Add animation info if necessary.
	if len(e.g.Image) > 1 {
		e.buf[0] = 0x21 // Extension Introducer.
		e.buf[1] = 0xff // Application Label.
		e.buf[2] = 0x0b // Block Size.
		e.write(e.buf[:3])
		_, e.err = io.WriteString(e.w, "NETSCAPE2.0") // Application Identifier.
		if e.err != nil {
			return
		}
		e.buf[0] = 0x03 // Block Size.
		e.buf[1] = 0x01 // Sub-block Index.
		writeUint16(e.buf[2:4], uint16(e.g.LoopCount))
		e.buf[4] = 0x00 // Block Terminator.
		e.write(e.buf[:5])
	}
}

func encodeColorTable(dst []byte, p color.Palette, size int) int {
	n := log2Lookup[size]
	for i := 0; i < n; i++ {
		if i < len(p) {
			r, g, b, _ := p[i].RGBA()
			dst[3*i+0] = uint8(r >> 8)
			dst[3*i+1] = uint8(g >> 8)
			dst[3*i+2] = uint8(b >> 8)
		} else {
			// Pad with black.
			dst[3*i+0] = 0x00
			dst[3*i+1] = 0x00
			dst[3*i+2] = 0x00
		}
	}
	return 3 * n
}

func (e *encoder) writeImageBlock(pm *image.Paletted, delay int, disposal byte) {
	if e.err != nil {
		return
	}

	if len(pm.Palette) == 0 {
		e.err = errors.New("gif: cannot encode image block with empty palette")
		return
	}

	b := pm.Bounds()
	if b.Min.X < 0 || b.Max.X >= 1<<16 || b.Min.Y < 0 || b.Max.Y >= 1<<16 {
		e.err = errors.New("gif: image block is too large to encode")
		return
	}
	if !b.In(image.Rectangle{Max: image.Point{e.g.Config.Width, e.g.Config.Height}}) {
		e.err = errors.New("gif: image block is out of bounds")
		return
	}

	transparentIndex := -1
	for i, c := range pm.Palette {
		if _, _, _, a := c.RGBA(); a == 0 {
			transparentIndex = i
			break
		}
	}

	if delay > 0 || disposal != 0 || transparentIndex != -1 {
		e.buf[0] = sExtension  // Extension Introducer.
		e.buf[1] = gcLabel     // Graphic Control Label.
		e.buf[2] = gcBlockSize // Block Size.
		if transparentIndex != -1 {
			e.buf[3] = 0x01 | disposal<<2
		} else {
			e.buf[3] = 0x00 | disposal<<2
		}
		writeUint16(e.buf[4:6], uint16(delay)) // Delay Time (1/100ths of a second)

		// Transparent color index.
		if transparentIndex != -1 {
			e.buf[6] = uint8(transparentIndex)
		} else {
			e.buf[6] = 0x00
		}
		e.buf[7] = 0x00 // Block Terminator.
		e.write(e.buf[:8])
	}
	e.buf[0] = sImageDescriptor
	writeUint16(e.buf[1:3], uint16(b.Min.X))
	writeUint16(e.buf[3:5], uint16(b.Min.Y))
	writeUint16(e.buf[5:7], uint16(b.Dx()))
	writeUint16(e.buf[7:9], uint16(b.Dy()))
	e.write(e.buf[:9])

	paddedSize := log2(len(pm.Palette)) // Size of Local Color Table: 2^(1+n).
	ct := encodeColorTable(e.localColorTable[:], pm.Palette, paddedSize)
	if ct != e.globalCT || !bytes.Equal(e.globalColorTable[:ct], e.localColorTable[:ct]) {
		// Use a local color table.
		e.writeByte(fColorTable | uint8(paddedSize))
		e.write(e.localColorTable[:ct])
	} else {
		// Use the global color table.
		e.writeByte(0)
	}

	litWidth := paddedSize + 1
	if litWidth < 2 {
		litWidth = 2
	}
	e.writeByte(uint8(litWidth)) // LZW Minimum Code Size.

	lzww := lzw.NewWriter(blockWriter{e: e}, lzw.LSB, litWidth)
	if dx := b.Dx(); dx == pm.Stride {
		_, e.err = lzww.Write(pm.Pix)
		if e.err != nil {
			lzww.Close()
			return
		}
	} else {
		for i, y := 0, b.Min.Y; y < b.Max.Y; i, y = i+pm.Stride, y+1 {
			_, e.err = lzww.Write(pm.Pix[i : i+dx])
			if e.err != nil {
				lzww.Close()
				return
			}
		}
	}
	lzww.Close()
	e.writeByte(0x00) // Block Terminator.
}

// Options are the encoding parameters.
type Options struct {
	// NumColors is the maximum number of colors used in the image.
	// It ranges from 1 to 256.
	NumColors int

	// Quantizer is used to produce a palette with size NumColors.
	// palette.Plan9 is used in place of a nil Quantizer.
	Quantizer draw.Quantizer

	// Drawer is used to convert the source image to the desired palette.
	// draw.FloydSteinberg is used in place of a nil Drawer.
	Drawer draw.Drawer
}

// EncodeAll writes the images in g to w in GIF format with the
// given loop count and delay between frames.
func EncodeAll(w io.Writer, g *GIF) error {
	if len(g.Image) == 0 {
		return errors.New("gif: must provide at least one image")
	}

	if len(g.Image) != len(g.Delay) {
		return errors.New("gif: mismatched image and delay lengths")
	}
	if g.LoopCount < 0 {
		g.LoopCount = 0
	}

	e := encoder{g: *g}
	// The GIF.Disposal, GIF.Config and GIF.BackgroundIndex fields were added
	// in Go 1.5. Valid Go 1.4 code, such as when the Disposal field is omitted
	// in a GIF struct literal, should still produce valid GIFs.
	if e.g.Disposal != nil && len(e.g.Image) != len(e.g.Disposal) {
		return errors.New("gif: mismatched image and disposal lengths")
	}
	if e.g.Config == (image.Config{}) {
		p := g.Image[0].Bounds().Max
		e.g.Config.Width = p.X
		e.g.Config.Height = p.Y
	} else if e.g.Config.ColorModel != nil {
		if _, ok := e.g.Config.ColorModel.(color.Palette); !ok {
			return errors.New("gif: GIF color model must be a color.Palette")
		}
	}

	if ww, ok := w.(writer); ok {
		e.w = ww
	} else {
		e.w = bufio.NewWriter(w)
	}

	e.writeHeader()
	for i, pm := range g.Image {
		disposal := uint8(0)
		if g.Disposal != nil {
			disposal = g.Disposal[i]
		}
		e.writeImageBlock(pm, g.Delay[i], disposal)
	}
	e.writeByte(sTrailer)
	e.flush()
	return e.err
}

// Encode writes the Image m to w in GIF format.
func Encode(w io.Writer, m image.Image, o *Options) error {
	// Check for bounds and size restrictions.
	b := m.Bounds()
	if b.Dx() >= 1<<16 || b.Dy() >= 1<<16 {
		return errors.New("gif: image is too large to encode")
	}

	opts := Options{}
	if o != nil {
		opts = *o
	}
	if opts.NumColors < 1 || 256 < opts.NumColors {
		opts.NumColors = 256
	}
	if opts.Drawer == nil {
		opts.Drawer = draw.FloydSteinberg
	}

	pm, ok := m.(*image.Paletted)
	if !ok || len(pm.Palette) > opts.NumColors {
		// TODO: Pick a better sub-sample of the Plan 9 palette.
		pm = image.NewPaletted(b, palette.Plan9[:opts.NumColors])
		if opts.Quantizer != nil {
			pm.Palette = opts.Quantizer.Quantize(make(color.Palette, 0, opts.NumColors), m)
		}
		opts.Drawer.Draw(pm, b, m, image.ZP)
	}

	// When calling Encode instead of EncodeAll, the single-frame image is
	// translated such that its top-left corner is (0, 0), so that the single
	// frame completely fills the overall GIF's bounds.
	if pm.Rect.Min != (image.Point{}) {
		dup := *pm
		dup.Rect = dup.Rect.Sub(dup.Rect.Min)
		pm = &dup
	}

	return EncodeAll(w, &GIF{
		Image: []*image.Paletted{pm},
		Delay: []int{0},
		Config: image.Config{
			ColorModel: pm.Palette,
			Width:      b.Dx(),
			Height:     b.Dy(),
		},
	})
}
