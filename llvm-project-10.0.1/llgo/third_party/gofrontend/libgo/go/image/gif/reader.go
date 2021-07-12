// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gif implements a GIF image decoder and encoder.
//
// The GIF specification is at http://www.w3.org/Graphics/GIF/spec-gif89a.txt.
package gif

import (
	"bufio"
	"compress/lzw"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
)

var (
	errNotEnough = errors.New("gif: not enough image data")
	errTooMuch   = errors.New("gif: too much image data")
	errBadPixel  = errors.New("gif: invalid pixel value")
)

// If the io.Reader does not also have ReadByte, then decode will introduce its own buffering.
type reader interface {
	io.Reader
	io.ByteReader
}

// Masks etc.
const (
	// Fields.
	fColorTable         = 1 << 7
	fInterlace          = 1 << 6
	fColorTableBitsMask = 7

	// Graphic control flags.
	gcTransparentColorSet = 1 << 0
	gcDisposalMethodMask  = 7 << 2
)

// Disposal Methods.
const (
	DisposalNone       = 0x01
	DisposalBackground = 0x02
	DisposalPrevious   = 0x03
)

// Section indicators.
const (
	sExtension       = 0x21
	sImageDescriptor = 0x2C
	sTrailer         = 0x3B
)

// Extensions.
const (
	eText           = 0x01 // Plain Text
	eGraphicControl = 0xF9 // Graphic Control
	eComment        = 0xFE // Comment
	eApplication    = 0xFF // Application
)

// decoder is the type used to decode a GIF file.
type decoder struct {
	r reader

	// From header.
	vers            string
	width           int
	height          int
	loopCount       int
	delayTime       int
	backgroundIndex byte
	disposalMethod  byte

	// From image descriptor.
	imageFields byte

	// From graphics control.
	transparentIndex    byte
	hasTransparentIndex bool

	// Computed.
	globalColorTable color.Palette

	// Used when decoding.
	delay    []int
	disposal []byte
	image    []*image.Paletted
	tmp      [1024]byte // must be at least 768 so we can read color table
}

// blockReader parses the block structure of GIF image data, which
// comprises (n, (n bytes)) blocks, with 1 <= n <= 255.  It is the
// reader given to the LZW decoder, which is thus immune to the
// blocking.  After the LZW decoder completes, there will be a 0-byte
// block remaining (0, ()), which is consumed when checking that the
// blockReader is exhausted.
type blockReader struct {
	r     reader
	slice []byte
	err   error
	tmp   [256]byte
}

func (b *blockReader) Read(p []byte) (int, error) {
	if b.err != nil {
		return 0, b.err
	}
	if len(p) == 0 {
		return 0, nil
	}
	if len(b.slice) == 0 {
		var blockLen uint8
		blockLen, b.err = b.r.ReadByte()
		if b.err != nil {
			return 0, b.err
		}
		if blockLen == 0 {
			b.err = io.EOF
			return 0, b.err
		}
		b.slice = b.tmp[:blockLen]
		if _, b.err = io.ReadFull(b.r, b.slice); b.err != nil {
			return 0, b.err
		}
	}
	n := copy(p, b.slice)
	b.slice = b.slice[n:]
	return n, nil
}

// decode reads a GIF image from r and stores the result in d.
func (d *decoder) decode(r io.Reader, configOnly bool) error {
	// Add buffering if r does not provide ReadByte.
	if rr, ok := r.(reader); ok {
		d.r = rr
	} else {
		d.r = bufio.NewReader(r)
	}

	err := d.readHeaderAndScreenDescriptor()
	if err != nil {
		return err
	}
	if configOnly {
		return nil
	}

	for {
		c, err := d.r.ReadByte()
		if err != nil {
			return err
		}
		switch c {
		case sExtension:
			if err = d.readExtension(); err != nil {
				return err
			}

		case sImageDescriptor:
			m, err := d.newImageFromDescriptor()
			if err != nil {
				return err
			}
			useLocalColorTable := d.imageFields&fColorTable != 0
			if useLocalColorTable {
				m.Palette, err = d.readColorTable(d.imageFields)
				if err != nil {
					return err
				}
			} else {
				if d.globalColorTable == nil {
					return errors.New("gif: no color table")
				}
				m.Palette = d.globalColorTable
			}
			if d.hasTransparentIndex && int(d.transparentIndex) < len(m.Palette) {
				if !useLocalColorTable {
					// Clone the global color table.
					m.Palette = append(color.Palette(nil), d.globalColorTable...)
				}
				m.Palette[d.transparentIndex] = color.RGBA{}
			}
			litWidth, err := d.r.ReadByte()
			if err != nil {
				return err
			}
			if litWidth < 2 || litWidth > 8 {
				return fmt.Errorf("gif: pixel size in decode out of range: %d", litWidth)
			}
			// A wonderfully Go-like piece of magic.
			br := &blockReader{r: d.r}
			lzwr := lzw.NewReader(br, lzw.LSB, int(litWidth))
			defer lzwr.Close()
			if _, err = io.ReadFull(lzwr, m.Pix); err != nil {
				if err != io.ErrUnexpectedEOF {
					return err
				}
				return errNotEnough
			}
			// Both lzwr and br should be exhausted. Reading from them should
			// yield (0, io.EOF).
			//
			// The spec (Appendix F - Compression), says that "An End of
			// Information code... must be the last code output by the encoder
			// for an image". In practice, though, giflib (a widely used C
			// library) does not enforce this, so we also accept lzwr returning
			// io.ErrUnexpectedEOF (meaning that the encoded stream hit io.EOF
			// before the LZW decoder saw an explict end code), provided that
			// the io.ReadFull call above successfully read len(m.Pix) bytes.
			// See https://golang.org/issue/9856 for an example GIF.
			if n, err := lzwr.Read(d.tmp[:1]); n != 0 || (err != io.EOF && err != io.ErrUnexpectedEOF) {
				if err != nil {
					return err
				}
				return errTooMuch
			}
			if n, err := br.Read(d.tmp[:1]); n != 0 || err != io.EOF {
				if err != nil {
					return err
				}
				return errTooMuch
			}

			// Check that the color indexes are inside the palette.
			if len(m.Palette) < 256 {
				for _, pixel := range m.Pix {
					if int(pixel) >= len(m.Palette) {
						return errBadPixel
					}
				}
			}

			// Undo the interlacing if necessary.
			if d.imageFields&fInterlace != 0 {
				uninterlace(m)
			}

			d.image = append(d.image, m)
			d.delay = append(d.delay, d.delayTime)
			d.disposal = append(d.disposal, d.disposalMethod)
			// The GIF89a spec, Section 23 (Graphic Control Extension) says:
			// "The scope of this extension is the first graphic rendering block
			// to follow." We therefore reset the GCE fields to zero.
			d.delayTime = 0
			d.hasTransparentIndex = false

		case sTrailer:
			if len(d.image) == 0 {
				return io.ErrUnexpectedEOF
			}
			return nil

		default:
			return fmt.Errorf("gif: unknown block type: 0x%.2x", c)
		}
	}
}

func (d *decoder) readHeaderAndScreenDescriptor() error {
	_, err := io.ReadFull(d.r, d.tmp[:13])
	if err != nil {
		return err
	}
	d.vers = string(d.tmp[:6])
	if d.vers != "GIF87a" && d.vers != "GIF89a" {
		return fmt.Errorf("gif: can't recognize format %s", d.vers)
	}
	d.width = int(d.tmp[6]) + int(d.tmp[7])<<8
	d.height = int(d.tmp[8]) + int(d.tmp[9])<<8
	if fields := d.tmp[10]; fields&fColorTable != 0 {
		d.backgroundIndex = d.tmp[11]
		// readColorTable overwrites the contents of d.tmp, but that's OK.
		if d.globalColorTable, err = d.readColorTable(fields); err != nil {
			return err
		}
	}
	// d.tmp[12] is the Pixel Aspect Ratio, which is ignored.
	return nil
}

func (d *decoder) readColorTable(fields byte) (color.Palette, error) {
	n := 1 << (1 + uint(fields&fColorTableBitsMask))
	_, err := io.ReadFull(d.r, d.tmp[:3*n])
	if err != nil {
		return nil, fmt.Errorf("gif: short read on color table: %s", err)
	}
	j, p := 0, make(color.Palette, n)
	for i := range p {
		p[i] = color.RGBA{d.tmp[j+0], d.tmp[j+1], d.tmp[j+2], 0xFF}
		j += 3
	}
	return p, nil
}

func (d *decoder) readExtension() error {
	extension, err := d.r.ReadByte()
	if err != nil {
		return err
	}
	size := 0
	switch extension {
	case eText:
		size = 13
	case eGraphicControl:
		return d.readGraphicControl()
	case eComment:
		// nothing to do but read the data.
	case eApplication:
		b, err := d.r.ReadByte()
		if err != nil {
			return err
		}
		// The spec requires size be 11, but Adobe sometimes uses 10.
		size = int(b)
	default:
		return fmt.Errorf("gif: unknown extension 0x%.2x", extension)
	}
	if size > 0 {
		if _, err := io.ReadFull(d.r, d.tmp[:size]); err != nil {
			return err
		}
	}

	// Application Extension with "NETSCAPE2.0" as string and 1 in data means
	// this extension defines a loop count.
	if extension == eApplication && string(d.tmp[:size]) == "NETSCAPE2.0" {
		n, err := d.readBlock()
		if n == 0 || err != nil {
			return err
		}
		if n == 3 && d.tmp[0] == 1 {
			d.loopCount = int(d.tmp[1]) | int(d.tmp[2])<<8
		}
	}
	for {
		n, err := d.readBlock()
		if n == 0 || err != nil {
			return err
		}
	}
}

func (d *decoder) readGraphicControl() error {
	if _, err := io.ReadFull(d.r, d.tmp[:6]); err != nil {
		return fmt.Errorf("gif: can't read graphic control: %s", err)
	}
	flags := d.tmp[1]
	d.disposalMethod = (flags & gcDisposalMethodMask) >> 2
	d.delayTime = int(d.tmp[2]) | int(d.tmp[3])<<8
	if flags&gcTransparentColorSet != 0 {
		d.transparentIndex = d.tmp[4]
		d.hasTransparentIndex = true
	}
	return nil
}

func (d *decoder) newImageFromDescriptor() (*image.Paletted, error) {
	if _, err := io.ReadFull(d.r, d.tmp[:9]); err != nil {
		return nil, fmt.Errorf("gif: can't read image descriptor: %s", err)
	}
	left := int(d.tmp[0]) + int(d.tmp[1])<<8
	top := int(d.tmp[2]) + int(d.tmp[3])<<8
	width := int(d.tmp[4]) + int(d.tmp[5])<<8
	height := int(d.tmp[6]) + int(d.tmp[7])<<8
	d.imageFields = d.tmp[8]

	// The GIF89a spec, Section 20 (Image Descriptor) says:
	// "Each image must fit within the boundaries of the Logical
	// Screen, as defined in the Logical Screen Descriptor."
	bounds := image.Rect(left, top, left+width, top+height)
	if bounds != bounds.Intersect(image.Rect(0, 0, d.width, d.height)) {
		return nil, errors.New("gif: frame bounds larger than image bounds")
	}
	return image.NewPaletted(bounds, nil), nil
}

func (d *decoder) readBlock() (int, error) {
	n, err := d.r.ReadByte()
	if n == 0 || err != nil {
		return 0, err
	}
	return io.ReadFull(d.r, d.tmp[:n])
}

// interlaceScan defines the ordering for a pass of the interlace algorithm.
type interlaceScan struct {
	skip, start int
}

// interlacing represents the set of scans in an interlaced GIF image.
var interlacing = []interlaceScan{
	{8, 0}, // Group 1 : Every 8th. row, starting with row 0.
	{8, 4}, // Group 2 : Every 8th. row, starting with row 4.
	{4, 2}, // Group 3 : Every 4th. row, starting with row 2.
	{2, 1}, // Group 4 : Every 2nd. row, starting with row 1.
}

// uninterlace rearranges the pixels in m to account for interlaced input.
func uninterlace(m *image.Paletted) {
	var nPix []uint8
	dx := m.Bounds().Dx()
	dy := m.Bounds().Dy()
	nPix = make([]uint8, dx*dy)
	offset := 0 // steps through the input by sequential scan lines.
	for _, pass := range interlacing {
		nOffset := pass.start * dx // steps through the output as defined by pass.
		for y := pass.start; y < dy; y += pass.skip {
			copy(nPix[nOffset:nOffset+dx], m.Pix[offset:offset+dx])
			offset += dx
			nOffset += dx * pass.skip
		}
	}
	m.Pix = nPix
}

// Decode reads a GIF image from r and returns the first embedded
// image as an image.Image.
func Decode(r io.Reader) (image.Image, error) {
	var d decoder
	if err := d.decode(r, false); err != nil {
		return nil, err
	}
	return d.image[0], nil
}

// GIF represents the possibly multiple images stored in a GIF file.
type GIF struct {
	Image     []*image.Paletted // The successive images.
	Delay     []int             // The successive delay times, one per frame, in 100ths of a second.
	LoopCount int               // The loop count.
	// Disposal is the successive disposal methods, one per frame. For
	// backwards compatibility, a nil Disposal is valid to pass to EncodeAll,
	// and implies that each frame's disposal method is 0 (no disposal
	// specified).
	Disposal []byte
	// Config is the global color table (palette), width and height. A nil or
	// empty-color.Palette Config.ColorModel means that each frame has its own
	// color table and there is no global color table. Each frame's bounds must
	// be within the rectangle defined by the two points (0, 0) and
	// (Config.Width, Config.Height).
	//
	// For backwards compatibility, a zero-valued Config is valid to pass to
	// EncodeAll, and implies that the overall GIF's width and height equals
	// the first frame's bounds' Rectangle.Max point.
	Config image.Config
	// BackgroundIndex is the background index in the global color table, for
	// use with the DisposalBackground disposal method.
	BackgroundIndex byte
}

// DecodeAll reads a GIF image from r and returns the sequential frames
// and timing information.
func DecodeAll(r io.Reader) (*GIF, error) {
	var d decoder
	if err := d.decode(r, false); err != nil {
		return nil, err
	}
	gif := &GIF{
		Image:     d.image,
		LoopCount: d.loopCount,
		Delay:     d.delay,
		Disposal:  d.disposal,
		Config: image.Config{
			ColorModel: d.globalColorTable,
			Width:      d.width,
			Height:     d.height,
		},
		BackgroundIndex: d.backgroundIndex,
	}
	return gif, nil
}

// DecodeConfig returns the global color model and dimensions of a GIF image
// without decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, error) {
	var d decoder
	if err := d.decode(r, true); err != nil {
		return image.Config{}, err
	}
	return image.Config{
		ColorModel: d.globalColorTable,
		Width:      d.width,
		Height:     d.height,
	}, nil
}

func init() {
	image.RegisterFormat("gif", "GIF8?a", Decode, DecodeConfig)
}
