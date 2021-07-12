// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package runtime_test

import (
	"bytes"
	"runtime"
	"testing"
)

const (
	typeScalar  = 0
	typePointer = 1
)

// TestGCInfo tests that various objects in heap, data and bss receive correct GC pointer type info.
func TestGCInfo(t *testing.T) {
	t.Skip("skipping on gccgo for now")

	verifyGCInfo(t, "stack Ptr", new(Ptr), infoPtr)
	verifyGCInfo(t, "stack ScalarPtr", new(ScalarPtr), infoScalarPtr)
	verifyGCInfo(t, "stack PtrScalar", new(PtrScalar), infoPtrScalar)
	verifyGCInfo(t, "stack BigStruct", new(BigStruct), infoBigStruct())
	verifyGCInfo(t, "stack string", new(string), infoString)
	verifyGCInfo(t, "stack slice", new([]string), infoSlice)
	verifyGCInfo(t, "stack eface", new(interface{}), infoEface)
	verifyGCInfo(t, "stack iface", new(Iface), infoIface)

	for i := 0; i < 10; i++ {
		verifyGCInfo(t, "heap Ptr", escape(new(Ptr)), trimDead(padDead(infoPtr)))
		verifyGCInfo(t, "heap PtrSlice", escape(&make([]*byte, 10)[0]), trimDead(infoPtr10))
		verifyGCInfo(t, "heap ScalarPtr", escape(new(ScalarPtr)), trimDead(infoScalarPtr))
		verifyGCInfo(t, "heap ScalarPtrSlice", escape(&make([]ScalarPtr, 4)[0]), trimDead(infoScalarPtr4))
		verifyGCInfo(t, "heap PtrScalar", escape(new(PtrScalar)), trimDead(infoPtrScalar))
		verifyGCInfo(t, "heap BigStruct", escape(new(BigStruct)), trimDead(infoBigStruct()))
		verifyGCInfo(t, "heap string", escape(new(string)), trimDead(infoString))
		verifyGCInfo(t, "heap eface", escape(new(interface{})), trimDead(infoEface))
		verifyGCInfo(t, "heap iface", escape(new(Iface)), trimDead(infoIface))
	}
}

func verifyGCInfo(t *testing.T, name string, p interface{}, mask0 []byte) {
	mask := runtime.GCMask(p)
	if bytes.Compare(mask, mask0) != 0 {
		t.Errorf("bad GC program for %v:\nwant %+v\ngot  %+v", name, mask0, mask)
		return
	}
}

func padDead(mask []byte) []byte {
	// Because the dead bit isn't encoded until the third word,
	// and because on 32-bit systems a one-word allocation
	// uses a two-word block, the pointer info for a one-word
	// object needs to be expanded to include an extra scalar
	// on 32-bit systems to match the heap bitmap.
	if runtime.PtrSize == 4 && len(mask) == 1 {
		return []byte{mask[0], 0}
	}
	return mask
}

func trimDead(mask []byte) []byte {
	for len(mask) > 2 && mask[len(mask)-1] == typeScalar {
		mask = mask[:len(mask)-1]
	}
	return mask
}

var gcinfoSink interface{}

func escape(p interface{}) interface{} {
	gcinfoSink = p
	return p
}

var infoPtr = []byte{typePointer}

type Ptr struct {
	*byte
}

var infoPtr10 = []byte{typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer}

type ScalarPtr struct {
	q int
	w *int
	e int
	r *int
	t int
	y *int
}

var infoScalarPtr = []byte{typeScalar, typePointer, typeScalar, typePointer, typeScalar, typePointer}

var infoScalarPtr4 = append(append(append(append([]byte(nil), infoScalarPtr...), infoScalarPtr...), infoScalarPtr...), infoScalarPtr...)

type PtrScalar struct {
	q *int
	w int
	e *int
	r int
	t *int
	y int
}

var infoPtrScalar = []byte{typePointer, typeScalar, typePointer, typeScalar, typePointer, typeScalar}

type BigStruct struct {
	q *int
	w byte
	e [17]byte
	r []byte
	t int
	y uint16
	u uint64
	i string
}

func infoBigStruct() []byte {
	switch runtime.GOARCH {
	case "386", "arm":
		return []byte{
			typePointer,                                                // q *int
			typeScalar, typeScalar, typeScalar, typeScalar, typeScalar, // w byte; e [17]byte
			typePointer, typeScalar, typeScalar, // r []byte
			typeScalar, typeScalar, typeScalar, typeScalar, // t int; y uint16; u uint64
			typePointer, typeScalar, // i string
		}
	case "arm64", "amd64", "ppc64", "ppc64le":
		return []byte{
			typePointer,                        // q *int
			typeScalar, typeScalar, typeScalar, // w byte; e [17]byte
			typePointer, typeScalar, typeScalar, // r []byte
			typeScalar, typeScalar, typeScalar, // t int; y uint16; u uint64
			typePointer, typeScalar, // i string
		}
	case "amd64p32":
		return []byte{
			typePointer,                                                // q *int
			typeScalar, typeScalar, typeScalar, typeScalar, typeScalar, // w byte; e [17]byte
			typePointer, typeScalar, typeScalar, // r []byte
			typeScalar, typeScalar, typeScalar, typeScalar, typeScalar, // t int; y uint16; u uint64
			typePointer, typeScalar, // i string
		}
	default:
		panic("unknown arch")
	}
}

type Iface interface {
	f()
}

type IfaceImpl int

func (IfaceImpl) f() {
}

var (
	// BSS
	bssPtr       Ptr
	bssScalarPtr ScalarPtr
	bssPtrScalar PtrScalar
	bssBigStruct BigStruct
	bssString    string
	bssSlice     []string
	bssEface     interface{}
	bssIface     Iface

	// DATA
	dataPtr                   = Ptr{new(byte)}
	dataScalarPtr             = ScalarPtr{q: 1}
	dataPtrScalar             = PtrScalar{w: 1}
	dataBigStruct             = BigStruct{w: 1}
	dataString                = "foo"
	dataSlice                 = []string{"foo"}
	dataEface     interface{} = 42
	dataIface     Iface       = IfaceImpl(42)

	infoString = []byte{typePointer, typeScalar}
	infoSlice  = []byte{typePointer, typeScalar, typeScalar}
	infoEface  = []byte{typePointer, typePointer}
	infoIface  = []byte{typePointer, typePointer}
)
