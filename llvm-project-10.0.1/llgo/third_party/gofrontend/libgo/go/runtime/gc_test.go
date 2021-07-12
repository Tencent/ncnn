// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"testing"
	"time"
	"unsafe"
)

func TestGcSys(t *testing.T) {
	if os.Getenv("GOGC") == "off" {
		t.Skip("skipping test; GOGC=off in environment")
	}
	data := struct{ Short bool }{testing.Short()}
	got := executeTest(t, testGCSysSource, &data)
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

const testGCSysSource = `
package main

import (
	"fmt"
	"runtime"
)

func main() {
	runtime.GOMAXPROCS(1)
	memstats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(memstats)
	sys := memstats.Sys

	runtime.MemProfileRate = 0 // disable profiler

	itercount := 1000000
{{if .Short}}
	itercount = 100000
{{end}}
	for i := 0; i < itercount; i++ {
		workthegc()
	}

	// Should only be using a few MB.
	// We allocated 100 MB or (if not short) 1 GB.
	runtime.ReadMemStats(memstats)
	if sys > memstats.Sys {
		sys = 0
	} else {
		sys = memstats.Sys - sys
	}
	if sys > 16<<20 {
		fmt.Printf("using too much memory: %d bytes\n", sys)
		return
	}
	fmt.Printf("OK\n")
}

func workthegc() []byte {
	return make([]byte, 1029)
}
`

func TestGcDeepNesting(t *testing.T) {
	type T [2][2][2][2][2][2][2][2][2][2]*int
	a := new(T)

	// Prevent the compiler from applying escape analysis.
	// This makes sure new(T) is allocated on heap, not on the stack.
	t.Logf("%p", a)

	a[0][0][0][0][0][0][0][0][0][0] = new(int)
	*a[0][0][0][0][0][0][0][0][0][0] = 13
	runtime.GC()
	if *a[0][0][0][0][0][0][0][0][0][0] != 13 {
		t.Fail()
	}
}

func TestGcHashmapIndirection(t *testing.T) {
	defer debug.SetGCPercent(debug.SetGCPercent(1))
	runtime.GC()
	type T struct {
		a [256]int
	}
	m := make(map[T]T)
	for i := 0; i < 2000; i++ {
		var a T
		a.a[0] = i
		m[a] = T{}
	}
}

func TestGcArraySlice(t *testing.T) {
	type X struct {
		buf     [1]byte
		nextbuf []byte
		next    *X
	}
	var head *X
	for i := 0; i < 10; i++ {
		p := &X{}
		p.buf[0] = 42
		p.next = head
		if head != nil {
			p.nextbuf = head.buf[:]
		}
		head = p
		runtime.GC()
	}
	for p := head; p != nil; p = p.next {
		if p.buf[0] != 42 {
			t.Fatal("corrupted heap")
		}
	}
}

func TestGcRescan(t *testing.T) {
	type X struct {
		c     chan error
		nextx *X
	}
	type Y struct {
		X
		nexty *Y
		p     *int
	}
	var head *Y
	for i := 0; i < 10; i++ {
		p := &Y{}
		p.c = make(chan error)
		if head != nil {
			p.nextx = &head.X
		}
		p.nexty = head
		p.p = new(int)
		*p.p = 42
		head = p
		runtime.GC()
	}
	for p := head; p != nil; p = p.nexty {
		if *p.p != 42 {
			t.Fatal("corrupted heap")
		}
	}
}

func TestGcLastTime(t *testing.T) {
	ms := new(runtime.MemStats)
	t0 := time.Now().UnixNano()
	runtime.GC()
	t1 := time.Now().UnixNano()
	runtime.ReadMemStats(ms)
	last := int64(ms.LastGC)
	if t0 > last || last > t1 {
		t.Fatalf("bad last GC time: got %v, want [%v, %v]", last, t0, t1)
	}
	pause := ms.PauseNs[(ms.NumGC+255)%256]
	// Due to timer granularity, pause can actually be 0 on windows
	// or on virtualized environments.
	if pause == 0 {
		t.Logf("last GC pause was 0")
	} else if pause > 10e9 {
		t.Logf("bad last GC pause: got %v, want [0, 10e9]", pause)
	}
}

var hugeSink interface{}

func TestHugeGCInfo(t *testing.T) {
	// The test ensures that compiler can chew these huge types even on weakest machines.
	// The types are not allocated at runtime.
	if hugeSink != nil {
		// 400MB on 32 bots, 4TB on 64-bits.
		const n = (400 << 20) + (unsafe.Sizeof(uintptr(0))-4)<<40
		hugeSink = new([n]*byte)
		hugeSink = new([n]uintptr)
		hugeSink = new(struct {
			x float64
			y [n]*byte
			z []string
		})
		hugeSink = new(struct {
			x float64
			y [n]uintptr
			z []string
		})
	}
}

func BenchmarkSetTypePtr(b *testing.B) {
	benchSetType(b, new(*byte))
}

func BenchmarkSetTypePtr8(b *testing.B) {
	benchSetType(b, new([8]*byte))
}

func BenchmarkSetTypePtr16(b *testing.B) {
	benchSetType(b, new([16]*byte))
}

func BenchmarkSetTypePtr32(b *testing.B) {
	benchSetType(b, new([32]*byte))
}

func BenchmarkSetTypePtr64(b *testing.B) {
	benchSetType(b, new([64]*byte))
}

func BenchmarkSetTypePtr126(b *testing.B) {
	benchSetType(b, new([126]*byte))
}

func BenchmarkSetTypePtr128(b *testing.B) {
	benchSetType(b, new([128]*byte))
}

func BenchmarkSetTypePtrSlice(b *testing.B) {
	benchSetType(b, make([]*byte, 1<<10))
}

type Node1 struct {
	Value       [1]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode1(b *testing.B) {
	benchSetType(b, new(Node1))
}

func BenchmarkSetTypeNode1Slice(b *testing.B) {
	benchSetType(b, make([]Node1, 32))
}

type Node8 struct {
	Value       [8]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode8(b *testing.B) {
	benchSetType(b, new(Node8))
}

func BenchmarkSetTypeNode8Slice(b *testing.B) {
	benchSetType(b, make([]Node8, 32))
}

type Node64 struct {
	Value       [64]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode64(b *testing.B) {
	benchSetType(b, new(Node64))
}

func BenchmarkSetTypeNode64Slice(b *testing.B) {
	benchSetType(b, make([]Node64, 32))
}

type Node64Dead struct {
	Left, Right *byte
	Value       [64]uintptr
}

func BenchmarkSetTypeNode64Dead(b *testing.B) {
	benchSetType(b, new(Node64Dead))
}

func BenchmarkSetTypeNode64DeadSlice(b *testing.B) {
	benchSetType(b, make([]Node64Dead, 32))
}

type Node124 struct {
	Value       [124]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode124(b *testing.B) {
	benchSetType(b, new(Node124))
}

func BenchmarkSetTypeNode124Slice(b *testing.B) {
	benchSetType(b, make([]Node124, 32))
}

type Node126 struct {
	Value       [126]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode126(b *testing.B) {
	benchSetType(b, new(Node126))
}

func BenchmarkSetTypeNode126Slice(b *testing.B) {
	benchSetType(b, make([]Node126, 32))
}

type Node128 struct {
	Value       [128]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode128(b *testing.B) {
	benchSetType(b, new(Node128))
}

func BenchmarkSetTypeNode128Slice(b *testing.B) {
	benchSetType(b, make([]Node128, 32))
}

type Node130 struct {
	Value       [130]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode130(b *testing.B) {
	benchSetType(b, new(Node130))
}

func BenchmarkSetTypeNode130Slice(b *testing.B) {
	benchSetType(b, make([]Node130, 32))
}

type Node1024 struct {
	Value       [1024]uintptr
	Left, Right *byte
}

func BenchmarkSetTypeNode1024(b *testing.B) {
	benchSetType(b, new(Node1024))
}

func BenchmarkSetTypeNode1024Slice(b *testing.B) {
	benchSetType(b, make([]Node1024, 32))
}

func benchSetType(b *testing.B, x interface{}) {
	v := reflect.ValueOf(x)
	t := v.Type()
	switch t.Kind() {
	case reflect.Ptr:
		b.SetBytes(int64(t.Elem().Size()))
	case reflect.Slice:
		b.SetBytes(int64(t.Elem().Size()) * int64(v.Len()))
	}
	b.ResetTimer()
	//runtime.BenchSetType(b.N, x)
}

func BenchmarkAllocation(b *testing.B) {
	type T struct {
		x, y *byte
	}
	ngo := runtime.GOMAXPROCS(0)
	work := make(chan bool, b.N+ngo)
	result := make(chan *T)
	for i := 0; i < b.N; i++ {
		work <- true
	}
	for i := 0; i < ngo; i++ {
		work <- false
	}
	for i := 0; i < ngo; i++ {
		go func() {
			var x *T
			for <-work {
				for i := 0; i < 1000; i++ {
					x = &T{}
				}
			}
			result <- x
		}()
	}
	for i := 0; i < ngo; i++ {
		<-result
	}
}

func TestPrintGC(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	done := make(chan bool)
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				runtime.GC()
			}
		}
	}()
	for i := 0; i < 1e4; i++ {
		func() {
			defer print("")
		}()
	}
	close(done)
}

/*

// The implicit y, ok := x.(error) for the case error
// in testTypeSwitch used to not initialize the result y
// before passing &y to assertE2I2GC.
// Catch this by making assertE2I2 call runtime.GC,
// which will force a stack scan and failure if there are
// bad pointers, and then fill the stack with bad pointers
// and run the type switch.
func TestAssertE2I2Liveness(t *testing.T) {
	// Note that this flag is defined in export_test.go
	// and is not available to ordinary imports of runtime.
	*runtime.TestingAssertE2I2GC = true
	defer func() {
		*runtime.TestingAssertE2I2GC = false
	}()

	poisonStack()
	testTypeSwitch(io.EOF)
	poisonStack()
	testAssert(io.EOF)
	poisonStack()
	testAssertVar(io.EOF)
}

func poisonStack() uintptr {
	var x [1000]uintptr
	for i := range x {
		x[i] = 0xff
	}
	return x[123]
}

func testTypeSwitch(x interface{}) error {
	switch y := x.(type) {
	case nil:
		// ok
	case error:
		return y
	}
	return nil
}

func testAssert(x interface{}) error {
	if y, ok := x.(error); ok {
		return y
	}
	return nil
}

func testAssertVar(x interface{}) error {
	var y, ok = x.(error)
	if ok {
		return y
	}
	return nil
}

func TestAssertE2T2Liveness(t *testing.T) {
	*runtime.TestingAssertE2T2GC = true
	defer func() {
		*runtime.TestingAssertE2T2GC = false
	}()

	poisonStack()
	testIfaceEqual(io.EOF)
}

func testIfaceEqual(x interface{}) {
	if x == "abc" {
		// Prevent inlining
		panic("")
	}
}

*/
