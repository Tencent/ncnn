// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/internal/socktest"
	"runtime"
	"sync"
	"testing"
	"time"
)

var dialTimeoutTests = []struct {
	timeout time.Duration
	delta   time.Duration // for deadline

	guard time.Duration
	max   time.Duration
}{
	// Tests that dial timeouts, deadlines in the past work.
	{-5 * time.Second, 0, -5 * time.Second, 100 * time.Millisecond},
	{0, -5 * time.Second, -5 * time.Second, 100 * time.Millisecond},
	{-5 * time.Second, 5 * time.Second, -5 * time.Second, 100 * time.Millisecond}, // timeout over deadline

	{50 * time.Millisecond, 0, 100 * time.Millisecond, time.Second},
	{0, 50 * time.Millisecond, 100 * time.Millisecond, time.Second},
	{50 * time.Millisecond, 5 * time.Second, 100 * time.Millisecond, time.Second}, // timeout over deadline
}

func TestDialTimeout(t *testing.T) {
	origTestHookDialChannel := testHookDialChannel
	defer func() { testHookDialChannel = origTestHookDialChannel }()
	defer sw.Set(socktest.FilterConnect, nil)

	// Avoid tracking open-close jitterbugs between netFD and
	// socket that leads to confusion of information inside
	// socktest.Switch.
	// It may happen when the Dial call bumps against TCP
	// simultaneous open. See selfConnect in tcpsock_posix.go.
	defer func() {
		sw.Set(socktest.FilterClose, nil)
		forceCloseSockets()
	}()
	sw.Set(socktest.FilterClose, func(so *socktest.Status) (socktest.AfterFilter, error) {
		return nil, errTimedout
	})

	for i, tt := range dialTimeoutTests {
		switch runtime.GOOS {
		case "plan9", "windows":
			testHookDialChannel = func() { time.Sleep(tt.guard) }
			if runtime.GOOS == "plan9" {
				break
			}
			fallthrough
		default:
			sw.Set(socktest.FilterConnect, func(so *socktest.Status) (socktest.AfterFilter, error) {
				time.Sleep(tt.guard)
				return nil, errTimedout
			})
		}

		ch := make(chan error)
		d := Dialer{Timeout: tt.timeout}
		if tt.delta != 0 {
			d.Deadline = time.Now().Add(tt.delta)
		}
		max := time.NewTimer(tt.max)
		defer max.Stop()
		go func() {
			// This dial never starts to send any TCP SYN
			// segment because of above socket filter and
			// test hook.
			c, err := d.Dial("tcp", "127.0.0.1:0")
			if err == nil {
				err = fmt.Errorf("unexpectedly established: tcp:%s->%s", c.LocalAddr(), c.RemoteAddr())
				c.Close()
			}
			ch <- err
		}()

		select {
		case <-max.C:
			t.Fatalf("#%d: Dial didn't return in an expected time", i)
		case err := <-ch:
			if perr := parseDialError(err); perr != nil {
				t.Errorf("#%d: %v", i, perr)
			}
			if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
				t.Fatalf("#%d: %v", i, err)
			}
		}
	}
}

var acceptTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that accept deadlines in the past work, even if
	// there's incoming connections available.
	{-5 * time.Second, [2]error{errTimeout, errTimeout}},

	{50 * time.Millisecond, [2]error{nil, errTimeout}},
}

func TestAcceptTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	for i, tt := range acceptTimeoutTests {
		if tt.timeout < 0 {
			go func() {
				c, err := Dial(ln.Addr().Network(), ln.Addr().String())
				if err != nil {
					t.Error(err)
					return
				}
				var b [1]byte
				c.Read(b[:])
				c.Close()
			}()
		}

		if err := ln.(*TCPListener).SetDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("$%d: %v", i, err)
		}
		for j, xerr := range tt.xerrs {
			for {
				c, err := ln.Accept()
				if xerr != nil {
					if perr := parseAcceptError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					c.Close()
					time.Sleep(tt.timeout / 3)
					continue
				}
				break
			}
		}
	}
}

func TestAcceptTimeoutMustReturn(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	max := time.NewTimer(time.Second)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := ln.(*TCPListener).SetDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		if err := ln.(*TCPListener).SetDeadline(time.Now().Add(10 * time.Millisecond)); err != nil {
			t.Error(err)
		}
		c, err := ln.Accept()
		if err == nil {
			c.Close()
		}
		ch <- err
	}()

	select {
	case <-max.C:
		ln.Close()
		<-ch // wait for tester goroutine to stop
		t.Fatal("Accept didn't return in an expected time")
	case err := <-ch:
		if perr := parseAcceptError(err); perr != nil {
			t.Error(perr)
		}
		if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
			t.Fatal(err)
		}
	}
}

func TestAcceptTimeoutMustNotReturn(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := ln.(*TCPListener).SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := ln.(*TCPListener).SetDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		_, err := ln.Accept()
		ch <- err
	}()

	select {
	case err := <-ch:
		if perr := parseAcceptError(err); perr != nil {
			t.Error(perr)
		}
		t.Fatalf("expected Accept to not return, but it returned with %v", err)
	case <-max.C:
		ln.Close()
		<-ch // wait for tester goroutine to stop
	}
}

var readTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that read deadlines work, even if there's data ready
	// to be read.
	{-5 * time.Second, [2]error{errTimeout, errTimeout}},

	{50 * time.Millisecond, [2]error{nil, errTimeout}},
}

func TestReadTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		c.Write([]byte("READ TIMEOUT TEST"))
		defer c.Close()
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for i, tt := range readTimeoutTests {
		if err := c.SetReadDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		var b [1]byte
		for j, xerr := range tt.xerrs {
			for {
				n, err := c.Read(b[:])
				if xerr != nil {
					if perr := parseReadError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					time.Sleep(tt.timeout / 3)
					continue
				}
				if n != 0 {
					t.Fatalf("#%d/%d: read %d; want 0", i, j, n)
				}
				break
			}
		}
	}
}

func TestReadTimeoutMustNotReturn(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := c.SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetWriteDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetReadDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		var b [1]byte
		_, err := c.Read(b[:])
		ch <- err
	}()

	select {
	case err := <-ch:
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		t.Fatalf("expected Read to not return, but it returned with %v", err)
	case <-max.C:
		c.Close()
		err := <-ch // wait for tester goroutine to stop
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		if err == io.EOF && runtime.GOOS == "nacl" { // see golang.org/issue/8044
			return
		}
		if nerr, ok := err.(Error); !ok || nerr.Timeout() || nerr.Temporary() {
			t.Fatal(err)
		}
	}
}

var readFromTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that read deadlines work, even if there's data ready
	// to be read.
	{-5 * time.Second, [2]error{errTimeout, errTimeout}},

	{50 * time.Millisecond, [2]error{nil, errTimeout}},
}

func TestReadFromTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS) // see golang.org/issue/8916
	}

	ch := make(chan Addr)
	defer close(ch)
	handler := func(ls *localPacketServer, c PacketConn) {
		if dst, ok := <-ch; ok {
			c.WriteTo([]byte("READFROM TIMEOUT TEST"), dst)
		}
	}
	ls, err := newLocalPacketServer("udp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	host, _, err := SplitHostPort(ls.PacketConn.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenPacket(ls.PacketConn.LocalAddr().Network(), JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ch <- c.LocalAddr()

	for i, tt := range readFromTimeoutTests {
		if err := c.SetReadDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		var b [1]byte
		for j, xerr := range tt.xerrs {
			for {
				n, _, err := c.ReadFrom(b[:])
				if xerr != nil {
					if perr := parseReadError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					time.Sleep(tt.timeout / 3)
					continue
				}
				if n != 0 {
					t.Fatalf("#%d/%d: read %d; want 0", i, j, n)
				}
				break
			}
		}
	}
}

var writeTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that write deadlines work, even if there's buffer
	// space available to write.
	{-5 * time.Second, [2]error{errTimeout, errTimeout}},

	{10 * time.Millisecond, [2]error{nil, errTimeout}},
}

func TestWriteTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	for i, tt := range writeTimeoutTests {
		c, err := Dial(ln.Addr().Network(), ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		if err := c.SetWriteDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		for j, xerr := range tt.xerrs {
			for {
				n, err := c.Write([]byte("WRITE TIMEOUT TEST"))
				if xerr != nil {
					if perr := parseWriteError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					time.Sleep(tt.timeout / 3)
					continue
				}
				if n != 0 {
					t.Fatalf("#%d/%d: wrote %d; want 0", i, j, n)
				}
				break
			}
		}
	}
}

func TestWriteTimeoutMustNotReturn(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	max := time.NewTimer(100 * time.Millisecond)
	defer max.Stop()
	ch := make(chan error)
	go func() {
		if err := c.SetDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetReadDeadline(time.Now().Add(-5 * time.Second)); err != nil {
			t.Error(err)
		}
		if err := c.SetWriteDeadline(noDeadline); err != nil {
			t.Error(err)
		}
		var b [1]byte
		for {
			if _, err := c.Write(b[:]); err != nil {
				ch <- err
				break
			}
		}
	}()

	select {
	case err := <-ch:
		if perr := parseWriteError(err); perr != nil {
			t.Error(perr)
		}
		t.Fatalf("expected Write to not return, but it returned with %v", err)
	case <-max.C:
		c.Close()
		err := <-ch // wait for tester goroutine to stop
		if perr := parseWriteError(err); perr != nil {
			t.Error(perr)
		}
		if nerr, ok := err.(Error); !ok || nerr.Timeout() || nerr.Temporary() {
			t.Fatal(err)
		}
	}
}

var writeToTimeoutTests = []struct {
	timeout time.Duration
	xerrs   [2]error // expected errors in transition
}{
	// Tests that write deadlines work, even if there's buffer
	// space available to write.
	{-5 * time.Second, [2]error{errTimeout, errTimeout}},

	{10 * time.Millisecond, [2]error{nil, errTimeout}},
}

func TestWriteToTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	c1, err := newLocalPacketListener("udp")
	if err != nil {
		t.Fatal(err)
	}
	defer c1.Close()

	host, _, err := SplitHostPort(c1.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}

	for i, tt := range writeToTimeoutTests {
		c2, err := ListenPacket(c1.LocalAddr().Network(), JoinHostPort(host, "0"))
		if err != nil {
			t.Fatal(err)
		}
		defer c2.Close()

		if err := c2.SetWriteDeadline(time.Now().Add(tt.timeout)); err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		for j, xerr := range tt.xerrs {
			for {
				n, err := c2.WriteTo([]byte("WRITETO TIMEOUT TEST"), c1.LocalAddr())
				if xerr != nil {
					if perr := parseWriteError(err); perr != nil {
						t.Errorf("#%d/%d: %v", i, j, perr)
					}
					if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
						t.Fatalf("#%d/%d: %v", i, j, err)
					}
				}
				if err == nil {
					time.Sleep(tt.timeout / 3)
					continue
				}
				if n != 0 {
					t.Fatalf("#%d/%d: wrote %d; want 0", i, j, n)
				}
				break
			}
		}
	}
}

func TestReadTimeoutFluctuation(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	max := time.NewTimer(time.Second)
	defer max.Stop()
	ch := make(chan error)
	go timeoutReceiver(c, 100*time.Millisecond, 50*time.Millisecond, 250*time.Millisecond, ch)

	select {
	case <-max.C:
		t.Fatal("Read took over 1s; expected 0.1s")
	case err := <-ch:
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
			t.Fatal(err)
		}
	}
}

func TestReadFromTimeoutFluctuation(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	c1, err := newLocalPacketListener("udp")
	if err != nil {
		t.Fatal(err)
	}
	defer c1.Close()

	c2, err := Dial(c1.LocalAddr().Network(), c1.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c2.Close()

	max := time.NewTimer(time.Second)
	defer max.Stop()
	ch := make(chan error)
	go timeoutPacketReceiver(c2.(PacketConn), 100*time.Millisecond, 50*time.Millisecond, 250*time.Millisecond, ch)

	select {
	case <-max.C:
		t.Fatal("ReadFrom took over 1s; expected 0.1s")
	case err := <-ch:
		if perr := parseReadError(err); perr != nil {
			t.Error(perr)
		}
		if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
			t.Fatal(err)
		}
	}
}

func TestWriteTimeoutFluctuation(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	d := time.Second
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		d = 3 * time.Second // see golang.org/issue/10775
	}
	max := time.NewTimer(d)
	defer max.Stop()
	ch := make(chan error)
	go timeoutTransmitter(c, 100*time.Millisecond, 50*time.Millisecond, 250*time.Millisecond, ch)

	select {
	case <-max.C:
		t.Fatalf("Write took over %v; expected 0.1s", d)
	case err := <-ch:
		if perr := parseWriteError(err); perr != nil {
			t.Error(perr)
		}
		if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
			t.Fatal(err)
		}
	}
}

func TestVariousDeadlines1Proc(t *testing.T) {
	testVariousDeadlines(t, 1)
}

func TestVariousDeadlines4Proc(t *testing.T) {
	testVariousDeadlines(t, 4)
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

func testVariousDeadlines(t *testing.T, maxProcs int) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(maxProcs))

	type result struct {
		n   int64
		err error
		d   time.Duration
	}

	ch := make(chan error, 1)
	pasvch := make(chan result)
	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				ch <- err
				return
			}
			// The server, with no timeouts of its own,
			// sending bytes to clients as fast as it can.
			go func() {
				t0 := time.Now()
				n, err := io.Copy(c, neverEnding('a'))
				dt := time.Since(t0)
				c.Close()
				pasvch <- result{n, err, dt}
			}()
		}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	for _, timeout := range []time.Duration{
		1 * time.Nanosecond,
		2 * time.Nanosecond,
		5 * time.Nanosecond,
		50 * time.Nanosecond,
		100 * time.Nanosecond,
		200 * time.Nanosecond,
		500 * time.Nanosecond,
		750 * time.Nanosecond,
		1 * time.Microsecond,
		5 * time.Microsecond,
		25 * time.Microsecond,
		250 * time.Microsecond,
		500 * time.Microsecond,
		1 * time.Millisecond,
		5 * time.Millisecond,
		100 * time.Millisecond,
		250 * time.Millisecond,
		500 * time.Millisecond,
		1 * time.Second,
	} {
		numRuns := 3
		if testing.Short() {
			numRuns = 1
			if timeout > 500*time.Microsecond {
				continue
			}
		}
		for run := 0; run < numRuns; run++ {
			name := fmt.Sprintf("%v run %d/%d", timeout, run+1, numRuns)
			t.Log(name)

			c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
			if err != nil {
				t.Fatal(err)
			}

			tooLong := 5 * time.Second
			max := time.NewTimer(tooLong)
			defer max.Stop()
			actvch := make(chan result)
			go func() {
				t0 := time.Now()
				if err := c.SetDeadline(t0.Add(timeout)); err != nil {
					t.Error(err)
				}
				n, err := io.Copy(ioutil.Discard, c)
				dt := time.Since(t0)
				c.Close()
				actvch <- result{n, err, dt}
			}()

			select {
			case res := <-actvch:
				if nerr, ok := res.err.(Error); ok && nerr.Timeout() {
					t.Logf("for %v, good client timeout after %v, reading %d bytes", name, res.d, res.n)
				} else {
					t.Fatalf("for %v, client Copy = %d, %v; want timeout", name, res.n, res.err)
				}
			case <-max.C:
				t.Fatalf("for %v, timeout (%v) waiting for client to timeout (%v) reading", name, tooLong, timeout)
			}

			select {
			case res := <-pasvch:
				t.Logf("for %v, server in %v wrote %d: %v", name, res.d, res.n, res.err)
			case err := <-ch:
				t.Fatalf("for %v, Accept = %v", name, err)
			case <-max.C:
				t.Fatalf("for %v, timeout waiting for server to finish writing", name)
			}
		}
	}
}

// TestReadWriteProlongedTimeout tests concurrent deadline
// modification. Known to cause data races in the past.
func TestReadWriteProlongedTimeout(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		defer c.Close()

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			var b [1]byte
			for {
				if err := c.SetReadDeadline(time.Now().Add(time.Hour)); err != nil {
					if perr := parseCommonError(err); perr != nil {
						t.Error(perr)
					}
					t.Error(err)
					return
				}
				if _, err := c.Read(b[:]); err != nil {
					if perr := parseReadError(err); perr != nil {
						t.Error(perr)
					}
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			var b [1]byte
			for {
				if err := c.SetWriteDeadline(time.Now().Add(time.Hour)); err != nil {
					if perr := parseCommonError(err); perr != nil {
						t.Error(perr)
					}
					t.Error(err)
					return
				}
				if _, err := c.Write(b[:]); err != nil {
					if perr := parseWriteError(err); perr != nil {
						t.Error(perr)
					}
					return
				}
			}
		}()
		wg.Wait()
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	var b [1]byte
	for i := 0; i < 1000; i++ {
		c.Write(b[:])
		c.Read(b[:])
	}
}

func TestReadWriteDeadlineRace(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	N := 1000
	if testing.Short() {
		N = 50
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		tic := time.NewTicker(2 * time.Microsecond)
		defer tic.Stop()
		for i := 0; i < N; i++ {
			if err := c.SetReadDeadline(time.Now().Add(2 * time.Microsecond)); err != nil {
				if perr := parseCommonError(err); perr != nil {
					t.Error(perr)
				}
				break
			}
			if err := c.SetWriteDeadline(time.Now().Add(2 * time.Microsecond)); err != nil {
				if perr := parseCommonError(err); perr != nil {
					t.Error(perr)
				}
				break
			}
			<-tic.C
		}
	}()
	go func() {
		defer wg.Done()
		var b [1]byte
		for i := 0; i < N; i++ {
			c.Read(b[:]) // ignore possible timeout errors
		}
	}()
	go func() {
		defer wg.Done()
		var b [1]byte
		for i := 0; i < N; i++ {
			c.Write(b[:]) // ignore possible timeout errors
		}
	}()
	wg.Wait() // wait for tester goroutine to stop
}
