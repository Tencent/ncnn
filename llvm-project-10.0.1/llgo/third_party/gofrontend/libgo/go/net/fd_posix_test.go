// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package net

import (
	"io"
	"syscall"
	"testing"
)

var eofErrorTests = []struct {
	n        int
	err      error
	fd       *netFD
	expected error
}{
	{100, nil, &netFD{sotype: syscall.SOCK_STREAM}, nil},
	{100, io.EOF, &netFD{sotype: syscall.SOCK_STREAM}, io.EOF},
	{100, errClosing, &netFD{sotype: syscall.SOCK_STREAM}, errClosing},
	{0, nil, &netFD{sotype: syscall.SOCK_STREAM}, io.EOF},
	{0, io.EOF, &netFD{sotype: syscall.SOCK_STREAM}, io.EOF},
	{0, errClosing, &netFD{sotype: syscall.SOCK_STREAM}, errClosing},

	{100, nil, &netFD{sotype: syscall.SOCK_DGRAM}, nil},
	{100, io.EOF, &netFD{sotype: syscall.SOCK_DGRAM}, io.EOF},
	{100, errClosing, &netFD{sotype: syscall.SOCK_DGRAM}, errClosing},
	{0, nil, &netFD{sotype: syscall.SOCK_DGRAM}, nil},
	{0, io.EOF, &netFD{sotype: syscall.SOCK_DGRAM}, io.EOF},
	{0, errClosing, &netFD{sotype: syscall.SOCK_DGRAM}, errClosing},

	{100, nil, &netFD{sotype: syscall.SOCK_SEQPACKET}, nil},
	{100, io.EOF, &netFD{sotype: syscall.SOCK_SEQPACKET}, io.EOF},
	{100, errClosing, &netFD{sotype: syscall.SOCK_SEQPACKET}, errClosing},
	{0, nil, &netFD{sotype: syscall.SOCK_SEQPACKET}, io.EOF},
	{0, io.EOF, &netFD{sotype: syscall.SOCK_SEQPACKET}, io.EOF},
	{0, errClosing, &netFD{sotype: syscall.SOCK_SEQPACKET}, errClosing},

	{100, nil, &netFD{sotype: syscall.SOCK_RAW}, nil},
	{100, io.EOF, &netFD{sotype: syscall.SOCK_RAW}, io.EOF},
	{100, errClosing, &netFD{sotype: syscall.SOCK_RAW}, errClosing},
	{0, nil, &netFD{sotype: syscall.SOCK_RAW}, nil},
	{0, io.EOF, &netFD{sotype: syscall.SOCK_RAW}, io.EOF},
	{0, errClosing, &netFD{sotype: syscall.SOCK_RAW}, errClosing},
}

func TestEOFError(t *testing.T) {
	for _, tt := range eofErrorTests {
		actual := tt.fd.eofError(tt.n, tt.err)
		if actual != tt.expected {
			t.Errorf("eofError(%v, %v, %v): expected %v, actual %v", tt.n, tt.err, tt.fd.sotype, tt.expected, actual)
		}
	}
}
