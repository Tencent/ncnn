// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"syscall"
)

// maxSendfileSize is the largest chunk size we ask the kernel to copy
// at a time.
const maxSendfileSize int = 4 << 20

// sendFile copies the contents of r to c using the sendfile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number of bytes copied and any
// non-EOF error.
//
// if handled == false, sendFile performed no work.
func sendFile(c *netFD, r io.Reader) (written int64, err error, handled bool) {
	var remain int64 = 1 << 62 // by default, copy until EOF

	lr, ok := r.(*io.LimitedReader)
	if ok {
		remain, r = lr.N, lr.R
		if remain <= 0 {
			return 0, nil, true
		}
	}
	f, ok := r.(*os.File)
	if !ok {
		return 0, nil, false
	}

	if err := c.writeLock(); err != nil {
		return 0, err, true
	}
	defer c.writeUnlock()

	dst := c.sysfd
	src := int(f.Fd())
	for remain > 0 {
		n := maxSendfileSize
		if int64(n) > remain {
			n = int(remain)
		}
		n, err1 := syscall.Sendfile(dst, src, nil, n)
		if n > 0 {
			written += int64(n)
			remain -= int64(n)
		}
		if n == 0 && err1 == nil {
			break
		}
		if err1 == syscall.EAGAIN {
			if err1 = c.pd.WaitWrite(); err1 == nil {
				continue
			}
		}
		if err1 != nil {
			// This includes syscall.ENOSYS (no kernel
			// support) and syscall.EINVAL (fd types which
			// don't implement sendfile)
			err = err1
			break
		}
	}
	if lr != nil {
		lr.N = remain
	}
	if err != nil {
		err = os.NewSyscallError("sendfile", err)
	}
	return written, err, written > 0
}
