// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements sysSocket and accept for platforms that
// provide a fast path for setting SetNonblock and CloseOnExec.

// +build freebsd linux

package net

import (
	"os"
	"syscall"
)

// Wrapper around the socket system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func sysSocket(family, sotype, proto int) (int, error) {
	s, err := socketFunc(family, sotype|syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC, proto)
	// On Linux the SOCK_NONBLOCK and SOCK_CLOEXEC flags were
	// introduced in 2.6.27 kernel and on FreeBSD both flags were
	// introduced in 10 kernel. If we get an EINVAL error on Linux
	// or EPROTONOSUPPORT error on FreeBSD, fall back to using
	// socket without them.
	switch err {
	case nil:
		return s, nil
	default:
		return -1, os.NewSyscallError("socket", err)
	case syscall.EPROTONOSUPPORT, syscall.EINVAL:
	}

	// See ../syscall/exec_unix.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err = socketFunc(family, sotype, proto)
	if err == nil {
		syscall.CloseOnExec(s)
	}
	syscall.ForkLock.RUnlock()
	if err != nil {
		return -1, os.NewSyscallError("socket", err)
	}
	if err = syscall.SetNonblock(s, true); err != nil {
		closeFunc(s)
		return -1, os.NewSyscallError("setnonblock", err)
	}
	return s, nil
}

// Wrapper around the accept system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func accept(s int) (int, syscall.Sockaddr, error) {
	ns, sa, err := accept4Func(s, syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC)
	// On Linux the accept4 system call was introduced in 2.6.28
	// kernel and on FreeBSD it was introduced in 10 kernel. If we
	// get an ENOSYS error on both Linux and FreeBSD, or EINVAL
	// error on Linux, fall back to using accept.
	switch err {
	case nil:
		return ns, sa, nil
	default: // errors other than the ones listed
		return -1, sa, os.NewSyscallError("accept4", err)
	case syscall.ENOSYS: // syscall missing
	case syscall.EINVAL: // some Linux use this instead of ENOSYS
	case syscall.EACCES: // some Linux use this instead of ENOSYS
	case syscall.EFAULT: // some Linux use this instead of ENOSYS
	}

	// See ../syscall/exec_unix.go for description of ForkLock.
	// It is probably okay to hold the lock across syscall.Accept
	// because we have put fd.sysfd into non-blocking mode.
	// However, a call to the File method will put it back into
	// blocking mode. We can't take that risk, so no use of ForkLock here.
	ns, sa, err = acceptFunc(s)
	if err == nil {
		syscall.CloseOnExec(ns)
	}
	if err != nil {
		return -1, nil, os.NewSyscallError("accept", err)
	}
	if err = syscall.SetNonblock(ns, true); err != nil {
		closeFunc(ns)
		return -1, nil, os.NewSyscallError("setnonblock", err)
	}
	return ns, sa, nil
}
