// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package os

import (
	"syscall"
	"time"
)

func sigpipe() // implemented in package runtime

// Readlink returns the destination of the named symbolic link.
// If there is an error, it will be of type *PathError.
func Readlink(name string) (string, error) {
	for len := 128; ; len *= 2 {
		b := make([]byte, len)
		n, e := fixCount(syscall.Readlink(name, b))
		if e != nil {
			return "", &PathError{"readlink", name, e}
		}
		if n < len {
			return string(b[0:n]), nil
		}
	}
}

// syscallMode returns the syscall-specific mode bits from Go's portable mode bits.
func syscallMode(i FileMode) (o uint32) {
	o |= uint32(i.Perm())
	if i&ModeSetuid != 0 {
		o |= syscall.S_ISUID
	}
	if i&ModeSetgid != 0 {
		o |= syscall.S_ISGID
	}
	if i&ModeSticky != 0 {
		o |= syscall.S_ISVTX
	}
	// No mapping for Go's ModeTemporary (plan9 only).
	return
}

// Chmod changes the mode of the named file to mode.
// If the file is a symbolic link, it changes the mode of the link's target.
// If there is an error, it will be of type *PathError.
func Chmod(name string, mode FileMode) error {
	if e := syscall.Chmod(name, syscallMode(mode)); e != nil {
		return &PathError{"chmod", name, e}
	}
	return nil
}

// Chmod changes the mode of the file to mode.
// If there is an error, it will be of type *PathError.
func (f *File) Chmod(mode FileMode) error {
	if f == nil {
		return ErrInvalid
	}
	if e := syscall.Fchmod(f.fd, syscallMode(mode)); e != nil {
		return &PathError{"chmod", f.name, e}
	}
	return nil
}

// Chown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link's target.
// If there is an error, it will be of type *PathError.
func Chown(name string, uid, gid int) error {
	if e := syscall.Chown(name, uid, gid); e != nil {
		return &PathError{"chown", name, e}
	}
	return nil
}

// Lchown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link itself.
// If there is an error, it will be of type *PathError.
func Lchown(name string, uid, gid int) error {
	if e := syscall.Lchown(name, uid, gid); e != nil {
		return &PathError{"lchown", name, e}
	}
	return nil
}

// Chown changes the numeric uid and gid of the named file.
// If there is an error, it will be of type *PathError.
func (f *File) Chown(uid, gid int) error {
	if f == nil {
		return ErrInvalid
	}
	if e := syscall.Fchown(f.fd, uid, gid); e != nil {
		return &PathError{"chown", f.name, e}
	}
	return nil
}

// Truncate changes the size of the file.
// It does not change the I/O offset.
// If there is an error, it will be of type *PathError.
func (f *File) Truncate(size int64) error {
	if f == nil {
		return ErrInvalid
	}
	if e := syscall.Ftruncate(f.fd, size); e != nil {
		return &PathError{"truncate", f.name, e}
	}
	return nil
}

// Sync commits the current contents of the file to stable storage.
// Typically, this means flushing the file system's in-memory copy
// of recently written data to disk.
func (f *File) Sync() error {
	if f == nil {
		return ErrInvalid
	}
	if e := syscall.Fsync(f.fd); e != nil {
		return NewSyscallError("fsync", e)
	}
	return nil
}

// Chtimes changes the access and modification times of the named
// file, similar to the Unix utime() or utimes() functions.
//
// The underlying filesystem may truncate or round the values to a
// less precise time unit.
// If there is an error, it will be of type *PathError.
func Chtimes(name string, atime time.Time, mtime time.Time) error {
	var utimes [2]syscall.Timespec
	utimes[0] = syscall.NsecToTimespec(atime.UnixNano())
	utimes[1] = syscall.NsecToTimespec(mtime.UnixNano())
	if e := syscall.UtimesNano(name, utimes[0:]); e != nil {
		return &PathError{"chtimes", name, e}
	}
	return nil
}
