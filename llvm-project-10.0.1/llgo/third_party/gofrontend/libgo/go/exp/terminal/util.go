// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

// Package terminal provides support functions for dealing with terminals, as
// commonly found on UNIX systems.
//
// Putting a terminal into raw mode is the most common requirement:
//
// 	oldState, err := terminal.MakeRaw(0)
// 	if err != nil {
// 	        panic(err)
// 	}
// 	defer terminal.Restore(0, oldState)
package terminal

import (
	"io"
	"syscall"
	"unsafe"
)

// State contains the state of a terminal.
type State struct {
	termios syscall.Termios
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd int) bool {
	var termios syscall.Termios
	err := syscall.Tcgetattr(fd, &termios)
	return err == nil
}

// MakeRaw put the terminal connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be
// restored.
func MakeRaw(fd int) (*State, error) {
	var oldState State
	if err := syscall.Tcgetattr(fd, &oldState.termios); err != nil {
		return nil, err
	}

	newState := oldState.termios
	newState.Iflag &^= syscall.ISTRIP | syscall.INLCR | syscall.ICRNL | syscall.IGNCR | syscall.IXON | syscall.IXOFF
	newState.Lflag &^= syscall.ECHO | syscall.ICANON | syscall.ISIG
	if err := syscall.Tcsetattr(fd, syscall.TCSANOW, &newState); err != nil {
		return nil, err
	}

	return &oldState, nil
}

// Restore restores the terminal connected to the given file descriptor to a
// previous state.
func Restore(fd int, state *State) error {
	err := syscall.Tcsetattr(fd, syscall.TCSANOW, &state.termios)
	return err
}

//extern ioctl
func ioctl(int, int, unsafe.Pointer) int

// GetSize returns the dimensions of the given terminal.
func GetSize(fd int) (width, height int, err error) {
	var dimensions [4]uint16

	if ioctl(fd, syscall.TIOCGWINSZ, unsafe.Pointer(&dimensions)) < 0 {
		return -1, -1, syscall.GetErrno()
	}
	return int(dimensions[1]), int(dimensions[0]), nil
}

// ReadPassword reads a line of input from a terminal without local echo.  This
// is commonly used for inputting passwords and other sensitive data. The slice
// returned does not include the \n.
func ReadPassword(fd int) ([]byte, error) {
	var oldState syscall.Termios
	if err := syscall.Tcgetattr(fd, &oldState); err != nil {
		return nil, err
	}

	newState := oldState
	newState.Lflag &^= syscall.ECHO
	if err := syscall.Tcsetattr(fd, syscall.TCSANOW, &newState); err != nil {
		return nil, err
	}

	defer func() {
		syscall.Tcsetattr(fd, syscall.TCSANOW, &oldState)
	}()

	var buf [16]byte
	var ret []byte
	for {
		n, err := syscall.Read(fd, buf[:])
		if err != nil {
			return nil, err
		}
		if n == 0 {
			if len(ret) == 0 {
				return nil, io.EOF
			}
			break
		}
		if buf[n-1] == '\n' {
			n--
		}
		ret = append(ret, buf[:n]...)
		if n < len(buf) {
			break
		}
	}

	return ret, nil
}
