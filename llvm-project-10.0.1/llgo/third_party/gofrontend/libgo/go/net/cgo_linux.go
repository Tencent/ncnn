// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !android,cgo,!netgo

package net

/*
#include <netdb.h>
*/

import "syscall"

// NOTE(rsc): In theory there are approximately balanced
// arguments for and against including AI_ADDRCONFIG
// in the flags (it includes IPv4 results only on IPv4 systems,
// and similarly for IPv6), but in practice setting it causes
// getaddrinfo to return the wrong canonical name on Linux.
// So definitely leave it out.
const cgoAddrInfoFlags = syscall.AI_CANONNAME | syscall.AI_V4MAPPED | syscall.AI_ALL
