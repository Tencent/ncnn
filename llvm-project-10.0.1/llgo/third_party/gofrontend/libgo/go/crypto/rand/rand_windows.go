// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows cryptographically secure pseudorandom number
// generator.

package rand

import (
	"os"
	"sync"
	"syscall"
)

// Implemented by using Windows CryptoAPI 2.0.

func init() { Reader = &rngReader{} }

// A rngReader satisfies reads by reading from the Windows CryptGenRandom API.
type rngReader struct {
	prov syscall.Handle
	mu   sync.Mutex
}

func (r *rngReader) Read(b []byte) (n int, err error) {
	r.mu.Lock()
	if r.prov == 0 {
		const provType = syscall.PROV_RSA_FULL
		const flags = syscall.CRYPT_VERIFYCONTEXT | syscall.CRYPT_SILENT
		err := syscall.CryptAcquireContext(&r.prov, nil, nil, provType, flags)
		if err != nil {
			r.mu.Unlock()
			return 0, os.NewSyscallError("CryptAcquireContext", err)
		}
	}
	r.mu.Unlock()

	if len(b) == 0 {
		return 0, nil
	}
	err = syscall.CryptGenRandom(r.prov, uint32(len(b)), &b[0])
	if err != nil {
		return 0, os.NewSyscallError("CryptGenRandom", err)
	}
	return len(b), nil
}
