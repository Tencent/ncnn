// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fnv implements FNV-1 and FNV-1a, non-cryptographic hash functions
// created by Glenn Fowler, Landon Curt Noll, and Phong Vo.
// See
// http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function.
package fnv

import (
	"hash"
)

type (
	sum32  uint32
	sum32a uint32
	sum64  uint64
	sum64a uint64
)

const (
	offset32 = 2166136261
	offset64 = 14695981039346656037
	prime32  = 16777619
	prime64  = 1099511628211
)

// New32 returns a new 32-bit FNV-1 hash.Hash.
func New32() hash.Hash32 {
	var s sum32 = offset32
	return &s
}

// New32a returns a new 32-bit FNV-1a hash.Hash.
func New32a() hash.Hash32 {
	var s sum32a = offset32
	return &s
}

// New64 returns a new 64-bit FNV-1 hash.Hash.
func New64() hash.Hash64 {
	var s sum64 = offset64
	return &s
}

// New64a returns a new 64-bit FNV-1a hash.Hash.
func New64a() hash.Hash64 {
	var s sum64a = offset64
	return &s
}

func (s *sum32) Reset()  { *s = offset32 }
func (s *sum32a) Reset() { *s = offset32 }
func (s *sum64) Reset()  { *s = offset64 }
func (s *sum64a) Reset() { *s = offset64 }

func (s *sum32) Sum32() uint32  { return uint32(*s) }
func (s *sum32a) Sum32() uint32 { return uint32(*s) }
func (s *sum64) Sum64() uint64  { return uint64(*s) }
func (s *sum64a) Sum64() uint64 { return uint64(*s) }

func (s *sum32) Write(data []byte) (int, error) {
	hash := *s
	for _, c := range data {
		hash *= prime32
		hash ^= sum32(c)
	}
	*s = hash
	return len(data), nil
}

func (s *sum32a) Write(data []byte) (int, error) {
	hash := *s
	for _, c := range data {
		hash ^= sum32a(c)
		hash *= prime32
	}
	*s = hash
	return len(data), nil
}

func (s *sum64) Write(data []byte) (int, error) {
	hash := *s
	for _, c := range data {
		hash *= prime64
		hash ^= sum64(c)
	}
	*s = hash
	return len(data), nil
}

func (s *sum64a) Write(data []byte) (int, error) {
	hash := *s
	for _, c := range data {
		hash ^= sum64a(c)
		hash *= prime64
	}
	*s = hash
	return len(data), nil
}

func (s *sum32) Size() int  { return 4 }
func (s *sum32a) Size() int { return 4 }
func (s *sum64) Size() int  { return 8 }
func (s *sum64a) Size() int { return 8 }

func (s *sum32) BlockSize() int  { return 1 }
func (s *sum32a) BlockSize() int { return 1 }
func (s *sum64) BlockSize() int  { return 1 }
func (s *sum64a) BlockSize() int { return 1 }

func (s *sum32) Sum(in []byte) []byte {
	v := uint32(*s)
	return append(in, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

func (s *sum32a) Sum(in []byte) []byte {
	v := uint32(*s)
	return append(in, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

func (s *sum64) Sum(in []byte) []byte {
	v := uint64(*s)
	return append(in, byte(v>>56), byte(v>>48), byte(v>>40), byte(v>>32), byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

func (s *sum64a) Sum(in []byte) []byte {
	v := uint64(*s)
	return append(in, byte(v>>56), byte(v>>48), byte(v>>40), byte(v>>32), byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}
