// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

const hexDigit = "0123456789abcdef"

// A HardwareAddr represents a physical hardware address.
type HardwareAddr []byte

func (a HardwareAddr) String() string {
	if len(a) == 0 {
		return ""
	}
	buf := make([]byte, 0, len(a)*3-1)
	for i, b := range a {
		if i > 0 {
			buf = append(buf, ':')
		}
		buf = append(buf, hexDigit[b>>4])
		buf = append(buf, hexDigit[b&0xF])
	}
	return string(buf)
}

// ParseMAC parses s as an IEEE 802 MAC-48, EUI-48, or EUI-64 using one of the
// following formats:
//   01:23:45:67:89:ab
//   01:23:45:67:89:ab:cd:ef
//   01-23-45-67-89-ab
//   01-23-45-67-89-ab-cd-ef
//   0123.4567.89ab
//   0123.4567.89ab.cdef
func ParseMAC(s string) (hw HardwareAddr, err error) {
	if len(s) < 14 {
		goto error
	}

	if s[2] == ':' || s[2] == '-' {
		if (len(s)+1)%3 != 0 {
			goto error
		}
		n := (len(s) + 1) / 3
		if n != 6 && n != 8 {
			goto error
		}
		hw = make(HardwareAddr, n)
		for x, i := 0, 0; i < n; i++ {
			var ok bool
			if hw[i], ok = xtoi2(s[x:], s[2]); !ok {
				goto error
			}
			x += 3
		}
	} else if s[4] == '.' {
		if (len(s)+1)%5 != 0 {
			goto error
		}
		n := 2 * (len(s) + 1) / 5
		if n != 6 && n != 8 {
			goto error
		}
		hw = make(HardwareAddr, n)
		for x, i := 0, 0; i < n; i += 2 {
			var ok bool
			if hw[i], ok = xtoi2(s[x:x+2], 0); !ok {
				goto error
			}
			if hw[i+1], ok = xtoi2(s[x+2:], s[4]); !ok {
				goto error
			}
			x += 5
		}
	} else {
		goto error
	}
	return hw, nil

error:
	return nil, &AddrError{Err: "invalid MAC address", Addr: s}
}
