// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"os"
)

func query(filename, query string, bufSize int) (res []string, err error) {
	file, err := os.OpenFile(filename, os.O_RDWR, 0)
	if err != nil {
		return
	}
	defer file.Close()

	_, err = file.Seek(0, 0)
	if err != nil {
		return
	}
	_, err = file.WriteString(query)
	if err != nil {
		return
	}
	_, err = file.Seek(0, 0)
	if err != nil {
		return
	}
	buf := make([]byte, bufSize)
	for {
		n, _ := file.Read(buf)
		if n <= 0 {
			break
		}
		res = append(res, string(buf[:n]))
	}
	return
}

func queryCS(net, host, service string) (res []string, err error) {
	switch net {
	case "tcp4", "tcp6":
		net = "tcp"
	case "udp4", "udp6":
		net = "udp"
	}
	if host == "" {
		host = "*"
	}
	return query(netdir+"/cs", net+"!"+host+"!"+service, 128)
}

func queryCS1(net string, ip IP, port int) (clone, dest string, err error) {
	ips := "*"
	if len(ip) != 0 && !ip.IsUnspecified() {
		ips = ip.String()
	}
	lines, err := queryCS(net, ips, itoa(port))
	if err != nil {
		return
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return "", "", errors.New("bad response from ndb/cs")
	}
	clone, dest = f[0], f[1]
	return
}

func queryDNS(addr string, typ string) (res []string, err error) {
	return query(netdir+"/dns", addr+" "+typ, 1024)
}

// toLower returns a lower-case version of in. Restricting us to
// ASCII is sufficient to handle the IP protocol names and allow
// us to not depend on the strings and unicode packages.
func toLower(in string) string {
	for _, c := range in {
		if 'A' <= c && c <= 'Z' {
			// Has upper case; need to fix.
			out := []byte(in)
			for i := 0; i < len(in); i++ {
				c := in[i]
				if 'A' <= c && c <= 'Z' {
					c += 'a' - 'A'
				}
				out[i] = c
			}
			return string(out)
		}
	}
	return in
}

// lookupProtocol looks up IP protocol name and returns
// the corresponding protocol number.
func lookupProtocol(name string) (proto int, err error) {
	lines, err := query(netdir+"/cs", "!protocol="+toLower(name), 128)
	if err != nil {
		return 0, err
	}
	if len(lines) == 0 {
		return 0, UnknownNetworkError(name)
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return 0, UnknownNetworkError(name)
	}
	s := f[1]
	if n, _, ok := dtoi(s, byteIndex(s, '=')+1); ok {
		return n, nil
	}
	return 0, UnknownNetworkError(name)
}

func lookupHost(host string) (addrs []string, err error) {
	// Use netdir/cs instead of netdir/dns because cs knows about
	// host names in local network (e.g. from /lib/ndb/local)
	lines, err := queryCS("net", host, "1")
	if err != nil {
		return
	}
loop:
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 2 {
			continue
		}
		addr := f[1]
		if i := byteIndex(addr, '!'); i >= 0 {
			addr = addr[:i] // remove port
		}
		if ParseIP(addr) == nil {
			continue
		}
		// only return unique addresses
		for _, a := range addrs {
			if a == addr {
				continue loop
			}
		}
		addrs = append(addrs, addr)
	}
	return
}

func lookupIP(host string) (addrs []IPAddr, err error) {
	lits, err := LookupHost(host)
	if err != nil {
		return
	}
	for _, lit := range lits {
		host, zone := splitHostZone(lit)
		if ip := ParseIP(host); ip != nil {
			addr := IPAddr{IP: ip, Zone: zone}
			addrs = append(addrs, addr)
		}
	}
	return
}

func lookupPort(network, service string) (port int, err error) {
	switch network {
	case "tcp4", "tcp6":
		network = "tcp"
	case "udp4", "udp6":
		network = "udp"
	}
	lines, err := queryCS(network, "127.0.0.1", service)
	if err != nil {
		return
	}
	unknownPortError := &AddrError{Err: "unknown port", Addr: network + "/" + service}
	if len(lines) == 0 {
		return 0, unknownPortError
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return 0, unknownPortError
	}
	s := f[1]
	if i := byteIndex(s, '!'); i >= 0 {
		s = s[i+1:] // remove address
	}
	if n, _, ok := dtoi(s, 0); ok {
		return n, nil
	}
	return 0, unknownPortError
}

func lookupCNAME(name string) (cname string, err error) {
	lines, err := queryDNS(name, "cname")
	if err != nil {
		return
	}
	if len(lines) > 0 {
		if f := getFields(lines[0]); len(f) >= 3 {
			return f[2] + ".", nil
		}
	}
	return "", errors.New("bad response from ndb/dns")
}

func lookupSRV(service, proto, name string) (cname string, addrs []*SRV, err error) {
	var target string
	if service == "" && proto == "" {
		target = name
	} else {
		target = "_" + service + "._" + proto + "." + name
	}
	lines, err := queryDNS(target, "srv")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 6 {
			continue
		}
		port, _, portOk := dtoi(f[4], 0)
		priority, _, priorityOk := dtoi(f[3], 0)
		weight, _, weightOk := dtoi(f[2], 0)
		if !(portOk && priorityOk && weightOk) {
			continue
		}
		addrs = append(addrs, &SRV{f[5], uint16(port), uint16(priority), uint16(weight)})
		cname = f[0]
	}
	byPriorityWeight(addrs).sort()
	return
}

func lookupMX(name string) (mx []*MX, err error) {
	lines, err := queryDNS(name, "mx")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 4 {
			continue
		}
		if pref, _, ok := dtoi(f[2], 0); ok {
			mx = append(mx, &MX{f[3], uint16(pref)})
		}
	}
	byPref(mx).sort()
	return
}

func lookupNS(name string) (ns []*NS, err error) {
	lines, err := queryDNS(name, "ns")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 3 {
			continue
		}
		ns = append(ns, &NS{f[2]})
	}
	return
}

func lookupTXT(name string) (txt []string, err error) {
	lines, err := queryDNS(name, "txt")
	if err != nil {
		return
	}
	for _, line := range lines {
		if i := byteIndex(line, '\t'); i >= 0 {
			txt = append(txt, line[i+1:])
		}
	}
	return
}

func lookupAddr(addr string) (name []string, err error) {
	arpa, err := reverseaddr(addr)
	if err != nil {
		return
	}
	lines, err := queryDNS(arpa, "ptr")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 3 {
			continue
		}
		name = append(name, f[2])
	}
	return
}
