// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import "syscall"

//go:generate go run ../../../syscall/mksyscall_windows.go -output zsyscall_windows.go syscall_windows.go

const GAA_FLAG_INCLUDE_PREFIX = 0x00000010

const IF_TYPE_SOFTWARE_LOOPBACK = 24

type SocketAddress struct {
	Sockaddr       *syscall.RawSockaddrAny
	SockaddrLength int32
}

type IpAdapterUnicastAddress struct {
	Length             uint32
	Flags              uint32
	Next               *IpAdapterUnicastAddress
	Address            SocketAddress
	PrefixOrigin       int32
	SuffixOrigin       int32
	DadState           int32
	ValidLifetime      uint32
	PreferredLifetime  uint32
	LeaseLifetime      uint32
	OnLinkPrefixLength uint8
}

type IpAdapterAnycastAddress struct {
	Length  uint32
	Flags   uint32
	Next    *IpAdapterAnycastAddress
	Address SocketAddress
}

type IpAdapterMulticastAddress struct {
	Length  uint32
	Flags   uint32
	Next    *IpAdapterMulticastAddress
	Address SocketAddress
}

type IpAdapterDnsServerAdapter struct {
	Length   uint32
	Reserved uint32
	Next     *IpAdapterDnsServerAdapter
	Address  SocketAddress
}

type IpAdapterPrefix struct {
	Length       uint32
	Flags        uint32
	Next         *IpAdapterPrefix
	Address      SocketAddress
	PrefixLength uint32
}

type IpAdapterAddresses struct {
	Length                uint32
	IfIndex               uint32
	Next                  *IpAdapterAddresses
	AdapterName           *byte
	FirstUnicastAddress   *IpAdapterUnicastAddress
	FirstAnycastAddress   *IpAdapterAnycastAddress
	FirstMulticastAddress *IpAdapterMulticastAddress
	FirstDnsServerAddress *IpAdapterDnsServerAdapter
	DnsSuffix             *uint16
	Description           *uint16
	FriendlyName          *uint16
	PhysicalAddress       [syscall.MAX_ADAPTER_ADDRESS_LENGTH]byte
	PhysicalAddressLength uint32
	Flags                 uint32
	Mtu                   uint32
	IfType                uint32
	OperStatus            uint32
	Ipv6IfIndex           uint32
	ZoneIndices           [16]uint32
	FirstPrefix           *IpAdapterPrefix
	/* more fields might be present here. */
}

const (
	IfOperStatusUp             = 1
	IfOperStatusDown           = 2
	IfOperStatusTesting        = 3
	IfOperStatusUnknown        = 4
	IfOperStatusDormant        = 5
	IfOperStatusNotPresent     = 6
	IfOperStatusLowerLayerDown = 7
)

//sys GetAdaptersAddresses(family uint32, flags uint32, reserved uintptr, adapterAddresses *IpAdapterAddresses, sizeOfPointer *uint32) (errcode error) = iphlpapi.GetAdaptersAddresses
//sys	GetComputerNameEx(nameformat uint32, buf *uint16, n *uint32) (err error) = GetComputerNameExW
//sys	MoveFileEx(from *uint16, to *uint16, flags uint32) (err error) = MoveFileExW

const (
	ComputerNameNetBIOS                   = 0
	ComputerNameDnsHostname               = 1
	ComputerNameDnsDomain                 = 2
	ComputerNameDnsFullyQualified         = 3
	ComputerNamePhysicalNetBIOS           = 4
	ComputerNamePhysicalDnsHostname       = 5
	ComputerNamePhysicalDnsDomain         = 6
	ComputerNamePhysicalDnsFullyQualified = 7
	ComputerNameMax                       = 8

	MOVEFILE_REPLACE_EXISTING      = 0x1
	MOVEFILE_COPY_ALLOWED          = 0x2
	MOVEFILE_DELAY_UNTIL_REBOOT    = 0x4
	MOVEFILE_WRITE_THROUGH         = 0x8
	MOVEFILE_CREATE_HARDLINK       = 0x10
	MOVEFILE_FAIL_IF_NOT_TRACKABLE = 0x20
)

func Rename(oldpath, newpath string) error {
	from, err := syscall.UTF16PtrFromString(oldpath)
	if err != nil {
		return err
	}
	to, err := syscall.UTF16PtrFromString(newpath)
	if err != nil {
		return err
	}
	return MoveFileEx(from, to, MOVEFILE_REPLACE_EXISTING)
}
