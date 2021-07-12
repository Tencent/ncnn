// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"testing"
	"time"
)

func TestVersion3(t *testing.T) {
	t.Skip("gccgo does not use the zip file")
	time.ForceZipFileForTesting(true)
	defer time.ForceZipFileForTesting(false)
	_, err := time.LoadLocation("Asia/Jerusalem")
	if err != nil {
		t.Fatal(err)
	}
}

// Test that we get the correct results for times before the first
// transition time.  To do this we explicitly check early dates in a
// couple of specific timezones.
func TestFirstZone(t *testing.T) {
	t.Skip("gccgo does not use the zip file")

	time.ForceZipFileForTesting(true)
	defer time.ForceZipFileForTesting(false)

	const format = "Mon, 02 Jan 2006 15:04:05 -0700 (MST)"
	var tests = []struct {
		zone  string
		unix  int64
		want1 string
		want2 string
	}{
		{
			"PST8PDT",
			-1633269601,
			"Sun, 31 Mar 1918 01:59:59 -0800 (PST)",
			"Sun, 31 Mar 1918 03:00:00 -0700 (PDT)",
		},
		{
			"Pacific/Fakaofo",
			1325242799,
			"Thu, 29 Dec 2011 23:59:59 -1100 (TKT)",
			"Sat, 31 Dec 2011 00:00:00 +1300 (TKT)",
		},
	}

	for _, test := range tests {
		z, err := time.LoadLocation(test.zone)
		if err != nil {
			t.Fatal(err)
		}
		s := time.Unix(test.unix, 0).In(z).Format(format)
		if s != test.want1 {
			t.Errorf("for %s %d got %q want %q", test.zone, test.unix, s, test.want1)
		}
		s = time.Unix(test.unix+1, 0).In(z).Format(format)
		if s != test.want2 {
			t.Errorf("for %s %d got %q want %q", test.zone, test.unix, s, test.want2)
		}
	}
}
