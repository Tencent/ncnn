// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"sync"
)

func ResetLocalOnceForTest() {
	localOnce = sync.Once{}
	localLoc = Location{}
}

func ForceUSPacificForTesting() {
	ResetLocalOnceForTest()
	localOnce.Do(initTestingZone)
}

var (
	ForceZipFileForTesting = forceZipFileForTesting
	ParseTimeZone          = parseTimeZone
)
