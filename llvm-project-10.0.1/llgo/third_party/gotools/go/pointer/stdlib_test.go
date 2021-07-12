// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file runs the pointer analysis on all packages and tests beneath
// $GOROOT.  It provides a "smoke test" that the analysis doesn't crash
// on a large input, and a benchmark for performance measurement.
//
// Because it is relatively slow, the --stdlib flag must be enabled for
// this test to run:
//    % go test -v golang.org/x/tools/go/pointer --stdlib

import (
	"flag"
	"go/build"
	"go/token"
	"testing"
	"time"

	"llvm.org/llgo/third_party/gotools/go/buildutil"
	"llvm.org/llgo/third_party/gotools/go/loader"
	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/ssa/ssautil"
)

var runStdlibTest = flag.Bool("stdlib", false, "Run the (slow) stdlib test")

func TestStdlib(t *testing.T) {
	if !*runStdlibTest {
		t.Skip("skipping (slow) stdlib test (use --stdlib)")
	}

	// Load, parse and type-check the program.
	ctxt := build.Default // copy
	ctxt.GOPATH = ""      // disable GOPATH
	conf := loader.Config{Build: &ctxt}
	if _, err := conf.FromArgs(buildutil.AllPackages(conf.Build), true); err != nil {
		t.Errorf("FromArgs failed: %v", err)
		return
	}

	iprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Create SSA packages.
	prog := ssa.Create(iprog, 0)
	prog.BuildAll()

	numPkgs := len(prog.AllPackages())
	if want := 240; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	// Determine the set of packages/tests to analyze.
	var testPkgs []*ssa.Package
	for _, info := range iprog.InitialPackages() {
		testPkgs = append(testPkgs, prog.Package(info.Pkg))
	}
	testmain := prog.CreateTestMainPackage(testPkgs...)
	if testmain == nil {
		t.Fatal("analysis scope has tests")
	}

	// Run the analysis.
	config := &Config{
		Reflection:     false, // TODO(adonovan): fix remaining bug in rVCallConstraint, then enable.
		BuildCallGraph: true,
		Mains:          []*ssa.Package{testmain},
	}
	// TODO(adonovan): add some query values (affects track bits).

	t0 := time.Now()

	result, err := Analyze(config)
	if err != nil {
		t.Fatal(err) // internal error in pointer analysis
	}
	_ = result // TODO(adonovan): measure something

	t1 := time.Now()

	// Dump some statistics.
	allFuncs := ssautil.AllFunctions(prog)
	var numInstrs int
	for fn := range allFuncs {
		for _, b := range fn.Blocks {
			numInstrs += len(b.Instrs)
		}
	}

	// determine line count
	var lineCount int
	prog.Fset.Iterate(func(f *token.File) bool {
		lineCount += f.LineCount()
		return true
	})

	t.Log("#Source lines:          ", lineCount)
	t.Log("#Instructions:          ", numInstrs)
	t.Log("Pointer analysis:       ", t1.Sub(t0))
}
