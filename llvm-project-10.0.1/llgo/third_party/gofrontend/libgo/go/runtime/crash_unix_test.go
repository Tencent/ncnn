// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package runtime_test

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
)

func TestCrashDumpsAllThreads(t *testing.T) {
	switch runtime.GOOS {
	case "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
	default:
		t.Skipf("skipping; not supported on %v", runtime.GOOS)
	}

	// We don't use executeTest because we need to kill the
	// program while it is running.

	testenv.MustHaveGoBuild(t)

	checkStaleRuntime(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := ioutil.WriteFile(filepath.Join(dir, "main.go"), []byte(crashDumpsAllThreadsSource), 0666); err != nil {
		t.Fatalf("failed to create Go file: %v", err)
	}

	cmd := exec.Command("go", "build", "-o", "a.exe")
	cmd.Dir = dir
	out, err := testEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source: %v\n%s", err, out)
	}

	cmd = exec.Command(filepath.Join(dir, "a.exe"))
	cmd = testEnv(cmd)
	cmd.Env = append(cmd.Env, "GOTRACEBACK=crash")
	var outbuf bytes.Buffer
	cmd.Stdout = &outbuf
	cmd.Stderr = &outbuf

	rp, wp, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.ExtraFiles = []*os.File{wp}

	if err := cmd.Start(); err != nil {
		t.Fatalf("starting program: %v", err)
	}

	if err := wp.Close(); err != nil {
		t.Logf("closing write pipe: %v", err)
	}
	if _, err := rp.Read(make([]byte, 1)); err != nil {
		t.Fatalf("reading from pipe: %v", err)
	}

	if err := cmd.Process.Signal(syscall.SIGQUIT); err != nil {
		t.Fatalf("signal: %v", err)
	}

	// No point in checking the error return from Wait--we expect
	// it to fail.
	cmd.Wait()

	// We want to see a stack trace for each thread.
	// Before https://golang.org/cl/2811 running threads would say
	// "goroutine running on other thread; stack unavailable".
	out = outbuf.Bytes()
	n := bytes.Count(out, []byte("main.loop("))
	if n != 4 {
		t.Errorf("found %d instances of main.loop; expected 4", n)
		t.Logf("%s", out)
	}
}

const crashDumpsAllThreadsSource = `
package main

import (
	"fmt"
	"os"
	"runtime"
)

func main() {
	const count = 4
	runtime.GOMAXPROCS(count + 1)

	chans := make([]chan bool, count)
	for i := range chans {
		chans[i] = make(chan bool)
		go loop(i, chans[i])
	}

	// Wait for all the goroutines to start executing.
	for _, c := range chans {
		<-c
	}

	// Tell our parent that all the goroutines are executing.
	if _, err := os.NewFile(3, "pipe").WriteString("x"); err != nil {
		fmt.Fprintf(os.Stderr, "write to pipe failed: %v\n", err)
		os.Exit(2)
	}

	select {}
}

func loop(i int, c chan bool) {
	close(c)
	for {
		for j := 0; j < 0x7fffffff; j++ {
		}
	}
}
`
