// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"container/heap"
	"debug/elf"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

var cmdBuild = &Command{
	UsageLine: "build [-o output] [-i] [build flags] [packages]",
	Short:     "compile packages and dependencies",
	Long: `
Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

If the arguments to build are a list of .go files, build treats
them as a list of source files specifying a single package.

When compiling a single main package, build writes
the resulting executable to an output file named after
the first source file ('go build ed.go rx.go' writes 'ed' or 'ed.exe')
or the source code directory ('go build unix/sam' writes 'sam' or 'sam.exe').
The '.exe' suffix is added when writing a Windows executable.

When compiling multiple packages or a single non-main package,
build compiles the packages but discards the resulting object,
serving only as a check that the packages can be built.

The -o flag, only allowed when compiling a single package,
forces build to write the resulting executable or object
to the named output file, instead of the default behavior described
in the last two paragraphs.

The -i flag installs the packages that are dependencies of the target.

The build flags are shared by the build, clean, get, install, list, run,
and test commands:

	-a
		force rebuilding of packages that are already up-to-date.
	-n
		print the commands but do not run them.
	-p n
		the number of builds that can be run in parallel.
		The default is the number of CPUs available, except
		on darwin/arm which defaults to 1.
	-race
		enable data race detection.
		Supported only on linux/amd64, freebsd/amd64, darwin/amd64 and windows/amd64.
	-v
		print the names of packages as they are compiled.
	-work
		print the name of the temporary work directory and
		do not delete it when exiting.
	-x
		print the commands.

	-asmflags 'flag list'
		arguments to pass on each go tool asm invocation.
	-buildmode mode
		build mode to use. See 'go help buildmode' for more.
	-compiler name
		name of compiler to use, as in runtime.Compiler (gccgo or gc).
	-gccgoflags 'arg list'
		arguments to pass on each gccgo compiler/linker invocation.
	-gcflags 'arg list'
		arguments to pass on each go tool compile invocation.
	-installsuffix suffix
		a suffix to use in the name of the package installation directory,
		in order to keep output separate from default builds.
		If using the -race flag, the install suffix is automatically set to race
		or, if set explicitly, has _race appended to it.  Using a -buildmode
		option that requires non-default compile flags has a similar effect.
	-ldflags 'flag list'
		arguments to pass on each go tool link invocation.
	-linkshared
		link against shared libraries previously created with
		-buildmode=shared
	-pkgdir dir
		install and load all packages from dir instead of the usual locations.
		For example, when building with a non-standard configuration,
		use -pkgdir to keep generated packages in a separate location.
	-tags 'tag list'
		a list of build tags to consider satisfied during the build.
		For more information about build tags, see the description of
		build constraints in the documentation for the go/build package.
	-toolexec 'cmd args'
		a program to use to invoke toolchain programs like vet and asm.
		For example, instead of running asm, the go command will run
		'cmd args /path/to/asm <arguments for asm>'.

The list flags accept a space-separated list of strings. To embed spaces
in an element in the list, surround it with either single or double quotes.

For more about specifying packages, see 'go help packages'.
For more about where packages and binaries are installed,
run 'go help gopath'.
For more about calling between Go and C/C++, run 'go help c'.

Note: Build adheres to certain conventions such as those described
by 'go help gopath'. Not all projects can follow these conventions,
however. Installations that have their own conventions or that use
a separate software build system may choose to use lower-level
invocations such as 'go tool compile' and 'go tool link' to avoid
some of the overheads and design decisions of the build tool.

See also: go install, go get, go clean.
	`,
}

func init() {
	// break init cycle
	cmdBuild.Run = runBuild
	cmdInstall.Run = runInstall

	cmdBuild.Flag.BoolVar(&buildI, "i", false, "")

	addBuildFlags(cmdBuild)
	addBuildFlags(cmdInstall)

	if buildContext.GOOS == "darwin" {
		switch buildContext.GOARCH {
		case "arm", "arm64":
			// darwin/arm cannot run multiple tests simultaneously.
			// Parallelism is limited in go_darwin_arm_exec, but
			// also needs to be limited here so go test std does not
			// timeout tests that waiting to run.
			buildP = 1
		}
	}
}

// Flags set by multiple commands.
var buildA bool               // -a flag
var buildN bool               // -n flag
var buildP = runtime.NumCPU() // -p flag
var buildV bool               // -v flag
var buildX bool               // -x flag
var buildI bool               // -i flag
var buildO = cmdBuild.Flag.String("o", "", "output file")
var buildWork bool           // -work flag
var buildAsmflags []string   // -asmflags flag
var buildGcflags []string    // -gcflags flag
var buildLdflags []string    // -ldflags flag
var buildGccgoflags []string // -gccgoflags flag
var buildRace bool           // -race flag
var buildToolExec []string   // -toolexec flag
var buildBuildmode string    // -buildmode flag
var buildLinkshared bool     // -linkshared flag
var buildPkgdir string       // -pkgdir flag

// Require the source for go std packages
var reqStdPkgSrc bool
var buildContext = build.Default
var buildToolchain toolchain = noToolchain{}
var ldBuildmode string

// buildCompiler implements flag.Var.
// It implements Set by updating both
// buildToolchain and buildContext.Compiler.
type buildCompiler struct{}

func (c buildCompiler) Set(value string) error {
	switch value {
	case "gc":
		buildToolchain = gcToolchain{}
	case "gccgo":
		buildToolchain = gccgoToolchain{}
	default:
		return fmt.Errorf("unknown compiler %q", value)
	}
	buildContext.Compiler = value
	return nil
}

func (c buildCompiler) String() string {
	return buildContext.Compiler
}

func init() {
	switch build.Default.Compiler {
	case "gc":
		buildToolchain = gcToolchain{}
	case "gccgo":
		buildToolchain = gccgoToolchain{}
	}
}

// addBuildFlags adds the flags common to the build, clean, get,
// install, list, run, and test commands.
func addBuildFlags(cmd *Command) {
	cmd.Flag.BoolVar(&buildA, "a", false, "")
	cmd.Flag.BoolVar(&buildN, "n", false, "")
	cmd.Flag.IntVar(&buildP, "p", buildP, "")
	cmd.Flag.BoolVar(&buildV, "v", false, "")
	cmd.Flag.BoolVar(&buildX, "x", false, "")

	cmd.Flag.Var((*stringsFlag)(&buildAsmflags), "asmflags", "")
	cmd.Flag.Var(buildCompiler{}, "compiler", "")
	cmd.Flag.StringVar(&buildBuildmode, "buildmode", "default", "")
	cmd.Flag.Var((*stringsFlag)(&buildGcflags), "gcflags", "")
	cmd.Flag.Var((*stringsFlag)(&buildGccgoflags), "gccgoflags", "")
	cmd.Flag.StringVar(&buildContext.InstallSuffix, "installsuffix", "", "")
	cmd.Flag.Var((*stringsFlag)(&buildLdflags), "ldflags", "")
	cmd.Flag.BoolVar(&buildLinkshared, "linkshared", false, "")
	cmd.Flag.StringVar(&buildPkgdir, "pkgdir", "", "")
	cmd.Flag.BoolVar(&buildRace, "race", false, "")
	cmd.Flag.Var((*stringsFlag)(&buildContext.BuildTags), "tags", "")
	cmd.Flag.Var((*stringsFlag)(&buildToolExec), "toolexec", "")
	cmd.Flag.BoolVar(&buildWork, "work", false, "")
	switch build.Default.Compiler {
	case "gc":
		reqStdPkgSrc = true
	case "gccgo":
		reqStdPkgSrc = false
	}
}

func addBuildFlagsNX(cmd *Command) {
	cmd.Flag.BoolVar(&buildN, "n", false, "")
	cmd.Flag.BoolVar(&buildX, "x", false, "")
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

// fileExtSplit expects a filename and returns the name
// and ext (without the dot). If the file has no
// extension, ext will be empty.
func fileExtSplit(file string) (name, ext string) {
	dotExt := filepath.Ext(file)
	name = file[:len(file)-len(dotExt)]
	if dotExt != "" {
		ext = dotExt[1:]
	}
	return
}

type stringsFlag []string

func (v *stringsFlag) Set(s string) error {
	var err error
	*v, err = splitQuotedFields(s)
	if *v == nil {
		*v = []string{}
	}
	return err
}

func splitQuotedFields(s string) ([]string, error) {
	// Split fields allowing '' or "" around elements.
	// Quotes further inside the string do not count.
	var f []string
	for len(s) > 0 {
		for len(s) > 0 && isSpaceByte(s[0]) {
			s = s[1:]
		}
		if len(s) == 0 {
			break
		}
		// Accepted quoted string. No unescaping inside.
		if s[0] == '"' || s[0] == '\'' {
			quote := s[0]
			s = s[1:]
			i := 0
			for i < len(s) && s[i] != quote {
				i++
			}
			if i >= len(s) {
				return nil, fmt.Errorf("unterminated %c string", quote)
			}
			f = append(f, s[:i])
			s = s[i+1:]
			continue
		}
		i := 0
		for i < len(s) && !isSpaceByte(s[i]) {
			i++
		}
		f = append(f, s[:i])
		s = s[i:]
	}
	return f, nil
}

func (v *stringsFlag) String() string {
	return "<stringsFlag>"
}

func pkgsMain(pkgs []*Package) (res []*Package) {
	for _, p := range pkgs {
		if p.Name == "main" {
			res = append(res, p)
		}
	}
	return res
}

func pkgsNotMain(pkgs []*Package) (res []*Package) {
	for _, p := range pkgs {
		if p.Name != "main" {
			res = append(res, p)
		}
	}
	return res
}

var pkgsFilter = func(pkgs []*Package) []*Package { return pkgs }

func buildModeInit() {
	_, gccgo := buildToolchain.(gccgoToolchain)
	var codegenArg string
	platform := goos + "/" + goarch
	switch buildBuildmode {
	case "archive":
		pkgsFilter = pkgsNotMain
	case "c-archive":
		pkgsFilter = func(p []*Package) []*Package {
			if len(p) != 1 || p[0].Name != "main" {
				fatalf("-buildmode=c-archive requires exactly one main package")
			}
			return p
		}
		exeSuffix = ".a"
		ldBuildmode = "c-archive"
	case "c-shared":
		pkgsFilter = pkgsMain
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			switch platform {
			case "linux/amd64":
				codegenArg = "-shared"
			case "linux/arm":
				buildAsmflags = append(buildAsmflags, "-shared")
			case "darwin/amd64":
			case "android/arm":
			default:
				fatalf("-buildmode=c-shared not supported on %s\n", platform)
			}
		}
		ldBuildmode = "c-shared"
	case "default":
		ldBuildmode = "exe"
	case "exe":
		pkgsFilter = pkgsMain
		ldBuildmode = "exe"
	case "shared":
		pkgsFilter = pkgsNotMain
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			switch platform {
			case "linux/amd64":
			default:
				fatalf("-buildmode=shared not supported on %s\n", platform)
			}
			codegenArg = "-dynlink"
		}
		if *buildO != "" {
			fatalf("-buildmode=shared and -o not supported together")
		}
		ldBuildmode = "shared"
	default:
		fatalf("buildmode=%s not supported", buildBuildmode)
	}
	if buildLinkshared {
		if gccgo {
			codegenArg = "-fPIC"
		} else {
			if platform != "linux/amd64" {
				fmt.Fprintf(os.Stderr, "go %s: -linkshared is only supported on linux/amd64\n", flag.Args()[0])
				os.Exit(2)
			}
			codegenArg = "-dynlink"
			// TODO(mwhudson): remove -w when that gets fixed in linker.
			buildLdflags = append(buildLdflags, "-linkshared", "-w")
		}
	}
	if codegenArg != "" {
		if gccgo {
			buildGccgoflags = append(buildGccgoflags, codegenArg)
		} else {
			buildAsmflags = append(buildAsmflags, codegenArg)
			buildGcflags = append(buildGcflags, codegenArg)
		}
		if buildContext.InstallSuffix != "" {
			buildContext.InstallSuffix += "_"
		}
		buildContext.InstallSuffix += codegenArg[1:]
	}
}

func runBuild(cmd *Command, args []string) {
	raceInit()
	buildModeInit()
	var b builder
	b.init()

	pkgs := packagesForBuild(args)

	if len(pkgs) == 1 && pkgs[0].Name == "main" && *buildO == "" {
		_, *buildO = path.Split(pkgs[0].ImportPath)
		*buildO += exeSuffix
	}

	// sanity check some often mis-used options
	switch buildContext.Compiler {
	case "gccgo":
		if len(buildGcflags) != 0 {
			fmt.Println("go build: when using gccgo toolchain, please pass compiler flags using -gccgoflags, not -gcflags")
		}
		if len(buildLdflags) != 0 {
			fmt.Println("go build: when using gccgo toolchain, please pass linker flags using -gccgoflags, not -ldflags")
		}
	case "gc":
		if len(buildGccgoflags) != 0 {
			fmt.Println("go build: when using gc toolchain, please pass compile flags using -gcflags, and linker flags using -ldflags")
		}
	}

	depMode := modeBuild
	if buildI {
		depMode = modeInstall
	}

	if *buildO != "" {
		if len(pkgs) > 1 {
			fatalf("go build: cannot use -o with multiple packages")
		} else if len(pkgs) == 0 {
			fatalf("no packages to build")
		}
		p := pkgs[0]
		p.target = *buildO
		p.Stale = true // must build - not up to date
		a := b.action(modeInstall, depMode, p)
		b.do(a)
		return
	}

	var a *action
	if buildBuildmode == "shared" {
		a = b.libaction(libname(args), pkgsFilter(packages(args)), modeBuild, depMode)
	} else {
		a = &action{}
		for _, p := range pkgsFilter(packages(args)) {
			a.deps = append(a.deps, b.action(modeBuild, depMode, p))
		}
	}
	b.do(a)
}

var cmdInstall = &Command{
	UsageLine: "install [build flags] [packages]",
	Short:     "compile and install packages and dependencies",
	Long: `
Install compiles and installs the packages named by the import paths,
along with their dependencies.

For more about the build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build, go get, go clean.
	`,
}

// libname returns the filename to use for the shared library when using
// -buildmode=shared.  The rules we use are:
//  1) Drop any trailing "/..."s if present
//  2) Change / to -
//  3) Join arguments with ,
// So std -> libstd.so
//    a b/... -> liba,b.so
//    gopkg.in/tomb.v2 -> libgopkg.in-tomb.v2.so
func libname(args []string) string {
	var libname string
	for _, arg := range args {
		arg = strings.TrimSuffix(arg, "/...")
		arg = strings.Replace(arg, "/", "-", -1)
		if libname == "" {
			libname = arg
		} else {
			libname += "," + arg
		}
	}
	// TODO(mwhudson): Needs to change for platforms that use different naming
	// conventions...
	return "lib" + libname + ".so"
}

func runInstall(cmd *Command, args []string) {
	raceInit()
	buildModeInit()
	pkgs := pkgsFilter(packagesForBuild(args))

	for _, p := range pkgs {
		if p.Target == "" && (!p.Standard || p.ImportPath != "unsafe") {
			switch {
			case p.gobinSubdir:
				errorf("go install: cannot install cross-compiled binaries when GOBIN is set")
			case p.cmdline:
				errorf("go install: no install location for .go files listed on command line (GOBIN not set)")
			case p.ConflictDir != "":
				errorf("go install: no install location for %s: hidden by %s", p.Dir, p.ConflictDir)
			default:
				errorf("go install: no install location for directory %s outside GOPATH\n"+
					"\tFor more details see: go help gopath", p.Dir)
			}
		}
	}
	exitIfErrors()

	var b builder
	b.init()
	var a *action
	if buildBuildmode == "shared" {
		a = b.libaction(libname(args), pkgs, modeInstall, modeInstall)
	} else {
		a = &action{}
		var tools []*action
		for _, p := range pkgs {
			// If p is a tool, delay the installation until the end of the build.
			// This avoids installing assemblers/compilers that are being executed
			// by other steps in the build.
			// cmd/cgo is handled specially in b.action, so that we can
			// both build and use it in the same 'go install'.
			action := b.action(modeInstall, modeInstall, p)
			if goTools[p.ImportPath] == toTool && p.ImportPath != "cmd/cgo" {
				a.deps = append(a.deps, action.deps...)
				action.deps = append(action.deps, a)
				tools = append(tools, action)
				continue
			}
			a.deps = append(a.deps, action)
		}
		if len(tools) > 0 {
			a = &action{
				deps: tools,
			}
		}
	}
	b.do(a)
	exitIfErrors()

	// Success. If this command is 'go install' with no arguments
	// and the current directory (the implicit argument) is a command,
	// remove any leftover command binary from a previous 'go build'.
	// The binary is installed; it's not needed here anymore.
	// And worse it might be a stale copy, which you don't want to find
	// instead of the installed one if $PATH contains dot.
	// One way to view this behavior is that it is as if 'go install' first
	// runs 'go build' and the moves the generated file to the install dir.
	// See issue 9645.
	if len(args) == 0 && len(pkgs) == 1 && pkgs[0].Name == "main" {
		// Compute file 'go build' would have created.
		// If it exists and is an executable file, remove it.
		_, targ := filepath.Split(pkgs[0].ImportPath)
		targ += exeSuffix
		if filepath.Join(pkgs[0].Dir, targ) != pkgs[0].Target { // maybe $GOBIN is the current directory
			fi, err := os.Stat(targ)
			if err == nil {
				m := fi.Mode()
				if m.IsRegular() {
					if m&0111 != 0 || goos == "windows" { // windows never sets executable bit
						os.Remove(targ)
					}
				}
			}
		}
	}
}

// Global build parameters (used during package load)
var (
	goarch    string
	goos      string
	exeSuffix string
)

func init() {
	goarch = buildContext.GOARCH
	goos = buildContext.GOOS
	if goos == "windows" {
		exeSuffix = ".exe"
	}
}

// A builder holds global state about a build.
// It does not hold per-package state, because we
// build packages in parallel, and the builder is shared.
type builder struct {
	work        string               // the temporary work directory (ends in filepath.Separator)
	actionCache map[cacheKey]*action // a cache of already-constructed actions
	mkdirCache  map[string]bool      // a cache of created directories
	print       func(args ...interface{}) (int, error)

	output    sync.Mutex
	scriptDir string // current directory in printed script

	exec      sync.Mutex
	readySema chan bool
	ready     actionQueue
}

// An action represents a single action in the action graph.
type action struct {
	p          *Package      // the package this action works on
	deps       []*action     // actions that must happen before this one
	triggers   []*action     // inverse of deps
	cgo        *action       // action for cgo binary if needed
	args       []string      // additional args for runProgram
	testOutput *bytes.Buffer // test output buffer

	f          func(*builder, *action) error // the action itself (nil = no-op)
	ignoreFail bool                          // whether to run f even if dependencies fail

	// Generated files, directories.
	link   bool   // target is executable, not just package
	pkgdir string // the -I or -L argument to use when importing this package
	objdir string // directory for intermediate objects
	objpkg string // the intermediate package .a file created during the action
	target string // goal of the action: the created package or executable

	// Execution state.
	pending  int  // number of deps yet to complete
	priority int  // relative execution priority
	failed   bool // whether the action failed
}

// cacheKey is the key for the action cache.
type cacheKey struct {
	mode  buildMode
	p     *Package
	shlib string
}

// buildMode specifies the build mode:
// are we just building things or also installing the results?
type buildMode int

const (
	modeBuild buildMode = iota
	modeInstall
)

var (
	goroot    = filepath.Clean(runtime.GOROOT())
	gobin     = os.Getenv("GOBIN")
	gorootBin = filepath.Join(goroot, "bin")
	gorootPkg = filepath.Join(goroot, "pkg")
	gorootSrc = filepath.Join(goroot, "src")
)

func (b *builder) init() {
	var err error
	b.print = func(a ...interface{}) (int, error) {
		return fmt.Fprint(os.Stderr, a...)
	}
	b.actionCache = make(map[cacheKey]*action)
	b.mkdirCache = make(map[string]bool)

	if buildN {
		b.work = "$WORK"
	} else {
		b.work, err = ioutil.TempDir("", "go-build")
		if err != nil {
			fatalf("%s", err)
		}
		if buildX || buildWork {
			fmt.Fprintf(os.Stderr, "WORK=%s\n", b.work)
		}
		if !buildWork {
			workdir := b.work
			atexit(func() { os.RemoveAll(workdir) })
		}
	}
}

// goFilesPackage creates a package for building a collection of Go files
// (typically named on the command line).  The target is named p.a for
// package p or named after the first Go file for package main.
func goFilesPackage(gofiles []string) *Package {
	// TODO: Remove this restriction.
	for _, f := range gofiles {
		if !strings.HasSuffix(f, ".go") {
			fatalf("named files must be .go files")
		}
	}

	var stk importStack
	ctxt := buildContext
	ctxt.UseAllFiles = true

	// Synthesize fake "directory" that only shows the named files,
	// to make it look like this is a standard package or
	// command directory.  So that local imports resolve
	// consistently, the files must all be in the same directory.
	var dirent []os.FileInfo
	var dir string
	for _, file := range gofiles {
		fi, err := os.Stat(file)
		if err != nil {
			fatalf("%s", err)
		}
		if fi.IsDir() {
			fatalf("%s is a directory, should be a Go file", file)
		}
		dir1, _ := filepath.Split(file)
		if dir1 == "" {
			dir1 = "./"
		}
		if dir == "" {
			dir = dir1
		} else if dir != dir1 {
			fatalf("named files must all be in one directory; have %s and %s", dir, dir1)
		}
		dirent = append(dirent, fi)
	}
	ctxt.ReadDir = func(string) ([]os.FileInfo, error) { return dirent, nil }

	var err error
	if dir == "" {
		dir = cwd
	}
	dir, err = filepath.Abs(dir)
	if err != nil {
		fatalf("%s", err)
	}

	bp, err := ctxt.ImportDir(dir, 0)
	pkg := new(Package)
	pkg.local = true
	pkg.cmdline = true
	pkg.load(&stk, bp, err)
	pkg.localPrefix = dirToImportPath(dir)
	pkg.ImportPath = "command-line-arguments"
	pkg.target = ""

	if pkg.Name == "main" {
		_, elem := filepath.Split(gofiles[0])
		exe := elem[:len(elem)-len(".go")] + exeSuffix
		if *buildO == "" {
			*buildO = exe
		}
		if gobin != "" {
			pkg.target = filepath.Join(gobin, exe)
		}
	}

	pkg.Target = pkg.target
	pkg.Stale = true

	computeStale(pkg)
	return pkg
}

// readpkglist returns the list of packages that were built into the shared library
// at shlibpath. For the native toolchain this list is stored, newline separated, in
// an ELF note with name "Go\x00\x00" and type 1. For GCCGO it is extracted from the
// .go_export section.
func readpkglist(shlibpath string) (pkgs []*Package) {
	var stk importStack
	if _, gccgo := buildToolchain.(gccgoToolchain); gccgo {
		f, _ := elf.Open(shlibpath)
		sect := f.Section(".go_export")
		data, _ := sect.Data()
		scanner := bufio.NewScanner(bytes.NewBuffer(data))
		for scanner.Scan() {
			t := scanner.Text()
			if strings.HasPrefix(t, "pkgpath ") {
				t = strings.TrimPrefix(t, "pkgpath ")
				t = strings.TrimSuffix(t, ";")
				pkgs = append(pkgs, loadPackage(t, &stk))
			}
		}
	} else {
		pkglistbytes, err := readELFNote(shlibpath, "Go\x00\x00", 1)
		if err != nil {
			fatalf("readELFNote failed: %v", err)
		}
		scanner := bufio.NewScanner(bytes.NewBuffer(pkglistbytes))
		for scanner.Scan() {
			t := scanner.Text()
			pkgs = append(pkgs, loadPackage(t, &stk))
		}
	}
	return
}

// action returns the action for applying the given operation (mode) to the package.
// depMode is the action to use when building dependencies.
// action never looks for p in a shared library.
func (b *builder) action(mode buildMode, depMode buildMode, p *Package) *action {
	return b.action1(mode, depMode, p, false)
}

// action1 returns the action for applying the given operation (mode) to the package.
// depMode is the action to use when building dependencies.
// action1 will look for p in a shared library if lookshared is true.
func (b *builder) action1(mode buildMode, depMode buildMode, p *Package, lookshared bool) *action {
	shlib := ""
	if lookshared {
		shlib = p.Shlib
	}
	key := cacheKey{mode, p, shlib}

	a := b.actionCache[key]
	if a != nil {
		return a
	}
	if shlib != "" {
		key2 := cacheKey{modeInstall, nil, shlib}
		a = b.actionCache[key2]
		if a != nil {
			b.actionCache[key] = a
			return a
		}
		pkgs := readpkglist(shlib)
		a = b.libaction(filepath.Base(shlib), pkgs, modeInstall, depMode)
		b.actionCache[key2] = a
		b.actionCache[key] = a
		return a
	}

	a = &action{p: p, pkgdir: p.build.PkgRoot}
	if p.pkgdir != "" { // overrides p.t
		a.pkgdir = p.pkgdir
	}
	b.actionCache[key] = a

	for _, p1 := range p.imports {
		ls := buildLinkshared
		// If p1 is part of the same shared library as p, we need the action
		// that builds p here, not the shared libary or we get action loops.
		if p1.Shlib == p.Shlib {
			ls = false
		}
		a.deps = append(a.deps, b.action1(depMode, depMode, p1, ls))
	}

	// If we are not doing a cross-build, then record the binary we'll
	// generate for cgo as a dependency of the build of any package
	// using cgo, to make sure we do not overwrite the binary while
	// a package is using it.  If this is a cross-build, then the cgo we
	// are writing is not the cgo we need to use.
	if goos == runtime.GOOS && goarch == runtime.GOARCH && !buildRace && reqStdPkgSrc {
		if (len(p.CgoFiles) > 0 || p.Standard && p.ImportPath == "runtime/cgo") && !buildLinkshared && buildBuildmode != "shared" {
			var stk importStack
			p1 := loadPackage("cmd/cgo", &stk)
			if p1.Error != nil {
				fatalf("load cmd/cgo: %v", p1.Error)
			}
			a.cgo = b.action(depMode, depMode, p1)
			a.deps = append(a.deps, a.cgo)
		}
	}

	if p.Standard {
		switch p.ImportPath {
		case "builtin", "unsafe":
			// Fake packages - nothing to build.
			return a
		}
		// gccgo standard library is "fake" too.
		if _, ok := buildToolchain.(gccgoToolchain); ok {
			// the target name is needed for cgo.
			a.target = p.target
			return a
		}
	}

	if !p.Stale && p.target != "" {
		// p.Stale==false implies that p.target is up-to-date.
		// Record target name for use by actions depending on this one.
		a.target = p.target
		return a
	}

	if p.local && p.target == "" {
		// Imported via local path.  No permanent target.
		mode = modeBuild
	}
	work := p.pkgdir
	if work == "" {
		work = b.work
	}
	a.objdir = filepath.Join(work, a.p.ImportPath, "_obj") + string(filepath.Separator)
	a.objpkg = buildToolchain.pkgpath(work, a.p)
	a.link = p.Name == "main"

	switch mode {
	case modeInstall:
		a.f = (*builder).install
		a.deps = []*action{b.action1(modeBuild, depMode, p, lookshared)}
		a.target = a.p.target

		// Install header for cgo in c-archive and c-shared modes.
		if p.usesCgo() && (buildBuildmode == "c-archive" || buildBuildmode == "c-shared") {
			ah := &action{
				p:      a.p,
				deps:   []*action{a.deps[0]},
				f:      (*builder).installHeader,
				pkgdir: a.pkgdir,
				objdir: a.objdir,
				target: a.target[:len(a.target)-len(filepath.Ext(a.target))] + ".h",
			}
			a.deps = append(a.deps, ah)
		}

	case modeBuild:
		a.f = (*builder).build
		a.target = a.objpkg
		if a.link {
			// An executable file. (This is the name of a temporary file.)
			// Because we run the temporary file in 'go run' and 'go test',
			// the name will show up in ps listings. If the caller has specified
			// a name, use that instead of a.out. The binary is generated
			// in an otherwise empty subdirectory named exe to avoid
			// naming conflicts.  The only possible conflict is if we were
			// to create a top-level package named exe.
			name := "a.out"
			if p.exeName != "" {
				name = p.exeName
			} else if goos == "darwin" && buildBuildmode == "c-shared" && p.target != "" {
				// On OS X, the linker output name gets recorded in the
				// shared library's LC_ID_DYLIB load command.
				// The code invoking the linker knows to pass only the final
				// path element. Arrange that the path element matches what
				// we'll install it as; otherwise the library is only loadable as "a.out".
				_, name = filepath.Split(p.target)
			}
			a.target = a.objdir + filepath.Join("exe", name) + exeSuffix
		}
	}

	return a
}

func (b *builder) libaction(libname string, pkgs []*Package, mode, depMode buildMode) *action {
	a := &action{}
	if mode == modeBuild {
		a.f = (*builder).linkShared
		a.target = filepath.Join(b.work, libname)
		for _, p := range pkgs {
			if p.target == "" {
				continue
			}
			a.deps = append(a.deps, b.action(depMode, depMode, p))
		}
	} else if mode == modeInstall {
		// Currently build mode shared forces external linking mode, and
		// external linking mode forces an import of runtime/cgo. So if it
		// was not passed on the command line and it is not present in
		// another shared library, add it here.
		seencgo := false
		_, gccgo := buildToolchain.(gccgoToolchain)
		if !gccgo {
			for _, p := range pkgs {
				seencgo = seencgo || (p.Standard && p.ImportPath == "runtime/cgo")
			}
			if !seencgo {
				var stk importStack
				p := loadPackage("runtime/cgo", &stk)
				if p.Error != nil {
					fatalf("load runtime/cgo: %v", p.Error)
				}
				computeStale(p)
				// If runtime/cgo is in another shared library, then that's
				// also the shared library that contains runtime, so
				// something will depend on it and so runtime/cgo's staleness
				// will be checked when processing that library.
				if p.Shlib == "" || p.Shlib == libname {
					pkgs = append([]*Package{}, pkgs...)
					pkgs = append(pkgs, p)
				}
			}
		}

		// Figure out where the library will go.
		var libdir string
		for _, p := range pkgs {
			plibdir := p.build.PkgTargetRoot
			if gccgo {
				plibdir = filepath.Join(plibdir, "shlibs")
			}
			if libdir == "" {
				libdir = plibdir
			} else if libdir != plibdir {
				fatalf("multiple roots %s & %s", libdir, plibdir)
			}
		}
		a.target = filepath.Join(libdir, libname)

		// Now we can check whether we need to rebuild it.
		stale := false
		var built time.Time
		if fi, err := os.Stat(a.target); err == nil {
			built = fi.ModTime()
		}
		for _, p := range pkgs {
			if p.target == "" {
				continue
			}
			stale = stale || p.Stale
			lstat, err := os.Stat(p.target)
			if err != nil || lstat.ModTime().After(built) {
				stale = true
			}
			a.deps = append(a.deps, b.action(depMode, depMode, p))
		}

		if stale {
			a.f = (*builder).install
			buildAction := b.libaction(libname, pkgs, modeBuild, depMode)
			a.deps = []*action{buildAction}
			for _, p := range pkgs {
				if p.target == "" {
					continue
				}
				shlibnameaction := &action{}
				shlibnameaction.f = (*builder).installShlibname
				shlibnameaction.target = p.target[:len(p.target)-2] + ".shlibname"
				a.deps = append(a.deps, shlibnameaction)
				shlibnameaction.deps = append(shlibnameaction.deps, buildAction)
			}
		}
	} else {
		fatalf("unregonized mode %v", mode)
	}
	return a
}

// actionList returns the list of actions in the dag rooted at root
// as visited in a depth-first post-order traversal.
func actionList(root *action) []*action {
	seen := map[*action]bool{}
	all := []*action{}
	var walk func(*action)
	walk = func(a *action) {
		if seen[a] {
			return
		}
		seen[a] = true
		for _, a1 := range a.deps {
			walk(a1)
		}
		all = append(all, a)
	}
	walk(root)
	return all
}

// allArchiveActions returns a list of the archive dependencies of root.
// This is needed because if package p depends on package q that is in libr.so, the
// action graph looks like p->libr.so->q and so just scanning through p's
// dependencies does not find the import dir for q.
func allArchiveActions(root *action) []*action {
	seen := map[*action]bool{}
	r := []*action{}
	var walk func(*action)
	walk = func(a *action) {
		if seen[a] {
			return
		}
		seen[a] = true
		if strings.HasSuffix(a.target, ".so") || a == root {
			for _, a1 := range a.deps {
				walk(a1)
			}
		} else if strings.HasSuffix(a.target, ".a") {
			r = append(r, a)
		}
	}
	walk(root)
	return r
}

// do runs the action graph rooted at root.
func (b *builder) do(root *action) {
	// Build list of all actions, assigning depth-first post-order priority.
	// The original implementation here was a true queue
	// (using a channel) but it had the effect of getting
	// distracted by low-level leaf actions to the detriment
	// of completing higher-level actions.  The order of
	// work does not matter much to overall execution time,
	// but when running "go test std" it is nice to see each test
	// results as soon as possible.  The priorities assigned
	// ensure that, all else being equal, the execution prefers
	// to do what it would have done first in a simple depth-first
	// dependency order traversal.
	all := actionList(root)
	for i, a := range all {
		a.priority = i
	}

	b.readySema = make(chan bool, len(all))

	// Initialize per-action execution state.
	for _, a := range all {
		for _, a1 := range a.deps {
			a1.triggers = append(a1.triggers, a)
		}
		a.pending = len(a.deps)
		if a.pending == 0 {
			b.ready.push(a)
			b.readySema <- true
		}
	}

	// Handle runs a single action and takes care of triggering
	// any actions that are runnable as a result.
	handle := func(a *action) {
		var err error
		if a.f != nil && (!a.failed || a.ignoreFail) {
			err = a.f(b, a)
		}

		// The actions run in parallel but all the updates to the
		// shared work state are serialized through b.exec.
		b.exec.Lock()
		defer b.exec.Unlock()

		if err != nil {
			if err == errPrintedOutput {
				setExitStatus(2)
			} else {
				errorf("%s", err)
			}
			a.failed = true
		}

		for _, a0 := range a.triggers {
			if a.failed {
				a0.failed = true
			}
			if a0.pending--; a0.pending == 0 {
				b.ready.push(a0)
				b.readySema <- true
			}
		}

		if a == root {
			close(b.readySema)
		}
	}

	var wg sync.WaitGroup

	// Kick off goroutines according to parallelism.
	// If we are using the -n flag (just printing commands)
	// drop the parallelism to 1, both to make the output
	// deterministic and because there is no real work anyway.
	par := buildP
	if buildN {
		par = 1
	}
	for i := 0; i < par; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case _, ok := <-b.readySema:
					if !ok {
						return
					}
					// Receiving a value from b.readySema entitles
					// us to take from the ready queue.
					b.exec.Lock()
					a := b.ready.pop()
					b.exec.Unlock()
					handle(a)
				case <-interrupted:
					setExitStatus(1)
					return
				}
			}
		}()
	}

	wg.Wait()
}

// hasString reports whether s appears in the list of strings.
func hasString(strings []string, s string) bool {
	for _, t := range strings {
		if s == t {
			return true
		}
	}
	return false
}

// build is the action for building a single package or command.
func (b *builder) build(a *action) (err error) {
	// Return an error if the package has CXX files but it's not using
	// cgo nor SWIG, since the CXX files can only be processed by cgo
	// and SWIG.
	if len(a.p.CXXFiles) > 0 && !a.p.usesCgo() && !a.p.usesSwig() {
		return fmt.Errorf("can't build package %s because it contains C++ files (%s) but it's not using cgo nor SWIG",
			a.p.ImportPath, strings.Join(a.p.CXXFiles, ","))
	}
	// Same as above for Objective-C files
	if len(a.p.MFiles) > 0 && !a.p.usesCgo() && !a.p.usesSwig() {
		return fmt.Errorf("can't build package %s because it contains Objective-C files (%s) but it's not using cgo nor SWIG",
			a.p.ImportPath, strings.Join(a.p.MFiles, ","))
	}
	defer func() {
		if err != nil && err != errPrintedOutput {
			err = fmt.Errorf("go build %s: %v", a.p.ImportPath, err)
		}
	}()
	if buildN {
		// In -n mode, print a banner between packages.
		// The banner is five lines so that when changes to
		// different sections of the bootstrap script have to
		// be merged, the banners give patch something
		// to use to find its context.
		fmt.Printf("\n#\n# %s\n#\n\n", a.p.ImportPath)
	}

	if buildV {
		fmt.Fprintf(os.Stderr, "%s\n", a.p.ImportPath)
	}

	if a.p.Standard && a.p.ImportPath == "runtime" && buildContext.Compiler == "gc" &&
		(!hasString(a.p.GoFiles, "zgoos_"+buildContext.GOOS+".go") ||
			!hasString(a.p.GoFiles, "zgoarch_"+buildContext.GOARCH+".go")) {
		return fmt.Errorf("%s/%s must be bootstrapped using make%v", buildContext.GOOS, buildContext.GOARCH, defaultSuffix())
	}

	// Make build directory.
	obj := a.objdir
	if err := b.mkdir(obj); err != nil {
		return err
	}

	// make target directory
	dir, _ := filepath.Split(a.target)
	if dir != "" {
		if err := b.mkdir(dir); err != nil {
			return err
		}
	}

	var gofiles, cgofiles, cfiles, sfiles, cxxfiles, objects, cgoObjects, pcCFLAGS, pcLDFLAGS []string

	gofiles = append(gofiles, a.p.GoFiles...)
	cgofiles = append(cgofiles, a.p.CgoFiles...)
	cfiles = append(cfiles, a.p.CFiles...)
	sfiles = append(sfiles, a.p.SFiles...)
	cxxfiles = append(cxxfiles, a.p.CXXFiles...)

	if a.p.usesCgo() || a.p.usesSwig() {
		if pcCFLAGS, pcLDFLAGS, err = b.getPkgConfigFlags(a.p); err != nil {
			return
		}
	}

	// Run SWIG on each .swig and .swigcxx file.
	// Each run will generate two files, a .go file and a .c or .cxx file.
	// The .go file will use import "C" and is to be processed by cgo.
	if a.p.usesSwig() {
		outGo, outC, outCXX, err := b.swig(a.p, obj, pcCFLAGS)
		if err != nil {
			return err
		}
		cgofiles = append(cgofiles, outGo...)
		cfiles = append(cfiles, outC...)
		cxxfiles = append(cxxfiles, outCXX...)
	}

	// Run cgo.
	if a.p.usesCgo() || a.p.usesSwig() {
		// In a package using cgo, cgo compiles the C, C++ and assembly files with gcc.
		// There is one exception: runtime/cgo's job is to bridge the
		// cgo and non-cgo worlds, so it necessarily has files in both.
		// In that case gcc only gets the gcc_* files.
		var gccfiles []string
		if a.p.Standard && a.p.ImportPath == "runtime/cgo" {
			filter := func(files, nongcc, gcc []string) ([]string, []string) {
				for _, f := range files {
					if strings.HasPrefix(f, "gcc_") {
						gcc = append(gcc, f)
					} else {
						nongcc = append(nongcc, f)
					}
				}
				return nongcc, gcc
			}
			cfiles, gccfiles = filter(cfiles, cfiles[:0], gccfiles)
			sfiles, gccfiles = filter(sfiles, sfiles[:0], gccfiles)
		} else {
			gccfiles = append(cfiles, sfiles...)
			cfiles = nil
			sfiles = nil
		}

		cgoExe := tool("cgo")
		if a.cgo != nil && a.cgo.target != "" {
			cgoExe = a.cgo.target
		}
		outGo, outObj, err := b.cgo(a.p, cgoExe, obj, pcCFLAGS, pcLDFLAGS, cgofiles, gccfiles, cxxfiles, a.p.MFiles)
		if err != nil {
			return err
		}
		cgoObjects = append(cgoObjects, outObj...)
		gofiles = append(gofiles, outGo...)
	}

	if len(gofiles) == 0 {
		return &build.NoGoError{Dir: a.p.Dir}
	}

	// If we're doing coverage, preprocess the .go files and put them in the work directory
	if a.p.coverMode != "" {
		for i, file := range gofiles {
			var sourceFile string
			var coverFile string
			var key string
			if strings.HasSuffix(file, ".cgo1.go") {
				// cgo files have absolute paths
				base := filepath.Base(file)
				sourceFile = file
				coverFile = filepath.Join(obj, base)
				key = strings.TrimSuffix(base, ".cgo1.go") + ".go"
			} else {
				sourceFile = filepath.Join(a.p.Dir, file)
				coverFile = filepath.Join(obj, file)
				key = file
			}
			cover := a.p.coverVars[key]
			if cover == nil || isTestFile(file) {
				// Not covering this file.
				continue
			}
			if err := b.cover(a, coverFile, sourceFile, 0666, cover.Var); err != nil {
				return err
			}
			gofiles[i] = coverFile
		}
	}

	// Prepare Go import path list.
	inc := b.includeArgs("-I", allArchiveActions(a))

	// Compile Go.
	ofile, out, err := buildToolchain.gc(b, a.p, a.objpkg, obj, len(sfiles) > 0, inc, gofiles)
	if len(out) > 0 {
		b.showOutput(a.p.Dir, a.p.ImportPath, b.processOutput(out))
		if err != nil {
			return errPrintedOutput
		}
	}
	if err != nil {
		return err
	}
	if ofile != a.objpkg {
		objects = append(objects, ofile)
	}

	// Copy .h files named for goos or goarch or goos_goarch
	// to names using GOOS and GOARCH.
	// For example, defs_linux_amd64.h becomes defs_GOOS_GOARCH.h.
	_goos_goarch := "_" + goos + "_" + goarch
	_goos := "_" + goos
	_goarch := "_" + goarch
	for _, file := range a.p.HFiles {
		name, ext := fileExtSplit(file)
		switch {
		case strings.HasSuffix(name, _goos_goarch):
			targ := file[:len(name)-len(_goos_goarch)] + "_GOOS_GOARCH." + ext
			if err := b.copyFile(a, obj+targ, filepath.Join(a.p.Dir, file), 0644, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goarch):
			targ := file[:len(name)-len(_goarch)] + "_GOARCH." + ext
			if err := b.copyFile(a, obj+targ, filepath.Join(a.p.Dir, file), 0644, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goos):
			targ := file[:len(name)-len(_goos)] + "_GOOS." + ext
			if err := b.copyFile(a, obj+targ, filepath.Join(a.p.Dir, file), 0644, true); err != nil {
				return err
			}
		}
	}

	for _, file := range cfiles {
		out := file[:len(file)-len(".c")] + ".o"
		if err := buildToolchain.cc(b, a.p, obj, obj+out, file); err != nil {
			return err
		}
		objects = append(objects, out)
	}

	// Assemble .s files.
	for _, file := range sfiles {
		out := file[:len(file)-len(".s")] + ".o"
		if err := buildToolchain.asm(b, a.p, obj, obj+out, file); err != nil {
			return err
		}
		objects = append(objects, out)
	}

	// NOTE(rsc): On Windows, it is critically important that the
	// gcc-compiled objects (cgoObjects) be listed after the ordinary
	// objects in the archive.  I do not know why this is.
	// https://golang.org/issue/2601
	objects = append(objects, cgoObjects...)

	// Add system object files.
	for _, syso := range a.p.SysoFiles {
		objects = append(objects, filepath.Join(a.p.Dir, syso))
	}

	// Pack into archive in obj directory.
	// If the Go compiler wrote an archive, we only need to add the
	// object files for non-Go sources to the archive.
	// If the Go compiler wrote an archive and the package is entirely
	// Go sources, there is no pack to execute at all.
	if len(objects) > 0 {
		if err := buildToolchain.pack(b, a.p, obj, a.objpkg, objects); err != nil {
			return err
		}
	}

	// Link if needed.
	if a.link {
		// The compiler only cares about direct imports, but the
		// linker needs the whole dependency tree.
		all := actionList(a)
		all = all[:len(all)-1] // drop a
		if err := buildToolchain.ld(b, a, a.target, all, a.objpkg, objects); err != nil {
			return err
		}
	}

	return nil
}

// Calls pkg-config if needed and returns the cflags/ldflags needed to build the package.
func (b *builder) getPkgConfigFlags(p *Package) (cflags, ldflags []string, err error) {
	if pkgs := p.CgoPkgConfig; len(pkgs) > 0 {
		var out []byte
		out, err = b.runOut(p.Dir, p.ImportPath, nil, "pkg-config", "--cflags", pkgs)
		if err != nil {
			b.showOutput(p.Dir, "pkg-config --cflags "+strings.Join(pkgs, " "), string(out))
			b.print(err.Error() + "\n")
			err = errPrintedOutput
			return
		}
		if len(out) > 0 {
			cflags = strings.Fields(string(out))
		}
		out, err = b.runOut(p.Dir, p.ImportPath, nil, "pkg-config", "--libs", pkgs)
		if err != nil {
			b.showOutput(p.Dir, "pkg-config --libs "+strings.Join(pkgs, " "), string(out))
			b.print(err.Error() + "\n")
			err = errPrintedOutput
			return
		}
		if len(out) > 0 {
			ldflags = strings.Fields(string(out))
		}
	}
	return
}

func (b *builder) installShlibname(a *action) error {
	a1 := a.deps[0]
	err := ioutil.WriteFile(a.target, []byte(filepath.Base(a1.target)+"\n"), 0644)
	if err != nil {
		return err
	}
	if buildX {
		b.showcmd("", "echo '%s' > %s # internal", filepath.Base(a1.target), a.target)
	}
	return nil
}

func (b *builder) linkShared(a *action) (err error) {
	allactions := actionList(a)
	allactions = allactions[:len(allactions)-1]
	return buildToolchain.ldShared(b, a.deps, a.target, allactions)
}

// install is the action for installing a single package or executable.
func (b *builder) install(a *action) (err error) {
	defer func() {
		if err != nil && err != errPrintedOutput {
			err = fmt.Errorf("go install %s: %v", a.p.ImportPath, err)
		}
	}()
	a1 := a.deps[0]
	perm := os.FileMode(0644)
	if a1.link {
		switch buildBuildmode {
		case "c-archive", "c-shared":
		default:
			perm = 0755
		}
	}

	// make target directory
	dir, _ := filepath.Split(a.target)
	if dir != "" {
		if err := b.mkdir(dir); err != nil {
			return err
		}
	}

	// remove object dir to keep the amount of
	// garbage down in a large build.  On an operating system
	// with aggressive buffering, cleaning incrementally like
	// this keeps the intermediate objects from hitting the disk.
	if !buildWork {
		defer os.RemoveAll(a1.objdir)
		defer os.Remove(a1.target)
	}

	return b.moveOrCopyFile(a, a.target, a1.target, perm, false)
}

// includeArgs returns the -I or -L directory list for access
// to the results of the list of actions.
func (b *builder) includeArgs(flag string, all []*action) []string {
	inc := []string{}
	incMap := map[string]bool{
		b.work:    true, // handled later
		gorootPkg: true,
		"":        true, // ignore empty strings
	}

	// Look in the temporary space for results of test-specific actions.
	// This is the $WORK/my/package/_test directory for the
	// package being built, so there are few of these.
	for _, a1 := range all {
		if a1.p == nil {
			continue
		}
		if dir := a1.pkgdir; dir != a1.p.build.PkgRoot && !incMap[dir] {
			incMap[dir] = true
			inc = append(inc, flag, dir)
		}
	}

	// Also look in $WORK for any non-test packages that have
	// been built but not installed.
	inc = append(inc, flag, b.work)

	// Finally, look in the installed package directories for each action.
	for _, a1 := range all {
		if a1.p == nil {
			continue
		}
		if dir := a1.pkgdir; dir == a1.p.build.PkgRoot && !incMap[dir] {
			incMap[dir] = true
			inc = append(inc, flag, a1.p.build.PkgTargetRoot)
		}
	}

	return inc
}

// moveOrCopyFile is like 'mv src dst' or 'cp src dst'.
func (b *builder) moveOrCopyFile(a *action, dst, src string, perm os.FileMode, force bool) error {
	if buildN {
		b.showcmd("", "mv %s %s", src, dst)
		return nil
	}

	// If we can update the mode and rename to the dst, do it.
	// Otherwise fall back to standard copy.
	if err := os.Chmod(src, perm); err == nil {
		if err := os.Rename(src, dst); err == nil {
			if buildX {
				b.showcmd("", "mv %s %s", src, dst)
			}
			return nil
		}
	}

	return b.copyFile(a, dst, src, perm, force)
}

// copyFile is like 'cp src dst'.
func (b *builder) copyFile(a *action, dst, src string, perm os.FileMode, force bool) error {
	if buildN || buildX {
		b.showcmd("", "cp %s %s", src, dst)
		if buildN {
			return nil
		}
	}

	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()

	// Be careful about removing/overwriting dst.
	// Do not remove/overwrite if dst exists and is a directory
	// or a non-object file.
	if fi, err := os.Stat(dst); err == nil {
		if fi.IsDir() {
			return fmt.Errorf("build output %q already exists and is a directory", dst)
		}
		if !force && !isObject(dst) {
			return fmt.Errorf("build output %q already exists and is not an object file", dst)
		}
	}

	// On Windows, remove lingering ~ file from last attempt.
	if toolIsWindows {
		if _, err := os.Stat(dst + "~"); err == nil {
			os.Remove(dst + "~")
		}
	}

	os.Remove(dst)
	df, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil && toolIsWindows {
		// Windows does not allow deletion of a binary file
		// while it is executing.  Try to move it out of the way.
		// If the move fails, which is likely, we'll try again the
		// next time we do an install of this binary.
		if err := os.Rename(dst, dst+"~"); err == nil {
			os.Remove(dst + "~")
		}
		df, err = os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	}
	if err != nil {
		return err
	}

	_, err = io.Copy(df, sf)
	df.Close()
	if err != nil {
		os.Remove(dst)
		return fmt.Errorf("copying %s to %s: %v", src, dst, err)
	}
	return nil
}

// Install the cgo export header file, if there is one.
func (b *builder) installHeader(a *action) error {
	src := a.objdir + "_cgo_install.h"
	if _, err := os.Stat(src); os.IsNotExist(err) {
		// If the file does not exist, there are no exported
		// functions, and we do not install anything.
		return nil
	}

	dir, _ := filepath.Split(a.target)
	if dir != "" {
		if err := b.mkdir(dir); err != nil {
			return err
		}
	}

	return b.moveOrCopyFile(a, a.target, src, 0644, true)
}

// cover runs, in effect,
//	go tool cover -mode=b.coverMode -var="varName" -o dst.go src.go
func (b *builder) cover(a *action, dst, src string, perm os.FileMode, varName string) error {
	return b.run(a.objdir, "cover "+a.p.ImportPath, nil,
		buildToolExec,
		tool("cover"),
		"-mode", a.p.coverMode,
		"-var", varName,
		"-o", dst,
		src)
}

var objectMagic = [][]byte{
	{'!', '<', 'a', 'r', 'c', 'h', '>', '\n'}, // Package archive
	{'\x7F', 'E', 'L', 'F'},                   // ELF
	{0xFE, 0xED, 0xFA, 0xCE},                  // Mach-O big-endian 32-bit
	{0xFE, 0xED, 0xFA, 0xCF},                  // Mach-O big-endian 64-bit
	{0xCE, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 32-bit
	{0xCF, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 64-bit
	{0x4d, 0x5a, 0x90, 0x00, 0x03, 0x00},      // PE (Windows) as generated by 6l/8l and gcc
	{0x00, 0x00, 0x01, 0xEB},                  // Plan 9 i386
	{0x00, 0x00, 0x8a, 0x97},                  // Plan 9 amd64
}

func isObject(s string) bool {
	f, err := os.Open(s)
	if err != nil {
		return false
	}
	defer f.Close()
	buf := make([]byte, 64)
	io.ReadFull(f, buf)
	for _, magic := range objectMagic {
		if bytes.HasPrefix(buf, magic) {
			return true
		}
	}
	return false
}

// fmtcmd formats a command in the manner of fmt.Sprintf but also:
//
//	If dir is non-empty and the script is not in dir right now,
//	fmtcmd inserts "cd dir\n" before the command.
//
//	fmtcmd replaces the value of b.work with $WORK.
//	fmtcmd replaces the value of goroot with $GOROOT.
//	fmtcmd replaces the value of b.gobin with $GOBIN.
//
//	fmtcmd replaces the name of the current directory with dot (.)
//	but only when it is at the beginning of a space-separated token.
//
func (b *builder) fmtcmd(dir string, format string, args ...interface{}) string {
	cmd := fmt.Sprintf(format, args...)
	if dir != "" && dir != "/" {
		cmd = strings.Replace(" "+cmd, " "+dir, " .", -1)[1:]
		if b.scriptDir != dir {
			b.scriptDir = dir
			cmd = "cd " + dir + "\n" + cmd
		}
	}
	if b.work != "" {
		cmd = strings.Replace(cmd, b.work, "$WORK", -1)
	}
	return cmd
}

// showcmd prints the given command to standard output
// for the implementation of -n or -x.
func (b *builder) showcmd(dir string, format string, args ...interface{}) {
	b.output.Lock()
	defer b.output.Unlock()
	b.print(b.fmtcmd(dir, format, args...) + "\n")
}

// showOutput prints "# desc" followed by the given output.
// The output is expected to contain references to 'dir', usually
// the source directory for the package that has failed to build.
// showOutput rewrites mentions of dir with a relative path to dir
// when the relative path is shorter.  This is usually more pleasant.
// For example, if fmt doesn't compile and we are in src/html,
// the output is
//
//	$ go build
//	# fmt
//	../fmt/print.go:1090: undefined: asdf
//	$
//
// instead of
//
//	$ go build
//	# fmt
//	/usr/gopher/go/src/fmt/print.go:1090: undefined: asdf
//	$
//
// showOutput also replaces references to the work directory with $WORK.
//
func (b *builder) showOutput(dir, desc, out string) {
	prefix := "# " + desc
	suffix := "\n" + out
	if reldir := shortPath(dir); reldir != dir {
		suffix = strings.Replace(suffix, " "+dir, " "+reldir, -1)
		suffix = strings.Replace(suffix, "\n"+dir, "\n"+reldir, -1)
	}
	suffix = strings.Replace(suffix, " "+b.work, " $WORK", -1)

	b.output.Lock()
	defer b.output.Unlock()
	b.print(prefix, suffix)
}

// shortPath returns an absolute or relative name for path, whatever is shorter.
func shortPath(path string) string {
	if rel, err := filepath.Rel(cwd, path); err == nil && len(rel) < len(path) {
		return rel
	}
	return path
}

// relPaths returns a copy of paths with absolute paths
// made relative to the current directory if they would be shorter.
func relPaths(paths []string) []string {
	var out []string
	pwd, _ := os.Getwd()
	for _, p := range paths {
		rel, err := filepath.Rel(pwd, p)
		if err == nil && len(rel) < len(p) {
			p = rel
		}
		out = append(out, p)
	}
	return out
}

// errPrintedOutput is a special error indicating that a command failed
// but that it generated output as well, and that output has already
// been printed, so there's no point showing 'exit status 1' or whatever
// the wait status was.  The main executor, builder.do, knows not to
// print this error.
var errPrintedOutput = errors.New("already printed output - no need to show error")

var cgoLine = regexp.MustCompile(`\[[^\[\]]+\.cgo1\.go:[0-9]+\]`)
var cgoTypeSigRe = regexp.MustCompile(`\b_Ctype_\B`)

// run runs the command given by cmdline in the directory dir.
// If the command fails, run prints information about the failure
// and returns a non-nil error.
func (b *builder) run(dir string, desc string, env []string, cmdargs ...interface{}) error {
	out, err := b.runOut(dir, desc, env, cmdargs...)
	if len(out) > 0 {
		if desc == "" {
			desc = b.fmtcmd(dir, "%s", strings.Join(stringList(cmdargs...), " "))
		}
		b.showOutput(dir, desc, b.processOutput(out))
		if err != nil {
			err = errPrintedOutput
		}
	}
	return err
}

// processOutput prepares the output of runOut to be output to the console.
func (b *builder) processOutput(out []byte) string {
	if out[len(out)-1] != '\n' {
		out = append(out, '\n')
	}
	messages := string(out)
	// Fix up output referring to cgo-generated code to be more readable.
	// Replace x.go:19[/tmp/.../x.cgo1.go:18] with x.go:19.
	// Replace *[100]_Ctype_foo with *[100]C.foo.
	// If we're using -x, assume we're debugging and want the full dump, so disable the rewrite.
	if !buildX && cgoLine.MatchString(messages) {
		messages = cgoLine.ReplaceAllString(messages, "")
		messages = cgoTypeSigRe.ReplaceAllString(messages, "C.")
	}
	return messages
}

// runOut runs the command given by cmdline in the directory dir.
// It returns the command output and any errors that occurred.
func (b *builder) runOut(dir string, desc string, env []string, cmdargs ...interface{}) ([]byte, error) {
	cmdline := stringList(cmdargs...)
	if buildN || buildX {
		var envcmdline string
		for i := range env {
			envcmdline += env[i]
			envcmdline += " "
		}
		envcmdline += joinUnambiguously(cmdline)
		b.showcmd(dir, "%s", envcmdline)
		if buildN {
			return nil, nil
		}
	}

	nbusy := 0
	for {
		var buf bytes.Buffer
		cmd := exec.Command(cmdline[0], cmdline[1:]...)
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		cmd.Dir = dir
		cmd.Env = mergeEnvLists(env, envForDir(cmd.Dir, os.Environ()))
		err := cmd.Run()

		// cmd.Run will fail on Unix if some other process has the binary
		// we want to run open for writing.  This can happen here because
		// we build and install the cgo command and then run it.
		// If another command was kicked off while we were writing the
		// cgo binary, the child process for that command may be holding
		// a reference to the fd, keeping us from running exec.
		//
		// But, you might reasonably wonder, how can this happen?
		// The cgo fd, like all our fds, is close-on-exec, so that we need
		// not worry about other processes inheriting the fd accidentally.
		// The answer is that running a command is fork and exec.
		// A child forked while the cgo fd is open inherits that fd.
		// Until the child has called exec, it holds the fd open and the
		// kernel will not let us run cgo.  Even if the child were to close
		// the fd explicitly, it would still be open from the time of the fork
		// until the time of the explicit close, and the race would remain.
		//
		// On Unix systems, this results in ETXTBSY, which formats
		// as "text file busy".  Rather than hard-code specific error cases,
		// we just look for that string.  If this happens, sleep a little
		// and try again.  We let this happen three times, with increasing
		// sleep lengths: 100+200+400 ms = 0.7 seconds.
		//
		// An alternate solution might be to split the cmd.Run into
		// separate cmd.Start and cmd.Wait, and then use an RWLock
		// to make sure that copyFile only executes when no cmd.Start
		// call is in progress.  However, cmd.Start (really syscall.forkExec)
		// only guarantees that when it returns, the exec is committed to
		// happen and succeed.  It uses a close-on-exec file descriptor
		// itself to determine this, so we know that when cmd.Start returns,
		// at least one close-on-exec file descriptor has been closed.
		// However, we cannot be sure that all of them have been closed,
		// so the program might still encounter ETXTBSY even with such
		// an RWLock.  The race window would be smaller, perhaps, but not
		// guaranteed to be gone.
		//
		// Sleeping when we observe the race seems to be the most reliable
		// option we have.
		//
		// https://golang.org/issue/3001
		//
		if err != nil && nbusy < 3 && strings.Contains(err.Error(), "text file busy") {
			time.Sleep(100 * time.Millisecond << uint(nbusy))
			nbusy++
			continue
		}

		// err can be something like 'exit status 1'.
		// Add information about what program was running.
		// Note that if buf.Bytes() is non-empty, the caller usually
		// shows buf.Bytes() and does not print err at all, so the
		// prefix here does not make most output any more verbose.
		if err != nil {
			err = errors.New(cmdline[0] + ": " + err.Error())
		}
		return buf.Bytes(), err
	}
}

// joinUnambiguously prints the slice, quoting where necessary to make the
// output unambiguous.
// TODO: See issue 5279. The printing of commands needs a complete redo.
func joinUnambiguously(a []string) string {
	var buf bytes.Buffer
	for i, s := range a {
		if i > 0 {
			buf.WriteByte(' ')
		}
		q := strconv.Quote(s)
		if s == "" || strings.Contains(s, " ") || len(q) > len(s)+2 {
			buf.WriteString(q)
		} else {
			buf.WriteString(s)
		}
	}
	return buf.String()
}

// mkdir makes the named directory.
func (b *builder) mkdir(dir string) error {
	b.exec.Lock()
	defer b.exec.Unlock()
	// We can be a little aggressive about being
	// sure directories exist.  Skip repeated calls.
	if b.mkdirCache[dir] {
		return nil
	}
	b.mkdirCache[dir] = true

	if buildN || buildX {
		b.showcmd("", "mkdir -p %s", dir)
		if buildN {
			return nil
		}
	}

	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}
	return nil
}

// mkAbs returns an absolute path corresponding to
// evaluating f in the directory dir.
// We always pass absolute paths of source files so that
// the error messages will include the full path to a file
// in need of attention.
func mkAbs(dir, f string) string {
	// Leave absolute paths alone.
	// Also, during -n mode we use the pseudo-directory $WORK
	// instead of creating an actual work directory that won't be used.
	// Leave paths beginning with $WORK alone too.
	if filepath.IsAbs(f) || strings.HasPrefix(f, "$WORK") {
		return f
	}
	return filepath.Join(dir, f)
}

type toolchain interface {
	// gc runs the compiler in a specific directory on a set of files
	// and returns the name of the generated output file.
	// The compiler runs in the directory dir.
	gc(b *builder, p *Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, out []byte, err error)
	// cc runs the toolchain's C compiler in a directory on a C file
	// to produce an output file.
	cc(b *builder, p *Package, objdir, ofile, cfile string) error
	// asm runs the assembler in a specific directory on a specific file
	// to generate the named output file.
	asm(b *builder, p *Package, obj, ofile, sfile string) error
	// pkgpath builds an appropriate path for a temporary package file.
	pkgpath(basedir string, p *Package) string
	// pack runs the archive packer in a specific directory to create
	// an archive from a set of object files.
	// typically it is run in the object directory.
	pack(b *builder, p *Package, objDir, afile string, ofiles []string) error
	// ld runs the linker to create an executable starting at mainpkg.
	ld(b *builder, root *action, out string, allactions []*action, mainpkg string, ofiles []string) error
	// ldShared runs the linker to create a shared library containing the pkgs built by toplevelactions
	ldShared(b *builder, toplevelactions []*action, out string, allactions []*action) error

	compiler() string
	linker() string
}

type noToolchain struct{}

func noCompiler() error {
	log.Fatalf("unknown compiler %q", buildContext.Compiler)
	return nil
}

func (noToolchain) compiler() string {
	noCompiler()
	return ""
}

func (noToolchain) linker() string {
	noCompiler()
	return ""
}

func (noToolchain) gc(b *builder, p *Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, out []byte, err error) {
	return "", nil, noCompiler()
}

func (noToolchain) asm(b *builder, p *Package, obj, ofile, sfile string) error {
	return noCompiler()
}

func (noToolchain) pkgpath(basedir string, p *Package) string {
	noCompiler()
	return ""
}

func (noToolchain) pack(b *builder, p *Package, objDir, afile string, ofiles []string) error {
	return noCompiler()
}

func (noToolchain) ld(b *builder, root *action, out string, allactions []*action, mainpkg string, ofiles []string) error {
	return noCompiler()
}

func (noToolchain) ldShared(b *builder, toplevelactions []*action, out string, allactions []*action) error {
	return noCompiler()
}

func (noToolchain) cc(b *builder, p *Package, objdir, ofile, cfile string) error {
	return noCompiler()
}

// The Go toolchain.
type gcToolchain struct{}

func (gcToolchain) compiler() string {
	return tool("compile")
}

func (gcToolchain) linker() string {
	return tool("link")
}

func (gcToolchain) gc(b *builder, p *Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, output []byte, err error) {
	if archive != "" {
		ofile = archive
	} else {
		out := "_go_.o"
		ofile = obj + out
	}

	gcargs := []string{"-p", p.ImportPath}
	if p.Name == "main" {
		gcargs[1] = "main"
	}
	if p.Standard && p.ImportPath == "runtime" {
		// runtime compiles with a special gc flag to emit
		// additional reflect type data.
		gcargs = append(gcargs, "-+")
	}

	// If we're giving the compiler the entire package (no C etc files), tell it that,
	// so that it can give good error messages about forward declarations.
	// Exceptions: a few standard packages have forward declarations for
	// pieces supplied behind-the-scenes by package runtime.
	extFiles := len(p.CgoFiles) + len(p.CFiles) + len(p.CXXFiles) + len(p.MFiles) + len(p.SFiles) + len(p.SysoFiles) + len(p.SwigFiles) + len(p.SwigCXXFiles)
	if p.Standard {
		switch p.ImportPath {
		case "bytes", "net", "os", "runtime/pprof", "sync", "time":
			extFiles++
		}
	}
	if extFiles == 0 {
		gcargs = append(gcargs, "-complete")
	}
	if buildContext.InstallSuffix != "" {
		gcargs = append(gcargs, "-installsuffix", buildContext.InstallSuffix)
	}
	if p.buildID != "" {
		gcargs = append(gcargs, "-buildid", p.buildID)
	}

	for _, path := range p.Imports {
		if i := strings.LastIndex(path, "/vendor/"); i >= 0 {
			gcargs = append(gcargs, "-importmap", path[i+len("/vendor/"):]+"="+path)
		} else if strings.HasPrefix(path, "vendor/") {
			gcargs = append(gcargs, "-importmap", path[len("vendor/"):]+"="+path)
		}
	}

	for _, path := range p.Imports {
		if i := strings.LastIndex(path, "/vendor/"); i >= 0 {
			gcargs = append(gcargs, "-importmap", path[i+len("/vendor/"):]+"="+path)
		} else if strings.HasPrefix(path, "vendor/") {
			gcargs = append(gcargs, "-importmap", path[len("vendor/"):]+"="+path)
		}
	}

	args := []interface{}{buildToolExec, tool("compile"), "-o", ofile, "-trimpath", b.work, buildGcflags, gcargs, "-D", p.localPrefix, importArgs}
	if ofile == archive {
		args = append(args, "-pack")
	}
	if asmhdr {
		args = append(args, "-asmhdr", obj+"go_asm.h")
	}
	for _, f := range gofiles {
		args = append(args, mkAbs(p.Dir, f))
	}

	output, err = b.runOut(p.Dir, p.ImportPath, nil, args...)
	return ofile, output, err
}

func (gcToolchain) asm(b *builder, p *Package, obj, ofile, sfile string) error {
	// Add -I pkg/GOOS_GOARCH so #include "textflag.h" works in .s files.
	inc := filepath.Join(goroot, "pkg", "include")
	sfile = mkAbs(p.Dir, sfile)
	args := []interface{}{buildToolExec, tool("asm"), "-o", ofile, "-trimpath", b.work, "-I", obj, "-I", inc, "-D", "GOOS_" + goos, "-D", "GOARCH_" + goarch, buildAsmflags, sfile}
	if err := b.run(p.Dir, p.ImportPath, nil, args...); err != nil {
		return err
	}
	return nil
}

// toolVerify checks that the command line args writes the same output file
// if run using newTool instead.
// Unused now but kept around for future use.
func toolVerify(b *builder, p *Package, newTool string, ofile string, args []interface{}) error {
	newArgs := make([]interface{}, len(args))
	copy(newArgs, args)
	newArgs[1] = tool(newTool)
	newArgs[3] = ofile + ".new" // x.6 becomes x.6.new
	if err := b.run(p.Dir, p.ImportPath, nil, newArgs...); err != nil {
		return err
	}
	data1, err := ioutil.ReadFile(ofile)
	if err != nil {
		return err
	}
	data2, err := ioutil.ReadFile(ofile + ".new")
	if err != nil {
		return err
	}
	if !bytes.Equal(data1, data2) {
		return fmt.Errorf("%s and %s produced different output files:\n%s\n%s", filepath.Base(args[1].(string)), newTool, strings.Join(stringList(args...), " "), strings.Join(stringList(newArgs...), " "))
	}
	os.Remove(ofile + ".new")
	return nil
}

func (gcToolchain) pkgpath(basedir string, p *Package) string {
	end := filepath.FromSlash(p.ImportPath + ".a")
	return filepath.Join(basedir, end)
}

func (gcToolchain) pack(b *builder, p *Package, objDir, afile string, ofiles []string) error {
	var absOfiles []string
	for _, f := range ofiles {
		absOfiles = append(absOfiles, mkAbs(objDir, f))
	}
	cmd := "c"
	absAfile := mkAbs(objDir, afile)
	appending := false
	if _, err := os.Stat(absAfile); err == nil {
		appending = true
		cmd = "r"
	}

	cmdline := stringList("pack", cmd, absAfile, absOfiles)

	if appending {
		if buildN || buildX {
			b.showcmd(p.Dir, "%s # internal", joinUnambiguously(cmdline))
		}
		if buildN {
			return nil
		}
		if err := packInternal(b, absAfile, absOfiles); err != nil {
			b.showOutput(p.Dir, p.ImportPath, err.Error()+"\n")
			return errPrintedOutput
		}
		return nil
	}

	// Need actual pack.
	cmdline[0] = tool("pack")
	return b.run(p.Dir, p.ImportPath, nil, buildToolExec, cmdline)
}

func packInternal(b *builder, afile string, ofiles []string) error {
	dst, err := os.OpenFile(afile, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return err
	}
	defer dst.Close() // only for error returns or panics
	w := bufio.NewWriter(dst)

	for _, ofile := range ofiles {
		src, err := os.Open(ofile)
		if err != nil {
			return err
		}
		fi, err := src.Stat()
		if err != nil {
			src.Close()
			return err
		}
		// Note: Not using %-16.16s format because we care
		// about bytes, not runes.
		name := fi.Name()
		if len(name) > 16 {
			name = name[:16]
		} else {
			name += strings.Repeat(" ", 16-len(name))
		}
		size := fi.Size()
		fmt.Fprintf(w, "%s%-12d%-6d%-6d%-8o%-10d`\n",
			name, 0, 0, 0, 0644, size)
		n, err := io.Copy(w, src)
		src.Close()
		if err == nil && n < size {
			err = io.ErrUnexpectedEOF
		} else if err == nil && n > size {
			err = fmt.Errorf("file larger than size reported by stat")
		}
		if err != nil {
			return fmt.Errorf("copying %s to %s: %v", ofile, afile, err)
		}
		if size&1 != 0 {
			w.WriteByte(0)
		}
	}

	if err := w.Flush(); err != nil {
		return err
	}
	return dst.Close()
}

// setextld sets the appropriate linker flags for the specified compiler.
func setextld(ldflags []string, compiler []string) []string {
	for _, f := range ldflags {
		if f == "-extld" || strings.HasPrefix(f, "-extld=") {
			// don't override -extld if supplied
			return ldflags
		}
	}
	ldflags = append(ldflags, "-extld="+compiler[0])
	if len(compiler) > 1 {
		extldflags := false
		add := strings.Join(compiler[1:], " ")
		for i, f := range ldflags {
			if f == "-extldflags" && i+1 < len(ldflags) {
				ldflags[i+1] = add + " " + ldflags[i+1]
				extldflags = true
				break
			} else if strings.HasPrefix(f, "-extldflags=") {
				ldflags[i] = "-extldflags=" + add + " " + ldflags[i][len("-extldflags="):]
				extldflags = true
				break
			}
		}
		if !extldflags {
			ldflags = append(ldflags, "-extldflags="+add)
		}
	}
	return ldflags
}

func (gcToolchain) ld(b *builder, root *action, out string, allactions []*action, mainpkg string, ofiles []string) error {
	importArgs := b.includeArgs("-L", allactions)
	cxx := len(root.p.CXXFiles) > 0 || len(root.p.SwigCXXFiles) > 0
	for _, a := range allactions {
		if a.p != nil && (len(a.p.CXXFiles) > 0 || len(a.p.SwigCXXFiles) > 0) {
			cxx = true
		}
	}
	var ldflags []string
	if buildContext.InstallSuffix != "" {
		ldflags = append(ldflags, "-installsuffix", buildContext.InstallSuffix)
	}
	if root.p.omitDWARF {
		ldflags = append(ldflags, "-w")
	}

	// If the user has not specified the -extld option, then specify the
	// appropriate linker. In case of C++ code, use the compiler named
	// by the CXX environment variable or defaultCXX if CXX is not set.
	// Else, use the CC environment variable and defaultCC as fallback.
	var compiler []string
	if cxx {
		compiler = envList("CXX", defaultCXX)
	} else {
		compiler = envList("CC", defaultCC)
	}
	ldflags = setextld(ldflags, compiler)
	ldflags = append(ldflags, "-buildmode="+ldBuildmode)
	if root.p.buildID != "" {
		ldflags = append(ldflags, "-buildid="+root.p.buildID)
	}
	ldflags = append(ldflags, buildLdflags...)

	// On OS X when using external linking to build a shared library,
	// the argument passed here to -o ends up recorded in the final
	// shared library in the LC_ID_DYLIB load command.
	// To avoid putting the temporary output directory name there
	// (and making the resulting shared library useless),
	// run the link in the output directory so that -o can name
	// just the final path element.
	dir := "."
	if goos == "darwin" && buildBuildmode == "c-shared" {
		dir, out = filepath.Split(out)
	}

	return b.run(dir, root.p.ImportPath, nil, buildToolExec, tool("link"), "-o", out, importArgs, ldflags, mainpkg)
}

func (gcToolchain) ldShared(b *builder, toplevelactions []*action, out string, allactions []*action) error {
	importArgs := b.includeArgs("-L", allactions)
	ldflags := []string{"-installsuffix", buildContext.InstallSuffix}
	ldflags = append(ldflags, "-buildmode=shared")
	ldflags = append(ldflags, buildLdflags...)
	cxx := false
	for _, a := range allactions {
		if a.p != nil && (len(a.p.CXXFiles) > 0 || len(a.p.SwigCXXFiles) > 0) {
			cxx = true
		}
	}
	// If the user has not specified the -extld option, then specify the
	// appropriate linker. In case of C++ code, use the compiler named
	// by the CXX environment variable or defaultCXX if CXX is not set.
	// Else, use the CC environment variable and defaultCC as fallback.
	var compiler []string
	if cxx {
		compiler = envList("CXX", defaultCXX)
	} else {
		compiler = envList("CC", defaultCC)
	}
	ldflags = setextld(ldflags, compiler)
	for _, d := range toplevelactions {
		if !strings.HasSuffix(d.target, ".a") { // omit unsafe etc and actions for other shared libraries
			continue
		}
		ldflags = append(ldflags, d.p.ImportPath+"="+d.target)
	}
	return b.run(".", out, nil, buildToolExec, tool("link"), "-o", out, importArgs, ldflags)
}

func (gcToolchain) cc(b *builder, p *Package, objdir, ofile, cfile string) error {
	return fmt.Errorf("%s: C source files not supported without cgo", mkAbs(p.Dir, cfile))
}

// The Gccgo toolchain.
type gccgoToolchain struct{}

var gccgoName, gccgoBin string

func init() {
	gccgoName = os.Getenv("GCCGO")
	if gccgoName == "" {
		gccgoName = defaultGCCGO
	}
	gccgoBin, _ = exec.LookPath(gccgoName)
}

func (gccgoToolchain) compiler() string {
	return gccgoBin
}

func (gccgoToolchain) linker() string {
	return gccgoBin
}

func (tools gccgoToolchain) gc(b *builder, p *Package, archive, obj string, asmhdr bool, importArgs []string, gofiles []string) (ofile string, output []byte, err error) {
	out := "_go_.o"
	ofile = obj + out
	gcargs := []string{"-g"}
	gcargs = append(gcargs, b.gccArchArgs()...)
	if pkgpath := gccgoPkgpath(p); pkgpath != "" {
		gcargs = append(gcargs, "-fgo-pkgpath="+pkgpath)
	}
	if p.localPrefix != "" {
		gcargs = append(gcargs, "-fgo-relative-import-path="+p.localPrefix)
	}
	args := stringList(tools.compiler(), importArgs, "-c", gcargs, "-o", ofile, buildGccgoflags)
	for _, f := range gofiles {
		args = append(args, mkAbs(p.Dir, f))
	}

	output, err = b.runOut(p.Dir, p.ImportPath, nil, args)
	return ofile, output, err
}

func (tools gccgoToolchain) asm(b *builder, p *Package, obj, ofile, sfile string) error {
	sfile = mkAbs(p.Dir, sfile)
	defs := []string{"-D", "GOOS_" + goos, "-D", "GOARCH_" + goarch}
	if pkgpath := gccgoCleanPkgpath(p); pkgpath != "" {
		defs = append(defs, `-D`, `GOPKGPATH="`+pkgpath+`"`)
	}
	defs = tools.maybePIC(defs)
	defs = append(defs, b.gccArchArgs()...)
	return b.run(p.Dir, p.ImportPath, nil, tools.compiler(), "-c", "-I", obj, "-o", ofile, defs, sfile)
}

func (gccgoToolchain) pkgpath(basedir string, p *Package) string {
	end := filepath.FromSlash(p.ImportPath + ".a")
	afile := filepath.Join(basedir, end)
	// add "lib" to the final element
	return filepath.Join(filepath.Dir(afile), "lib"+filepath.Base(afile))
}

func (gccgoToolchain) pack(b *builder, p *Package, objDir, afile string, ofiles []string) error {
	var absOfiles []string
	for _, f := range ofiles {
		absOfiles = append(absOfiles, mkAbs(objDir, f))
	}
	return b.run(p.Dir, p.ImportPath, nil, "ar", "cru", mkAbs(objDir, afile), absOfiles)
}

func (tools gccgoToolchain) ld(b *builder, root *action, out string, allactions []*action, mainpkg string, ofiles []string) error {
	// gccgo needs explicit linking with all package dependencies,
	// and all LDFLAGS from cgo dependencies.
	apackagesSeen := make(map[*Package]bool)
	afiles := []string{}
	shlibs := []string{}
	xfiles := []string{}
	ldflags := b.gccArchArgs()
	cgoldflags := []string{}
	usesCgo := false
	cxx := len(root.p.CXXFiles) > 0 || len(root.p.SwigCXXFiles) > 0
	objc := len(root.p.MFiles) > 0

	actionsSeen := make(map[*action]bool)
	// Make a pre-order depth-first traversal of the action graph, taking note of
	// whether a shared library action has been seen on the way to an action (the
	// construction of the graph means that if any path to a node passes through
	// a shared library action, they all do).
	var walk func(a *action, seenShlib bool)
	walk = func(a *action, seenShlib bool) {
		if actionsSeen[a] {
			return
		}
		actionsSeen[a] = true
		if a.p != nil && !seenShlib {
			if a.p.Standard {
				return
			}
			// We record the target of the first time we see a .a file
			// for a package to make sure that we prefer the 'install'
			// rather than the 'build' location (which may not exist any
			// more). We still need to traverse the dependencies of the
			// build action though so saying
			// if apackagesSeen[a.p] { return }
			// doesn't work.
			if !apackagesSeen[a.p] {
				apackagesSeen[a.p] = true
				if a.p.fake && a.p.external {
					// external _tests, if present must come before
					// internal _tests. Store these on a separate list
					// and place them at the head after this loop.
					xfiles = append(xfiles, a.target)
				} else if a.p.fake {
					// move _test files to the top of the link order
					afiles = append([]string{a.target}, afiles...)
				} else {
					afiles = append(afiles, a.target)
				}
			}
		}
		if strings.HasSuffix(a.target, ".so") {
			shlibs = append(shlibs, a.target)
			seenShlib = true
		}
		for _, a1 := range a.deps {
			walk(a1, seenShlib)
		}
	}
	for _, a1 := range root.deps {
		walk(a1, false)
	}
	afiles = append(xfiles, afiles...)

	for _, a := range allactions {
		// Gather CgoLDFLAGS, but not from standard packages.
		// The go tool can dig up runtime/cgo from GOROOT and
		// think that it should use its CgoLDFLAGS, but gccgo
		// doesn't use runtime/cgo.
		if a.p == nil {
			continue
		}
		if !a.p.Standard {
			cgoldflags = append(cgoldflags, a.p.CgoLDFLAGS...)
		}
		if len(a.p.CgoFiles) > 0 {
			usesCgo = true
		}
		if a.p.usesSwig() {
			usesCgo = true
		}
		if len(a.p.CXXFiles) > 0 || len(a.p.SwigCXXFiles) > 0 {
			cxx = true
		}
		if len(a.p.MFiles) > 0 {
			objc = true
		}
	}

	ldflags = append(ldflags, "-Wl,--whole-archive")
	ldflags = append(ldflags, afiles...)
	ldflags = append(ldflags, "-Wl,--no-whole-archive")

	ldflags = append(ldflags, cgoldflags...)
	ldflags = append(ldflags, envList("CGO_LDFLAGS", "")...)
	ldflags = append(ldflags, root.p.CgoLDFLAGS...)

	ldflags = stringList("-Wl,-(", ldflags, "-Wl,-)")

	for _, shlib := range shlibs {
		ldflags = append(
			ldflags,
			"-L"+filepath.Dir(shlib),
			"-Wl,-rpath="+filepath.Dir(shlib),
			"-l"+strings.TrimSuffix(
				strings.TrimPrefix(filepath.Base(shlib), "lib"),
				".so"))
	}

	var realOut string
	switch ldBuildmode {
	case "exe":
		if usesCgo && goos == "linux" {
			ldflags = append(ldflags, "-Wl,-E")
		}

	case "c-archive":
		// Link the Go files into a single .o, and also link
		// in -lgolibbegin.
		//
		// We need to use --whole-archive with -lgolibbegin
		// because it doesn't define any symbols that will
		// cause the contents to be pulled in; it's just
		// initialization code.
		//
		// The user remains responsible for linking against
		// -lgo -lpthread -lm in the final link.  We can't use
		// -r to pick them up because we can't combine
		// split-stack and non-split-stack code in a single -r
		// link, and libgo picks up non-split-stack code from
		// libffi.
		ldflags = append(ldflags, "-Wl,-r", "-nostdlib", "-Wl,--whole-archive", "-lgolibbegin", "-Wl,--no-whole-archive")

		// We are creating an object file, so we don't want a build ID.
		ldflags = b.disableBuildID(ldflags)

		realOut = out
		out = out + ".o"

	case "c-shared":
		ldflags = append(ldflags, "-shared", "-nostdlib", "-Wl,--whole-archive", "-lgolibbegin", "-Wl,--no-whole-archive", "-lgo", "-lgcc_s", "-lgcc")

	default:
		fatalf("-buildmode=%s not supported for gccgo", ldBuildmode)
	}

	switch ldBuildmode {
	case "exe", "c-shared":
		if cxx {
			ldflags = append(ldflags, "-lstdc++")
		}
		if objc {
			ldflags = append(ldflags, "-lobjc")
		}
	}

	if err := b.run(".", root.p.ImportPath, nil, tools.linker(), "-o", out, ofiles, ldflags, buildGccgoflags); err != nil {
		return err
	}

	switch ldBuildmode {
	case "c-archive":
		if err := b.run(".", root.p.ImportPath, nil, "ar", "rc", realOut, out); err != nil {
			return err
		}
	}
	return nil
}

func (tools gccgoToolchain) ldShared(b *builder, toplevelactions []*action, out string, allactions []*action) error {
	args := []string{"-o", out, "-shared", "-nostdlib", "-zdefs", "-Wl,--whole-archive"}
	for _, a := range toplevelactions {
		args = append(args, a.target)
	}
	args = append(args, "-Wl,--no-whole-archive", "-shared", "-nostdlib", "-lgo", "-lgcc_s", "-lgcc", "-lc")
	shlibs := []string{}
	for _, a := range allactions {
		if strings.HasSuffix(a.target, ".so") {
			shlibs = append(shlibs, a.target)
		}
	}
	for _, shlib := range shlibs {
		args = append(
			args,
			"-L"+filepath.Dir(shlib),
			"-Wl,-rpath="+filepath.Dir(shlib),
			"-l"+strings.TrimSuffix(
				strings.TrimPrefix(filepath.Base(shlib), "lib"),
				".so"))
	}
	return b.run(".", out, nil, tools.linker(), args, buildGccgoflags)
}

func (tools gccgoToolchain) cc(b *builder, p *Package, objdir, ofile, cfile string) error {
	inc := filepath.Join(goroot, "pkg", "include")
	cfile = mkAbs(p.Dir, cfile)
	defs := []string{"-D", "GOOS_" + goos, "-D", "GOARCH_" + goarch}
	defs = append(defs, b.gccArchArgs()...)
	if pkgpath := gccgoCleanPkgpath(p); pkgpath != "" {
		defs = append(defs, `-D`, `GOPKGPATH="`+pkgpath+`"`)
	}
	switch goarch {
	case "386", "amd64":
		defs = append(defs, "-fsplit-stack")
	}
	defs = tools.maybePIC(defs)
	return b.run(p.Dir, p.ImportPath, nil, envList("CC", defaultCC), "-Wall", "-g",
		"-I", objdir, "-I", inc, "-o", ofile, defs, "-c", cfile)
}

// maybePIC adds -fPIC to the list of arguments if needed.
func (tools gccgoToolchain) maybePIC(args []string) []string {
	switch buildBuildmode {
	case "c-shared", "shared":
		args = append(args, "-fPIC")
	}
	return args
}

func gccgoPkgpath(p *Package) string {
	if p.build.IsCommand() && !p.forceLibrary {
		return ""
	}
	return p.ImportPath
}

func gccgoCleanPkgpath(p *Package) string {
	clean := func(r rune) rune {
		switch {
		case 'A' <= r && r <= 'Z', 'a' <= r && r <= 'z',
			'0' <= r && r <= '9':
			return r
		}
		return '_'
	}
	return strings.Map(clean, gccgoPkgpath(p))
}

// libgcc returns the filename for libgcc, as determined by invoking gcc with
// the -print-libgcc-file-name option.
func (b *builder) libgcc(p *Package) (string, error) {
	var buf bytes.Buffer

	gccCmd := b.gccCmd(p.Dir)

	prev := b.print
	if buildN {
		// In -n mode we temporarily swap out the builder's
		// print function to capture the command-line. This
		// let's us assign it to $LIBGCC and produce a valid
		// buildscript for cgo packages.
		b.print = func(a ...interface{}) (int, error) {
			return fmt.Fprint(&buf, a...)
		}
	}
	f, err := b.runOut(p.Dir, p.ImportPath, nil, gccCmd, "-print-libgcc-file-name")
	if err != nil {
		return "", fmt.Errorf("gcc -print-libgcc-file-name: %v (%s)", err, f)
	}
	if buildN {
		s := fmt.Sprintf("LIBGCC=$(%s)\n", buf.Next(buf.Len()-1))
		b.print = prev
		b.print(s)
		return "$LIBGCC", nil
	}

	// The compiler might not be able to find libgcc, and in that case,
	// it will simply return "libgcc.a", which is of no use to us.
	if !filepath.IsAbs(string(f)) {
		return "", nil
	}

	return strings.Trim(string(f), "\r\n"), nil
}

// gcc runs the gcc C compiler to create an object from a single C file.
func (b *builder) gcc(p *Package, out string, flags []string, cfile string) error {
	return b.ccompile(p, out, flags, cfile, b.gccCmd(p.Dir))
}

// gxx runs the g++ C++ compiler to create an object from a single C++ file.
func (b *builder) gxx(p *Package, out string, flags []string, cxxfile string) error {
	return b.ccompile(p, out, flags, cxxfile, b.gxxCmd(p.Dir))
}

// ccompile runs the given C or C++ compiler and creates an object from a single source file.
func (b *builder) ccompile(p *Package, out string, flags []string, file string, compiler []string) error {
	file = mkAbs(p.Dir, file)
	return b.run(p.Dir, p.ImportPath, nil, compiler, flags, "-o", out, "-c", file)
}

// gccld runs the gcc linker to create an executable from a set of object files.
func (b *builder) gccld(p *Package, out string, flags []string, obj []string) error {
	var cmd []string
	if len(p.CXXFiles) > 0 || len(p.SwigCXXFiles) > 0 {
		cmd = b.gxxCmd(p.Dir)
	} else {
		cmd = b.gccCmd(p.Dir)
	}
	return b.run(p.Dir, p.ImportPath, nil, cmd, "-o", out, obj, flags)
}

// gccCmd returns a gcc command line prefix
// defaultCC is defined in zdefaultcc.go, written by cmd/dist.
func (b *builder) gccCmd(objdir string) []string {
	return b.ccompilerCmd("CC", defaultCC, objdir)
}

// gxxCmd returns a g++ command line prefix
// defaultCXX is defined in zdefaultcc.go, written by cmd/dist.
func (b *builder) gxxCmd(objdir string) []string {
	return b.ccompilerCmd("CXX", defaultCXX, objdir)
}

// ccompilerCmd returns a command line prefix for the given environment
// variable and using the default command when the variable is empty.
func (b *builder) ccompilerCmd(envvar, defcmd, objdir string) []string {
	// NOTE: env.go's mkEnv knows that the first three
	// strings returned are "gcc", "-I", objdir (and cuts them off).

	compiler := envList(envvar, defcmd)
	a := []string{compiler[0], "-I", objdir}
	a = append(a, compiler[1:]...)

	// Definitely want -fPIC but on Windows gcc complains
	// "-fPIC ignored for target (all code is position independent)"
	if goos != "windows" {
		a = append(a, "-fPIC")
	}
	a = append(a, b.gccArchArgs()...)
	// gcc-4.5 and beyond require explicit "-pthread" flag
	// for multithreading with pthread library.
	if buildContext.CgoEnabled {
		switch goos {
		case "windows":
			a = append(a, "-mthreads")
		default:
			a = append(a, "-pthread")
		}
	}

	if strings.Contains(a[0], "clang") {
		// disable ASCII art in clang errors, if possible
		a = append(a, "-fno-caret-diagnostics")
		// clang is too smart about command-line arguments
		a = append(a, "-Qunused-arguments")
	}

	// disable word wrapping in error messages
	a = append(a, "-fmessage-length=0")

	// On OS X, some of the compilers behave as if -fno-common
	// is always set, and the Mach-O linker in 6l/8l assumes this.
	// See https://golang.org/issue/3253.
	if goos == "darwin" {
		a = append(a, "-fno-common")
	}

	return a
}

// gccArchArgs returns arguments to pass to gcc based on the architecture.
func (b *builder) gccArchArgs() []string {
	switch goarch {
	case "386":
		return []string{"-m32"}
	case "amd64", "amd64p32":
		return []string{"-m64"}
	case "arm":
		return []string{"-marm"} // not thumb
	}
	return nil
}

// envList returns the value of the given environment variable broken
// into fields, using the default value when the variable is empty.
func envList(key, def string) []string {
	v := os.Getenv(key)
	if v == "" {
		v = def
	}
	return strings.Fields(v)
}

// Return the flags to use when invoking the C or C++ compilers, or cgo.
func (b *builder) cflags(p *Package, def bool) (cppflags, cflags, cxxflags, ldflags []string) {
	var defaults string
	if def {
		defaults = "-g -O2"
	}

	cppflags = stringList(envList("CGO_CPPFLAGS", ""), p.CgoCPPFLAGS)
	cflags = stringList(envList("CGO_CFLAGS", defaults), p.CgoCFLAGS)
	cxxflags = stringList(envList("CGO_CXXFLAGS", defaults), p.CgoCXXFLAGS)
	ldflags = stringList(envList("CGO_LDFLAGS", defaults), p.CgoLDFLAGS)
	return
}

var cgoRe = regexp.MustCompile(`[/\\:]`)

var (
	cgoLibGccFile     string
	cgoLibGccErr      error
	cgoLibGccFileOnce sync.Once
)

func (b *builder) cgo(p *Package, cgoExe, obj string, pcCFLAGS, pcLDFLAGS, cgofiles, gccfiles, gxxfiles, mfiles []string) (outGo, outObj []string, err error) {
	cgoCPPFLAGS, cgoCFLAGS, cgoCXXFLAGS, cgoLDFLAGS := b.cflags(p, true)
	_, cgoexeCFLAGS, _, _ := b.cflags(p, false)
	cgoCPPFLAGS = append(cgoCPPFLAGS, pcCFLAGS...)
	cgoLDFLAGS = append(cgoLDFLAGS, pcLDFLAGS...)
	// If we are compiling Objective-C code, then we need to link against libobjc
	if len(mfiles) > 0 {
		cgoLDFLAGS = append(cgoLDFLAGS, "-lobjc")
	}

	// Allows including _cgo_export.h from .[ch] files in the package.
	cgoCPPFLAGS = append(cgoCPPFLAGS, "-I", obj)

	// cgo
	// TODO: CGOPKGPATH, CGO_FLAGS?
	gofiles := []string{obj + "_cgo_gotypes.go"}
	cfiles := []string{"_cgo_main.c", "_cgo_export.c"}
	for _, fn := range cgofiles {
		f := cgoRe.ReplaceAllString(fn[:len(fn)-2], "_")
		gofiles = append(gofiles, obj+f+"cgo1.go")
		cfiles = append(cfiles, f+"cgo2.c")
	}
	defunC := obj + "_cgo_defun.c"

	cgoflags := []string{}
	// TODO: make cgo not depend on $GOARCH?

	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoflags = append(cgoflags, "-import_runtime_cgo=false")
	}
	if p.Standard && (p.ImportPath == "runtime/race" || p.ImportPath == "runtime/cgo") {
		cgoflags = append(cgoflags, "-import_syscall=false")
	}

	// Update $CGO_LDFLAGS with p.CgoLDFLAGS.
	var cgoenv []string
	if len(cgoLDFLAGS) > 0 {
		flags := make([]string, len(cgoLDFLAGS))
		for i, f := range cgoLDFLAGS {
			flags[i] = strconv.Quote(f)
		}
		cgoenv = []string{"CGO_LDFLAGS=" + strings.Join(flags, " ")}
	}

	if _, ok := buildToolchain.(gccgoToolchain); ok {
		switch goarch {
		case "386", "amd64":
			cgoCFLAGS = append(cgoCFLAGS, "-fsplit-stack")
		}
		cgoflags = append(cgoflags, "-gccgo")
		if pkgpath := gccgoPkgpath(p); pkgpath != "" {
			cgoflags = append(cgoflags, "-gccgopkgpath="+pkgpath)
		}
	}

	switch buildBuildmode {
	case "c-archive", "c-shared":
		// Tell cgo that if there are any exported functions
		// it should generate a header file that C code can
		// #include.
		cgoflags = append(cgoflags, "-exportheader="+obj+"_cgo_install.h")
	}

	if err := b.run(p.Dir, p.ImportPath, cgoenv, buildToolExec, cgoExe, "-objdir", obj, "-importpath", p.ImportPath, cgoflags, "--", cgoCPPFLAGS, cgoexeCFLAGS, cgofiles); err != nil {
		return nil, nil, err
	}
	outGo = append(outGo, gofiles...)

	// cc _cgo_defun.c
	_, gccgo := buildToolchain.(gccgoToolchain)
	if gccgo {
		defunObj := obj + "_cgo_defun.o"
		if err := buildToolchain.cc(b, p, obj, defunObj, defunC); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, defunObj)
	}

	// gcc
	var linkobj []string

	var bareLDFLAGS []string
	// filter out -lsomelib, -l somelib, *.{so,dll,dylib}, and (on Darwin) -framework X
	for i := 0; i < len(cgoLDFLAGS); i++ {
		f := cgoLDFLAGS[i]
		switch {
		// skip "-lc" or "-l somelib"
		case strings.HasPrefix(f, "-l"):
			if f == "-l" {
				i++
			}
		// skip "-framework X" on Darwin
		case goos == "darwin" && f == "-framework":
			i++
		// skip "*.{dylib,so,dll}"
		case strings.HasSuffix(f, ".dylib"),
			strings.HasSuffix(f, ".so"),
			strings.HasSuffix(f, ".dll"):
			continue
		// Remove any -fsanitize=foo flags.
		// Otherwise the compiler driver thinks that we are doing final link
		// and links sanitizer runtime into the object file. But we are not doing
		// the final link, we will link the resulting object file again. And
		// so the program ends up with two copies of sanitizer runtime.
		// See issue 8788 for details.
		case strings.HasPrefix(f, "-fsanitize="):
			continue
		default:
			bareLDFLAGS = append(bareLDFLAGS, f)
		}
	}

	cgoLibGccFileOnce.Do(func() {
		cgoLibGccFile, cgoLibGccErr = b.libgcc(p)
	})
	if cgoLibGccFile == "" && cgoLibGccErr != nil {
		return nil, nil, err
	}

	var staticLibs []string
	if goos == "windows" {
		// libmingw32 and libmingwex might also use libgcc, so libgcc must come last,
		// and they also have some inter-dependencies, so must use linker groups.
		staticLibs = []string{"-Wl,--start-group", "-lmingwex", "-lmingw32", "-Wl,--end-group"}
	}
	if cgoLibGccFile != "" {
		staticLibs = append(staticLibs, cgoLibGccFile)
	}

	cflags := stringList(cgoCPPFLAGS, cgoCFLAGS)
	for _, cfile := range cfiles {
		ofile := obj + cfile[:len(cfile)-1] + "o"
		if err := b.gcc(p, ofile, cflags, obj+cfile); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		if !strings.HasSuffix(ofile, "_cgo_main.o") {
			outObj = append(outObj, ofile)
		}
	}

	for _, file := range gccfiles {
		ofile := obj + cgoRe.ReplaceAllString(file[:len(file)-1], "_") + "o"
		if err := b.gcc(p, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		outObj = append(outObj, ofile)
	}

	cxxflags := stringList(cgoCPPFLAGS, cgoCXXFLAGS)
	for _, file := range gxxfiles {
		// Append .o to the file, just in case the pkg has file.c and file.cpp
		ofile := obj + cgoRe.ReplaceAllString(file, "_") + ".o"
		if err := b.gxx(p, ofile, cxxflags, file); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		outObj = append(outObj, ofile)
	}

	for _, file := range mfiles {
		// Append .o to the file, just in case the pkg has file.c and file.m
		ofile := obj + cgoRe.ReplaceAllString(file, "_") + ".o"
		if err := b.gcc(p, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		outObj = append(outObj, ofile)
	}

	linkobj = append(linkobj, p.SysoFiles...)
	dynobj := obj + "_cgo_.o"
	pie := goarch == "arm" && (goos == "linux" || goos == "android")
	if pie { // we need to use -pie for Linux/ARM to get accurate imported sym
		cgoLDFLAGS = append(cgoLDFLAGS, "-pie")
	}
	if err := b.gccld(p, dynobj, cgoLDFLAGS, linkobj); err != nil {
		return nil, nil, err
	}
	if pie { // but we don't need -pie for normal cgo programs
		cgoLDFLAGS = cgoLDFLAGS[0 : len(cgoLDFLAGS)-1]
	}

	if _, ok := buildToolchain.(gccgoToolchain); ok {
		// we don't use dynimport when using gccgo.
		return outGo, outObj, nil
	}

	// cgo -dynimport
	importGo := obj + "_cgo_import.go"
	cgoflags = []string{}
	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoflags = append(cgoflags, "-dynlinker") // record path to dynamic linker
	}
	if err := b.run(p.Dir, p.ImportPath, nil, buildToolExec, cgoExe, "-objdir", obj, "-dynpackage", p.Name, "-dynimport", dynobj, "-dynout", importGo, cgoflags); err != nil {
		return nil, nil, err
	}
	outGo = append(outGo, importGo)

	ofile := obj + "_all.o"
	var gccObjs, nonGccObjs []string
	for _, f := range outObj {
		if strings.HasSuffix(f, ".o") {
			gccObjs = append(gccObjs, f)
		} else {
			nonGccObjs = append(nonGccObjs, f)
		}
	}
	ldflags := stringList(bareLDFLAGS, "-Wl,-r", "-nostdlib", staticLibs)

	// We are creating an object file, so we don't want a build ID.
	ldflags = b.disableBuildID(ldflags)

	if err := b.gccld(p, ofile, ldflags, gccObjs); err != nil {
		return nil, nil, err
	}

	// NOTE(rsc): The importObj is a 5c/6c/8c object and on Windows
	// must be processed before the gcc-generated objects.
	// Put it first.  https://golang.org/issue/2601
	outObj = stringList(nonGccObjs, ofile)

	return outGo, outObj, nil
}

// Run SWIG on all SWIG input files.
// TODO: Don't build a shared library, once SWIG emits the necessary
// pragmas for external linking.
func (b *builder) swig(p *Package, obj string, pcCFLAGS []string) (outGo, outC, outCXX []string, err error) {
	if err := b.swigVersionCheck(); err != nil {
		return nil, nil, nil, err
	}

	intgosize, err := b.swigIntSize(obj)
	if err != nil {
		return nil, nil, nil, err
	}

	for _, f := range p.SwigFiles {
		goFile, cFile, err := b.swigOne(p, f, obj, pcCFLAGS, false, intgosize)
		if err != nil {
			return nil, nil, nil, err
		}
		if goFile != "" {
			outGo = append(outGo, goFile)
		}
		if cFile != "" {
			outC = append(outC, cFile)
		}
	}
	for _, f := range p.SwigCXXFiles {
		goFile, cxxFile, err := b.swigOne(p, f, obj, pcCFLAGS, true, intgosize)
		if err != nil {
			return nil, nil, nil, err
		}
		if goFile != "" {
			outGo = append(outGo, goFile)
		}
		if cxxFile != "" {
			outCXX = append(outCXX, cxxFile)
		}
	}
	return outGo, outC, outCXX, nil
}

// Make sure SWIG is new enough.
var (
	swigCheckOnce sync.Once
	swigCheck     error
)

func (b *builder) swigDoVersionCheck() error {
	out, err := b.runOut("", "", nil, "swig", "-version")
	if err != nil {
		return err
	}
	re := regexp.MustCompile(`[vV]ersion +([\d]+)([.][\d]+)?([.][\d]+)?`)
	matches := re.FindSubmatch(out)
	if matches == nil {
		// Can't find version number; hope for the best.
		return nil
	}

	major, err := strconv.Atoi(string(matches[1]))
	if err != nil {
		// Can't find version number; hope for the best.
		return nil
	}
	const errmsg = "must have SWIG version >= 3.0.6"
	if major < 3 {
		return errors.New(errmsg)
	}
	if major > 3 {
		// 4.0 or later
		return nil
	}

	// We have SWIG version 3.x.
	if len(matches[2]) > 0 {
		minor, err := strconv.Atoi(string(matches[2][1:]))
		if err != nil {
			return nil
		}
		if minor > 0 {
			// 3.1 or later
			return nil
		}
	}

	// We have SWIG version 3.0.x.
	if len(matches[3]) > 0 {
		patch, err := strconv.Atoi(string(matches[3][1:]))
		if err != nil {
			return nil
		}
		if patch < 6 {
			// Before 3.0.6.
			return errors.New(errmsg)
		}
	}

	return nil
}

func (b *builder) swigVersionCheck() error {
	swigCheckOnce.Do(func() {
		swigCheck = b.swigDoVersionCheck()
	})
	return swigCheck
}

// This code fails to build if sizeof(int) <= 32
const swigIntSizeCode = `
package main
const i int = 1 << 32
`

// Determine the size of int on the target system for the -intgosize option
// of swig >= 2.0.9
func (b *builder) swigIntSize(obj string) (intsize string, err error) {
	if buildN {
		return "$INTBITS", nil
	}
	src := filepath.Join(b.work, "swig_intsize.go")
	if err = ioutil.WriteFile(src, []byte(swigIntSizeCode), 0644); err != nil {
		return
	}
	srcs := []string{src}

	p := goFilesPackage(srcs)

	if _, _, e := buildToolchain.gc(b, p, "", obj, false, nil, srcs); e != nil {
		return "32", nil
	}
	return "64", nil
}

// Run SWIG on one SWIG input file.
func (b *builder) swigOne(p *Package, file, obj string, pcCFLAGS []string, cxx bool, intgosize string) (outGo, outC string, err error) {
	cgoCPPFLAGS, cgoCFLAGS, cgoCXXFLAGS, _ := b.cflags(p, true)
	var cflags []string
	if cxx {
		cflags = stringList(cgoCPPFLAGS, pcCFLAGS, cgoCXXFLAGS)
	} else {
		cflags = stringList(cgoCPPFLAGS, pcCFLAGS, cgoCFLAGS)
	}

	n := 5 // length of ".swig"
	if cxx {
		n = 8 // length of ".swigcxx"
	}
	base := file[:len(file)-n]
	goFile := base + ".go"
	gccBase := base + "_wrap."
	gccExt := "c"
	if cxx {
		gccExt = "cxx"
	}

	_, gccgo := buildToolchain.(gccgoToolchain)

	// swig
	args := []string{
		"-go",
		"-cgo",
		"-intgosize", intgosize,
		"-module", base,
		"-o", obj + gccBase + gccExt,
		"-outdir", obj,
	}

	for _, f := range cflags {
		if len(f) > 3 && f[:2] == "-I" {
			args = append(args, f)
		}
	}

	if gccgo {
		args = append(args, "-gccgo")
		if pkgpath := gccgoPkgpath(p); pkgpath != "" {
			args = append(args, "-go-pkgpath", pkgpath)
		}
	}
	if cxx {
		args = append(args, "-c++")
	}

	out, err := b.runOut(p.Dir, p.ImportPath, nil, "swig", args, file)
	if err != nil {
		if len(out) > 0 {
			if bytes.Contains(out, []byte("-intgosize")) || bytes.Contains(out, []byte("-cgo")) {
				return "", "", errors.New("must have SWIG version >= 3.0.6")
			}
			b.showOutput(p.Dir, p.ImportPath, b.processOutput(out)) // swig error
			return "", "", errPrintedOutput
		}
		return "", "", err
	}
	if len(out) > 0 {
		b.showOutput(p.Dir, p.ImportPath, b.processOutput(out)) // swig warning
	}

	return obj + goFile, obj + gccBase + gccExt, nil
}

// disableBuildID adjusts a linker command line to avoid creating a
// build ID when creating an object file rather than an executable or
// shared library.  Some systems, such as Ubuntu, always add
// --build-id to every link, but we don't want a build ID when we are
// producing an object file.  On some of those system a plain -r (not
// -Wl,-r) will turn off --build-id, but clang 3.0 doesn't support a
// plain -r.  I don't know how to turn off --build-id when using clang
// other than passing a trailing --build-id=none.  So that is what we
// do, but only on systems likely to support it, which is to say,
// systems that normally use gold or the GNU linker.
func (b *builder) disableBuildID(ldflags []string) []string {
	switch goos {
	case "android", "dragonfly", "linux", "netbsd":
		ldflags = append(ldflags, "-Wl,--build-id=none")
	}
	return ldflags
}

// An actionQueue is a priority queue of actions.
type actionQueue []*action

// Implement heap.Interface
func (q *actionQueue) Len() int           { return len(*q) }
func (q *actionQueue) Swap(i, j int)      { (*q)[i], (*q)[j] = (*q)[j], (*q)[i] }
func (q *actionQueue) Less(i, j int) bool { return (*q)[i].priority < (*q)[j].priority }
func (q *actionQueue) Push(x interface{}) { *q = append(*q, x.(*action)) }
func (q *actionQueue) Pop() interface{} {
	n := len(*q) - 1
	x := (*q)[n]
	*q = (*q)[:n]
	return x
}

func (q *actionQueue) push(a *action) {
	heap.Push(q, a)
}

func (q *actionQueue) pop() *action {
	return heap.Pop(q).(*action)
}

func raceInit() {
	if !buildRace {
		return
	}
	if goarch != "amd64" || goos != "linux" && goos != "freebsd" && goos != "darwin" && goos != "windows" {
		fmt.Fprintf(os.Stderr, "go %s: -race is only supported on linux/amd64, freebsd/amd64, darwin/amd64 and windows/amd64\n", flag.Args()[0])
		os.Exit(2)
	}
	buildGcflags = append(buildGcflags, "-race")
	buildLdflags = append(buildLdflags, "-race")
	if buildContext.InstallSuffix != "" {
		buildContext.InstallSuffix += "_"
	}
	buildContext.InstallSuffix += "race"
	buildContext.BuildTags = append(buildContext.BuildTags, "race")
}

// defaultSuffix returns file extension used for command files in
// current os environment.
func defaultSuffix() string {
	switch runtime.GOOS {
	case "windows":
		return ".bat"
	case "plan9":
		return ".rc"
	default:
		return ".bash"
	}
}
