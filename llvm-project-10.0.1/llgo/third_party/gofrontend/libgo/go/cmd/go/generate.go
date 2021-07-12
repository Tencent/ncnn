// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"unicode"
)

var cmdGenerate = &Command{
	Run:       runGenerate,
	UsageLine: "generate [-run regexp] [file.go... | packages]",
	Short:     "generate Go files by processing source",
	Long: `
Generate runs commands described by directives within existing
files. Those commands can run any process but the intent is to
create or update Go source files, for instance by running yacc.

Go generate is never run automatically by go build, go get, go test,
and so on. It must be run explicitly.

Go generate scans the file for directives, which are lines of
the form,

	//go:generate command argument...

(note: no leading spaces and no space in "//go") where command
is the generator to be run, corresponding to an executable file
that can be run locally. It must either be in the shell path
(gofmt), a fully qualified path (/usr/you/bin/mytool), or a
command alias, described below.

Note that go generate does not parse the file, so lines that look
like directives in comments or multiline strings will be treated
as directives.

The arguments to the directive are space-separated tokens or
double-quoted strings passed to the generator as individual
arguments when it is run.

Quoted strings use Go syntax and are evaluated before execution; a
quoted string appears as a single argument to the generator.

Go generate sets several variables when it runs the generator:

	$GOARCH
		The execution architecture (arm, amd64, etc.)
	$GOOS
		The execution operating system (linux, windows, etc.)
	$GOFILE
		The base name of the file.
	$GOLINE
		The line number of the directive in the source file.
	$GOPACKAGE
		The name of the package of the file containing the directive.
	$DOLLAR
		A dollar sign.

Other than variable substitution and quoted-string evaluation, no
special processing such as "globbing" is performed on the command
line.

As a last step before running the command, any invocations of any
environment variables with alphanumeric names, such as $GOFILE or
$HOME, are expanded throughout the command line. The syntax for
variable expansion is $NAME on all operating systems.  Due to the
order of evaluation, variables are expanded even inside quoted
strings. If the variable NAME is not set, $NAME expands to the
empty string.

A directive of the form,

	//go:generate -command xxx args...

specifies, for the remainder of this source file only, that the
string xxx represents the command identified by the arguments. This
can be used to create aliases or to handle multiword generators.
For example,

	//go:generate -command yacc go tool yacc

specifies that the command "yacc" represents the generator
"go tool yacc".

Generate processes packages in the order given on the command line,
one at a time. If the command line lists .go files, they are treated
as a single package. Within a package, generate processes the
source files in a package in file name order, one at a time. Within
a source file, generate runs generators in the order they appear
in the file, one at a time.

If any generator returns an error exit status, "go generate" skips
all further processing for that package.

The generator is run in the package's source directory.

Go generate accepts one specific flag:

	-run=""
		if non-empty, specifies a regular expression to select
		directives whose full original source text (excluding
		any trailing spaces and final newline) matches the
		expression.

It also accepts the standard build flags -v, -n, and -x.
The -v flag prints the names of packages and files as they are
processed.
The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

For more about specifying packages, see 'go help packages'.
	`,
}

var (
	generateRunFlag string         // generate -run flag
	generateRunRE   *regexp.Regexp // compiled expression for -run
)

func init() {
	addBuildFlags(cmdGenerate)
	cmdGenerate.Flag.StringVar(&generateRunFlag, "run", "", "")
}

func runGenerate(cmd *Command, args []string) {
	if generateRunFlag != "" {
		var err error
		generateRunRE, err = regexp.Compile(generateRunFlag)
		if err != nil {
			log.Fatalf("generate: %s", err)
		}
	}
	// Even if the arguments are .go files, this loop suffices.
	for _, pkg := range packages(args) {
		for _, file := range pkg.gofiles {
			if !generate(pkg.Name, file) {
				break
			}
		}
	}
}

// generate runs the generation directives for a single file.
func generate(pkg, absFile string) bool {
	fd, err := os.Open(absFile)
	if err != nil {
		log.Fatalf("generate: %s", err)
	}
	defer fd.Close()
	g := &Generator{
		r:        fd,
		path:     absFile,
		pkg:      pkg,
		commands: make(map[string][]string),
	}
	return g.run()
}

// A Generator represents the state of a single Go source file
// being scanned for generator commands.
type Generator struct {
	r        io.Reader
	path     string // full rooted path name.
	dir      string // full rooted directory of file.
	file     string // base name of file.
	pkg      string
	commands map[string][]string
	lineNum  int // current line number.
}

// run runs the generators in the current file.
func (g *Generator) run() (ok bool) {
	// Processing below here calls g.errorf on failure, which does panic(stop).
	// If we encounter an error, we abort the package.
	defer func() {
		e := recover()
		if e != nil {
			ok = false
			if e != stop {
				panic(e)
			}
			setExitStatus(1)
		}
	}()
	g.dir, g.file = filepath.Split(g.path)
	g.dir = filepath.Clean(g.dir) // No final separator please.
	if buildV {
		fmt.Fprintf(os.Stderr, "%s\n", shortPath(g.path))
	}

	// Scan for lines that start "//go:generate".
	// Can't use bufio.Scanner because it can't handle long lines,
	// which are likely to appear when using generate.
	input := bufio.NewReader(g.r)
	var err error
	// One line per loop.
	for {
		g.lineNum++ // 1-indexed.
		var buf []byte
		buf, err = input.ReadSlice('\n')
		if err == bufio.ErrBufferFull {
			// Line too long - consume and ignore.
			if isGoGenerate(buf) {
				g.errorf("directive too long")
			}
			for err == bufio.ErrBufferFull {
				_, err = input.ReadSlice('\n')
			}
			if err != nil {
				break
			}
			continue
		}

		if err != nil {
			// Check for marker at EOF without final \n.
			if err == io.EOF && isGoGenerate(buf) {
				err = io.ErrUnexpectedEOF
			}
			break
		}

		if !isGoGenerate(buf) {
			continue
		}
		if generateRunFlag != "" {
			if !generateRunRE.Match(bytes.TrimSpace(buf)) {
				continue
			}
		}

		words := g.split(string(buf))
		if len(words) == 0 {
			g.errorf("no arguments to directive")
		}
		if words[0] == "-command" {
			g.setShorthand(words)
			continue
		}
		// Run the command line.
		if buildN || buildX {
			fmt.Fprintf(os.Stderr, "%s\n", strings.Join(words, " "))
		}
		if buildN {
			continue
		}
		g.exec(words)
	}
	if err != nil && err != io.EOF {
		g.errorf("error reading %s: %s", shortPath(g.path), err)
	}
	return true
}

func isGoGenerate(buf []byte) bool {
	return bytes.HasPrefix(buf, []byte("//go:generate ")) || bytes.HasPrefix(buf, []byte("//go:generate\t"))
}

// split breaks the line into words, evaluating quoted
// strings and evaluating environment variables.
// The initial //go:generate element is present in line.
func (g *Generator) split(line string) []string {
	// Parse line, obeying quoted strings.
	var words []string
	line = line[len("//go:generate ") : len(line)-1] // Drop preamble and final newline.
	// There may still be a carriage return.
	if len(line) > 0 && line[len(line)-1] == '\r' {
		line = line[:len(line)-1]
	}
	// One (possibly quoted) word per iteration.
Words:
	for {
		line = strings.TrimLeft(line, " \t")
		if len(line) == 0 {
			break
		}
		if line[0] == '"' {
			for i := 1; i < len(line); i++ {
				c := line[i] // Only looking for ASCII so this is OK.
				switch c {
				case '\\':
					if i+1 == len(line) {
						g.errorf("bad backslash")
					}
					i++ // Absorb next byte (If it's a multibyte we'll get an error in Unquote).
				case '"':
					word, err := strconv.Unquote(line[0 : i+1])
					if err != nil {
						g.errorf("bad quoted string")
					}
					words = append(words, word)
					line = line[i+1:]
					// Check the next character is space or end of line.
					if len(line) > 0 && line[0] != ' ' && line[0] != '\t' {
						g.errorf("expect space after quoted argument")
					}
					continue Words
				}
			}
			g.errorf("mismatched quoted string")
		}
		i := strings.IndexAny(line, " \t")
		if i < 0 {
			i = len(line)
		}
		words = append(words, line[0:i])
		line = line[i:]
	}
	// Substitute command if required.
	if len(words) > 0 && g.commands[words[0]] != nil {
		// Replace 0th word by command substitution.
		words = append(g.commands[words[0]], words[1:]...)
	}
	// Substitute environment variables.
	for i, word := range words {
		words[i] = os.Expand(word, g.expandVar)
	}
	return words
}

var stop = fmt.Errorf("error in generation")

// errorf logs an error message prefixed with the file and line number.
// It then exits the program (with exit status 1) because generation stops
// at the first error.
func (g *Generator) errorf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s:%d: %s\n", shortPath(g.path), g.lineNum,
		fmt.Sprintf(format, args...))
	panic(stop)
}

// expandVar expands the $XXX invocation in word. It is called
// by os.Expand.
func (g *Generator) expandVar(word string) string {
	switch word {
	case "GOARCH":
		return buildContext.GOARCH
	case "GOOS":
		return buildContext.GOOS
	case "GOFILE":
		return g.file
	case "GOLINE":
		return fmt.Sprint(g.lineNum)
	case "GOPACKAGE":
		return g.pkg
	case "DOLLAR":
		return "$"
	default:
		return os.Getenv(word)
	}
}

// identLength returns the length of the identifier beginning the string.
func (g *Generator) identLength(word string) int {
	for i, r := range word {
		if r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) {
			continue
		}
		return i
	}
	return len(word)
}

// setShorthand installs a new shorthand as defined by a -command directive.
func (g *Generator) setShorthand(words []string) {
	// Create command shorthand.
	if len(words) == 1 {
		g.errorf("no command specified for -command")
	}
	command := words[1]
	if g.commands[command] != nil {
		g.errorf("command %q defined multiply defined", command)
	}
	g.commands[command] = words[2:len(words):len(words)] // force later append to make copy
}

// exec runs the command specified by the argument. The first word is
// the command name itself.
func (g *Generator) exec(words []string) {
	cmd := exec.Command(words[0], words[1:]...)
	// Standard in and out of generator should be the usual.
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Run the command in the package directory.
	cmd.Dir = g.dir
	env := []string{
		"GOARCH=" + runtime.GOARCH,
		"GOOS=" + runtime.GOOS,
		"GOFILE=" + g.file,
		"GOPACKAGE=" + g.pkg,
	}
	cmd.Env = mergeEnvLists(env, origEnv)
	err := cmd.Run()
	if err != nil {
		g.errorf("running %q: %s", words[0], err)
	}
}
