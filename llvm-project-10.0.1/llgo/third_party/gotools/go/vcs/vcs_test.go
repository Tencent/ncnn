// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcs

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Test that RepoRootForImportPath creates the correct RepoRoot for a given importPath.
// TODO(cmang): Add tests for SVN and BZR.
func TestRepoRootForImportPath(t *testing.T) {
	tests := []struct {
		path string
		want *RepoRoot
	}{
		{
			"code.google.com/p/go",
			&RepoRoot{
				VCS:  vcsHg,
				Repo: "https://code.google.com/p/go",
			},
		},
		{
			"code.google.com/r/go",
			&RepoRoot{
				VCS:  vcsHg,
				Repo: "https://code.google.com/r/go",
			},
		},
		{
			"github.com/golang/groupcache",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://github.com/golang/groupcache",
			},
		},
	}

	for _, test := range tests {
		got, err := RepoRootForImportPath(test.path, false)
		if err != nil {
			t.Errorf("RepoRootForImport(%q): %v", test.path, err)
			continue
		}
		want := test.want
		if got.VCS.Name != want.VCS.Name || got.Repo != want.Repo {
			t.Errorf("RepoRootForImport(%q) = VCS(%s) Repo(%s), want VCS(%s) Repo(%s)", test.path, got.VCS, got.Repo, want.VCS, want.Repo)
		}
	}
}

// Test that FromDir correctly inspects a given directory and returns the right VCS.
func TestFromDir(t *testing.T) {
	type testStruct struct {
		path string
		want *Cmd
	}

	tests := make([]testStruct, len(vcsList))
	tempDir, err := ioutil.TempDir("", "vcstest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	for i, vcs := range vcsList {
		tests[i] = testStruct{
			filepath.Join(tempDir, vcs.Name, "."+vcs.Cmd),
			vcs,
		}
	}

	for _, test := range tests {
		os.MkdirAll(test.path, 0755)
		got, _, _ := FromDir(test.path, tempDir)
		if got.Name != test.want.Name {
			t.Errorf("FromDir(%q, %q) = %s, want %s", test.path, tempDir, got, test.want)
		}
		os.RemoveAll(test.path)
	}
}

var parseMetaGoImportsTests = []struct {
	in  string
	out []metaImport
}{
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		<meta name="go-import" content="baz/quux git http://github.com/rsc/baz/quux">`,
		[]metaImport{
			{"foo/bar", "git", "https://github.com/rsc/foo/bar"},
			{"baz/quux", "git", "http://github.com/rsc/baz/quux"},
		},
	},
	{
		`<head>
		<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		</head>`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		<body>`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
}

func TestParseMetaGoImports(t *testing.T) {
	for i, tt := range parseMetaGoImportsTests {
		out, err := parseMetaGoImports(strings.NewReader(tt.in))
		if err != nil {
			t.Errorf("test#%d: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(out, tt.out) {
			t.Errorf("test#%d:\n\thave %q\n\twant %q", i, out, tt.out)
		}
	}
}
