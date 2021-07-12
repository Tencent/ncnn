// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// FormatMediaType serializes mediatype t and the parameters
// param as a media type conforming to RFC 2045 and RFC 2616.
// The type and parameter names are written in lower-case.
// When any of the arguments result in a standard violation then
// FormatMediaType returns the empty string.
func FormatMediaType(t string, param map[string]string) string {
	slash := strings.Index(t, "/")
	if slash == -1 {
		return ""
	}
	major, sub := t[:slash], t[slash+1:]
	if !isToken(major) || !isToken(sub) {
		return ""
	}
	var b bytes.Buffer
	b.WriteString(strings.ToLower(major))
	b.WriteByte('/')
	b.WriteString(strings.ToLower(sub))

	attrs := make([]string, 0, len(param))
	for a := range param {
		attrs = append(attrs, a)
	}
	sort.Strings(attrs)

	for _, attribute := range attrs {
		value := param[attribute]
		b.WriteByte(';')
		b.WriteByte(' ')
		if !isToken(attribute) {
			return ""
		}
		b.WriteString(strings.ToLower(attribute))
		b.WriteByte('=')
		if isToken(value) {
			b.WriteString(value)
			continue
		}

		b.WriteByte('"')
		offset := 0
		for index, character := range value {
			if character == '"' || character == '\\' {
				b.WriteString(value[offset:index])
				offset = index
				b.WriteByte('\\')
			}
			if character&0x80 != 0 {
				return ""
			}
		}
		b.WriteString(value[offset:])
		b.WriteByte('"')
	}
	return b.String()
}

func checkMediaTypeDisposition(s string) error {
	typ, rest := consumeToken(s)
	if typ == "" {
		return errors.New("mime: no media type")
	}
	if rest == "" {
		return nil
	}
	if !strings.HasPrefix(rest, "/") {
		return errors.New("mime: expected slash after first token")
	}
	subtype, rest := consumeToken(rest[1:])
	if subtype == "" {
		return errors.New("mime: expected token after slash")
	}
	if rest != "" {
		return errors.New("mime: unexpected content after media subtype")
	}
	return nil
}

// ParseMediaType parses a media type value and any optional
// parameters, per RFC 1521.  Media types are the values in
// Content-Type and Content-Disposition headers (RFC 2183).
// On success, ParseMediaType returns the media type converted
// to lowercase and trimmed of white space and a non-nil map.
// The returned map, params, maps from the lowercase
// attribute to the attribute value with its case preserved.
func ParseMediaType(v string) (mediatype string, params map[string]string, err error) {
	i := strings.Index(v, ";")
	if i == -1 {
		i = len(v)
	}
	mediatype = strings.TrimSpace(strings.ToLower(v[0:i]))

	err = checkMediaTypeDisposition(mediatype)
	if err != nil {
		return "", nil, err
	}

	params = make(map[string]string)

	// Map of base parameter name -> parameter name -> value
	// for parameters containing a '*' character.
	// Lazily initialized.
	var continuation map[string]map[string]string

	v = v[i:]
	for len(v) > 0 {
		v = strings.TrimLeftFunc(v, unicode.IsSpace)
		if len(v) == 0 {
			break
		}
		key, value, rest := consumeMediaParam(v)
		if key == "" {
			if strings.TrimSpace(rest) == ";" {
				// Ignore trailing semicolons.
				// Not an error.
				return
			}
			// Parse error.
			return "", nil, errors.New("mime: invalid media parameter")
		}

		pmap := params
		if idx := strings.Index(key, "*"); idx != -1 {
			baseName := key[:idx]
			if continuation == nil {
				continuation = make(map[string]map[string]string)
			}
			var ok bool
			if pmap, ok = continuation[baseName]; !ok {
				continuation[baseName] = make(map[string]string)
				pmap = continuation[baseName]
			}
		}
		if _, exists := pmap[key]; exists {
			// Duplicate parameter name is bogus.
			return "", nil, errors.New("mime: duplicate parameter name")
		}
		pmap[key] = value
		v = rest
	}

	// Stitch together any continuations or things with stars
	// (i.e. RFC 2231 things with stars: "foo*0" or "foo*")
	var buf bytes.Buffer
	for key, pieceMap := range continuation {
		singlePartKey := key + "*"
		if v, ok := pieceMap[singlePartKey]; ok {
			decv := decode2231Enc(v)
			params[key] = decv
			continue
		}

		buf.Reset()
		valid := false
		for n := 0; ; n++ {
			simplePart := fmt.Sprintf("%s*%d", key, n)
			if v, ok := pieceMap[simplePart]; ok {
				valid = true
				buf.WriteString(v)
				continue
			}
			encodedPart := simplePart + "*"
			if v, ok := pieceMap[encodedPart]; ok {
				valid = true
				if n == 0 {
					buf.WriteString(decode2231Enc(v))
				} else {
					decv, _ := percentHexUnescape(v)
					buf.WriteString(decv)
				}
			} else {
				break
			}
		}
		if valid {
			params[key] = buf.String()
		}
	}

	return
}

func decode2231Enc(v string) string {
	sv := strings.SplitN(v, "'", 3)
	if len(sv) != 3 {
		return ""
	}
	// TODO: ignoring lang in sv[1] for now. If anybody needs it we'll
	// need to decide how to expose it in the API. But I'm not sure
	// anybody uses it in practice.
	charset := strings.ToLower(sv[0])
	if charset != "us-ascii" && charset != "utf-8" {
		// TODO: unsupported encoding
		return ""
	}
	encv, _ := percentHexUnescape(sv[2])
	return encv
}

func isNotTokenChar(r rune) bool {
	return !isTokenChar(r)
}

// consumeToken consumes a token from the beginning of provided
// string, per RFC 2045 section 5.1 (referenced from 2183), and return
// the token consumed and the rest of the string.  Returns ("", v) on
// failure to consume at least one character.
func consumeToken(v string) (token, rest string) {
	notPos := strings.IndexFunc(v, isNotTokenChar)
	if notPos == -1 {
		return v, ""
	}
	if notPos == 0 {
		return "", v
	}
	return v[0:notPos], v[notPos:]
}

// consumeValue consumes a "value" per RFC 2045, where a value is
// either a 'token' or a 'quoted-string'.  On success, consumeValue
// returns the value consumed (and de-quoted/escaped, if a
// quoted-string) and the rest of the string.  On failure, returns
// ("", v).
func consumeValue(v string) (value, rest string) {
	if !strings.HasPrefix(v, `"`) && !strings.HasPrefix(v, `'`) {
		return consumeToken(v)
	}

	leadQuote := rune(v[0])

	// parse a quoted-string
	rest = v[1:] // consume the leading quote
	buffer := new(bytes.Buffer)
	var idx int
	var r rune
	var nextIsLiteral bool
	for idx, r = range rest {
		switch {
		case nextIsLiteral:
			buffer.WriteRune(r)
			nextIsLiteral = false
		case r == leadQuote:
			return buffer.String(), rest[idx+1:]
		case r == '\\':
			nextIsLiteral = true
		case r != '\r' && r != '\n':
			buffer.WriteRune(r)
		default:
			return "", v
		}
	}
	return "", v
}

func consumeMediaParam(v string) (param, value, rest string) {
	rest = strings.TrimLeftFunc(v, unicode.IsSpace)
	if !strings.HasPrefix(rest, ";") {
		return "", "", v
	}

	rest = rest[1:] // consume semicolon
	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	param, rest = consumeToken(rest)
	param = strings.ToLower(param)
	if param == "" {
		return "", "", v
	}

	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	if !strings.HasPrefix(rest, "=") {
		return "", "", v
	}
	rest = rest[1:] // consume equals sign
	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	value, rest = consumeValue(rest)
	if value == "" {
		return "", "", v
	}
	return param, value, rest
}

func percentHexUnescape(s string) (string, error) {
	// Count %, check that they're well-formed.
	percents := 0
	for i := 0; i < len(s); {
		if s[i] != '%' {
			i++
			continue
		}
		percents++
		if i+2 >= len(s) || !ishex(s[i+1]) || !ishex(s[i+2]) {
			s = s[i:]
			if len(s) > 3 {
				s = s[0:3]
			}
			return "", fmt.Errorf("mime: bogus characters after %%: %q", s)
		}
		i += 3
	}
	if percents == 0 {
		return s, nil
	}

	t := make([]byte, len(s)-2*percents)
	j := 0
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			t[j] = unhex(s[i+1])<<4 | unhex(s[i+2])
			j++
			i += 3
		default:
			t[j] = s[i]
			j++
			i++
		}
	}
	return string(t), nil
}

func ishex(c byte) bool {
	switch {
	case '0' <= c && c <= '9':
		return true
	case 'a' <= c && c <= 'f':
		return true
	case 'A' <= c && c <= 'F':
		return true
	}
	return false
}

func unhex(c byte) byte {
	switch {
	case '0' <= c && c <= '9':
		return c - '0'
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10
	}
	return 0
}
