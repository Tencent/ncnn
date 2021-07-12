// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP server.  See RFC 2616.

package http

import (
	"bufio"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/textproto"
	"net/url"
	"os"
	"path"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Errors introduced by the HTTP server.
var (
	ErrWriteAfterFlush = errors.New("Conn.Write called after Flush")
	ErrBodyNotAllowed  = errors.New("http: request method or response status code does not allow body")
	ErrHijacked        = errors.New("Conn has been hijacked")
	ErrContentLength   = errors.New("Conn.Write wrote more than the declared Content-Length")
)

// Objects implementing the Handler interface can be
// registered to serve a particular path or subtree
// in the HTTP server.
//
// ServeHTTP should write reply headers and data to the ResponseWriter
// and then return.  Returning signals that the request is finished
// and that the HTTP server can move on to the next request on
// the connection.
//
// If ServeHTTP panics, the server (the caller of ServeHTTP) assumes
// that the effect of the panic was isolated to the active request.
// It recovers the panic, logs a stack trace to the server error log,
// and hangs up the connection.
//
type Handler interface {
	ServeHTTP(ResponseWriter, *Request)
}

// A ResponseWriter interface is used by an HTTP handler to
// construct an HTTP response.
type ResponseWriter interface {
	// Header returns the header map that will be sent by
	// WriteHeader. Changing the header after a call to
	// WriteHeader (or Write) has no effect unless the modified
	// headers were declared as trailers by setting the
	// "Trailer" header before the call to WriteHeader (see example).
	// To suppress implicit response headers, set their value to nil.
	Header() Header

	// Write writes the data to the connection as part of an HTTP reply.
	// If WriteHeader has not yet been called, Write calls WriteHeader(http.StatusOK)
	// before writing the data.  If the Header does not contain a
	// Content-Type line, Write adds a Content-Type set to the result of passing
	// the initial 512 bytes of written data to DetectContentType.
	Write([]byte) (int, error)

	// WriteHeader sends an HTTP response header with status code.
	// If WriteHeader is not called explicitly, the first call to Write
	// will trigger an implicit WriteHeader(http.StatusOK).
	// Thus explicit calls to WriteHeader are mainly used to
	// send error codes.
	WriteHeader(int)
}

// The Flusher interface is implemented by ResponseWriters that allow
// an HTTP handler to flush buffered data to the client.
//
// Note that even for ResponseWriters that support Flush,
// if the client is connected through an HTTP proxy,
// the buffered data may not reach the client until the response
// completes.
type Flusher interface {
	// Flush sends any buffered data to the client.
	Flush()
}

// The Hijacker interface is implemented by ResponseWriters that allow
// an HTTP handler to take over the connection.
type Hijacker interface {
	// Hijack lets the caller take over the connection.
	// After a call to Hijack(), the HTTP server library
	// will not do anything else with the connection.
	//
	// It becomes the caller's responsibility to manage
	// and close the connection.
	//
	// The returned net.Conn may have read or write deadlines
	// already set, depending on the configuration of the
	// Server. It is the caller's responsibility to set
	// or clear those deadlines as needed.
	Hijack() (net.Conn, *bufio.ReadWriter, error)
}

// The CloseNotifier interface is implemented by ResponseWriters which
// allow detecting when the underlying connection has gone away.
//
// This mechanism can be used to cancel long operations on the server
// if the client has disconnected before the response is ready.
type CloseNotifier interface {
	// CloseNotify returns a channel that receives a single value
	// when the client connection has gone away.
	CloseNotify() <-chan bool
}

// A conn represents the server side of an HTTP connection.
type conn struct {
	remoteAddr string               // network address of remote side
	server     *Server              // the Server on which the connection arrived
	rwc        net.Conn             // i/o connection
	w          io.Writer            // checkConnErrorWriter's copy of wrc, not zeroed on Hijack
	werr       error                // any errors writing to w
	sr         liveSwitchReader     // where the LimitReader reads from; usually the rwc
	lr         *io.LimitedReader    // io.LimitReader(sr)
	buf        *bufio.ReadWriter    // buffered(lr,rwc), reading from bufio->limitReader->sr->rwc
	tlsState   *tls.ConnectionState // or nil when not using TLS
	lastMethod string               // method of previous request, or ""

	mu           sync.Mutex // guards the following
	clientGone   bool       // if client has disconnected mid-request
	closeNotifyc chan bool  // made lazily
	hijackedv    bool       // connection has been hijacked by handler
}

func (c *conn) hijacked() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hijackedv
}

func (c *conn) hijack() (rwc net.Conn, buf *bufio.ReadWriter, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.hijackedv {
		return nil, nil, ErrHijacked
	}
	if c.closeNotifyc != nil {
		return nil, nil, errors.New("http: Hijack is incompatible with use of CloseNotifier")
	}
	c.hijackedv = true
	rwc = c.rwc
	buf = c.buf
	c.rwc = nil
	c.buf = nil
	c.setState(rwc, StateHijacked)
	return
}

func (c *conn) closeNotify() <-chan bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closeNotifyc == nil {
		c.closeNotifyc = make(chan bool, 1)
		if c.hijackedv {
			// to obey the function signature, even though
			// it'll never receive a value.
			return c.closeNotifyc
		}
		pr, pw := io.Pipe()

		readSource := c.sr.r
		c.sr.Lock()
		c.sr.r = pr
		c.sr.Unlock()
		go func() {
			_, err := io.Copy(pw, readSource)
			if err == nil {
				err = io.EOF
			}
			pw.CloseWithError(err)
			c.noteClientGone()
		}()
	}
	return c.closeNotifyc
}

func (c *conn) noteClientGone() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closeNotifyc != nil && !c.clientGone {
		c.closeNotifyc <- true
	}
	c.clientGone = true
}

// A switchWriter can have its Writer changed at runtime.
// It's not safe for concurrent Writes and switches.
type switchWriter struct {
	io.Writer
}

// A liveSwitchReader can have its Reader changed at runtime. It's
// safe for concurrent reads and switches, if its mutex is held.
type liveSwitchReader struct {
	sync.Mutex
	r io.Reader
}

func (sr *liveSwitchReader) Read(p []byte) (n int, err error) {
	sr.Lock()
	r := sr.r
	sr.Unlock()
	return r.Read(p)
}

// This should be >= 512 bytes for DetectContentType,
// but otherwise it's somewhat arbitrary.
const bufferBeforeChunkingSize = 2048

// chunkWriter writes to a response's conn buffer, and is the writer
// wrapped by the response.bufw buffered writer.
//
// chunkWriter also is responsible for finalizing the Header, including
// conditionally setting the Content-Type and setting a Content-Length
// in cases where the handler's final output is smaller than the buffer
// size. It also conditionally adds chunk headers, when in chunking mode.
//
// See the comment above (*response).Write for the entire write flow.
type chunkWriter struct {
	res *response

	// header is either nil or a deep clone of res.handlerHeader
	// at the time of res.WriteHeader, if res.WriteHeader is
	// called and extra buffering is being done to calculate
	// Content-Type and/or Content-Length.
	header Header

	// wroteHeader tells whether the header's been written to "the
	// wire" (or rather: w.conn.buf). this is unlike
	// (*response).wroteHeader, which tells only whether it was
	// logically written.
	wroteHeader bool

	// set by the writeHeader method:
	chunking bool // using chunked transfer encoding for reply body
}

var (
	crlf       = []byte("\r\n")
	colonSpace = []byte(": ")
)

func (cw *chunkWriter) Write(p []byte) (n int, err error) {
	if !cw.wroteHeader {
		cw.writeHeader(p)
	}
	if cw.res.req.Method == "HEAD" {
		// Eat writes.
		return len(p), nil
	}
	if cw.chunking {
		_, err = fmt.Fprintf(cw.res.conn.buf, "%x\r\n", len(p))
		if err != nil {
			cw.res.conn.rwc.Close()
			return
		}
	}
	n, err = cw.res.conn.buf.Write(p)
	if cw.chunking && err == nil {
		_, err = cw.res.conn.buf.Write(crlf)
	}
	if err != nil {
		cw.res.conn.rwc.Close()
	}
	return
}

func (cw *chunkWriter) flush() {
	if !cw.wroteHeader {
		cw.writeHeader(nil)
	}
	cw.res.conn.buf.Flush()
}

func (cw *chunkWriter) close() {
	if !cw.wroteHeader {
		cw.writeHeader(nil)
	}
	if cw.chunking {
		bw := cw.res.conn.buf // conn's bufio writer
		// zero chunk to mark EOF
		bw.WriteString("0\r\n")
		if len(cw.res.trailers) > 0 {
			trailers := make(Header)
			for _, h := range cw.res.trailers {
				if vv := cw.res.handlerHeader[h]; len(vv) > 0 {
					trailers[h] = vv
				}
			}
			trailers.Write(bw) // the writer handles noting errors
		}
		// final blank line after the trailers (whether
		// present or not)
		bw.WriteString("\r\n")
	}
}

// A response represents the server side of an HTTP response.
type response struct {
	conn          *conn
	req           *Request // request for this response
	wroteHeader   bool     // reply header has been (logically) written
	wroteContinue bool     // 100 Continue response was written

	w  *bufio.Writer // buffers output in chunks to chunkWriter
	cw chunkWriter
	sw *switchWriter // of the bufio.Writer, for return to putBufioWriter

	// handlerHeader is the Header that Handlers get access to,
	// which may be retained and mutated even after WriteHeader.
	// handlerHeader is copied into cw.header at WriteHeader
	// time, and privately mutated thereafter.
	handlerHeader Header
	calledHeader  bool // handler accessed handlerHeader via Header

	written       int64 // number of bytes written in body
	contentLength int64 // explicitly-declared Content-Length; or -1
	status        int   // status code passed to WriteHeader

	// close connection after this reply.  set on request and
	// updated after response from handler if there's a
	// "Connection: keep-alive" response header and a
	// Content-Length.
	closeAfterReply bool

	// requestBodyLimitHit is set by requestTooLarge when
	// maxBytesReader hits its max size. It is checked in
	// WriteHeader, to make sure we don't consume the
	// remaining request body to try to advance to the next HTTP
	// request. Instead, when this is set, we stop reading
	// subsequent requests on this connection and stop reading
	// input from it.
	requestBodyLimitHit bool

	// trailers are the headers to be sent after the handler
	// finishes writing the body.  This field is initialized from
	// the Trailer response header when the response header is
	// written.
	trailers []string

	handlerDone bool // set true when the handler exits

	// Buffers for Date and Content-Length
	dateBuf [len(TimeFormat)]byte
	clenBuf [10]byte
}

// declareTrailer is called for each Trailer header when the
// response header is written. It notes that a header will need to be
// written in the trailers at the end of the response.
func (w *response) declareTrailer(k string) {
	k = CanonicalHeaderKey(k)
	switch k {
	case "Transfer-Encoding", "Content-Length", "Trailer":
		// Forbidden by RFC 2616 14.40.
		return
	}
	w.trailers = append(w.trailers, k)
}

// requestTooLarge is called by maxBytesReader when too much input has
// been read from the client.
func (w *response) requestTooLarge() {
	w.closeAfterReply = true
	w.requestBodyLimitHit = true
	if !w.wroteHeader {
		w.Header().Set("Connection", "close")
	}
}

// needsSniff reports whether a Content-Type still needs to be sniffed.
func (w *response) needsSniff() bool {
	_, haveType := w.handlerHeader["Content-Type"]
	return !w.cw.wroteHeader && !haveType && w.written < sniffLen
}

// writerOnly hides an io.Writer value's optional ReadFrom method
// from io.Copy.
type writerOnly struct {
	io.Writer
}

func srcIsRegularFile(src io.Reader) (isRegular bool, err error) {
	switch v := src.(type) {
	case *os.File:
		fi, err := v.Stat()
		if err != nil {
			return false, err
		}
		return fi.Mode().IsRegular(), nil
	case *io.LimitedReader:
		return srcIsRegularFile(v.R)
	default:
		return
	}
}

// ReadFrom is here to optimize copying from an *os.File regular file
// to a *net.TCPConn with sendfile.
func (w *response) ReadFrom(src io.Reader) (n int64, err error) {
	// Our underlying w.conn.rwc is usually a *TCPConn (with its
	// own ReadFrom method). If not, or if our src isn't a regular
	// file, just fall back to the normal copy method.
	rf, ok := w.conn.rwc.(io.ReaderFrom)
	regFile, err := srcIsRegularFile(src)
	if err != nil {
		return 0, err
	}
	if !ok || !regFile {
		return io.Copy(writerOnly{w}, src)
	}

	// sendfile path:

	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}

	if w.needsSniff() {
		n0, err := io.Copy(writerOnly{w}, io.LimitReader(src, sniffLen))
		n += n0
		if err != nil {
			return n, err
		}
	}

	w.w.Flush()  // get rid of any previous writes
	w.cw.flush() // make sure Header is written; flush data to rwc

	// Now that cw has been flushed, its chunking field is guaranteed initialized.
	if !w.cw.chunking && w.bodyAllowed() {
		n0, err := rf.ReadFrom(src)
		n += n0
		w.written += n0
		return n, err
	}

	n0, err := io.Copy(writerOnly{w}, src)
	n += n0
	return n, err
}

// noLimit is an effective infinite upper bound for io.LimitedReader
const noLimit int64 = (1 << 63) - 1

// debugServerConnections controls whether all server connections are wrapped
// with a verbose logging wrapper.
const debugServerConnections = false

// Create new connection from rwc.
func (srv *Server) newConn(rwc net.Conn) (c *conn, err error) {
	c = new(conn)
	c.remoteAddr = rwc.RemoteAddr().String()
	c.server = srv
	c.rwc = rwc
	c.w = rwc
	if debugServerConnections {
		c.rwc = newLoggingConn("server", c.rwc)
	}
	c.sr.r = c.rwc
	c.lr = io.LimitReader(&c.sr, noLimit).(*io.LimitedReader)
	br := newBufioReader(c.lr)
	bw := newBufioWriterSize(checkConnErrorWriter{c}, 4<<10)
	c.buf = bufio.NewReadWriter(br, bw)
	return c, nil
}

var (
	bufioReaderPool   sync.Pool
	bufioWriter2kPool sync.Pool
	bufioWriter4kPool sync.Pool
)

func bufioWriterPool(size int) *sync.Pool {
	switch size {
	case 2 << 10:
		return &bufioWriter2kPool
	case 4 << 10:
		return &bufioWriter4kPool
	}
	return nil
}

func newBufioReader(r io.Reader) *bufio.Reader {
	if v := bufioReaderPool.Get(); v != nil {
		br := v.(*bufio.Reader)
		br.Reset(r)
		return br
	}
	// Note: if this reader size is every changed, update
	// TestHandlerBodyClose's assumptions.
	return bufio.NewReader(r)
}

func putBufioReader(br *bufio.Reader) {
	br.Reset(nil)
	bufioReaderPool.Put(br)
}

func newBufioWriterSize(w io.Writer, size int) *bufio.Writer {
	pool := bufioWriterPool(size)
	if pool != nil {
		if v := pool.Get(); v != nil {
			bw := v.(*bufio.Writer)
			bw.Reset(w)
			return bw
		}
	}
	return bufio.NewWriterSize(w, size)
}

func putBufioWriter(bw *bufio.Writer) {
	bw.Reset(nil)
	if pool := bufioWriterPool(bw.Available()); pool != nil {
		pool.Put(bw)
	}
}

// DefaultMaxHeaderBytes is the maximum permitted size of the headers
// in an HTTP request.
// This can be overridden by setting Server.MaxHeaderBytes.
const DefaultMaxHeaderBytes = 1 << 20 // 1 MB

func (srv *Server) maxHeaderBytes() int {
	if srv.MaxHeaderBytes > 0 {
		return srv.MaxHeaderBytes
	}
	return DefaultMaxHeaderBytes
}

func (srv *Server) initialLimitedReaderSize() int64 {
	return int64(srv.maxHeaderBytes()) + 4096 // bufio slop
}

// wrapper around io.ReaderCloser which on first read, sends an
// HTTP/1.1 100 Continue header
type expectContinueReader struct {
	resp       *response
	readCloser io.ReadCloser
	closed     bool
	sawEOF     bool
}

func (ecr *expectContinueReader) Read(p []byte) (n int, err error) {
	if ecr.closed {
		return 0, ErrBodyReadAfterClose
	}
	if !ecr.resp.wroteContinue && !ecr.resp.conn.hijacked() {
		ecr.resp.wroteContinue = true
		ecr.resp.conn.buf.WriteString("HTTP/1.1 100 Continue\r\n\r\n")
		ecr.resp.conn.buf.Flush()
	}
	n, err = ecr.readCloser.Read(p)
	if err == io.EOF {
		ecr.sawEOF = true
	}
	return
}

func (ecr *expectContinueReader) Close() error {
	ecr.closed = true
	return ecr.readCloser.Close()
}

// TimeFormat is the time format to use with
// time.Parse and time.Time.Format when parsing
// or generating times in HTTP headers.
// It is like time.RFC1123 but hard codes GMT as the time zone.
const TimeFormat = "Mon, 02 Jan 2006 15:04:05 GMT"

// appendTime is a non-allocating version of []byte(t.UTC().Format(TimeFormat))
func appendTime(b []byte, t time.Time) []byte {
	const days = "SunMonTueWedThuFriSat"
	const months = "JanFebMarAprMayJunJulAugSepOctNovDec"

	t = t.UTC()
	yy, mm, dd := t.Date()
	hh, mn, ss := t.Clock()
	day := days[3*t.Weekday():]
	mon := months[3*(mm-1):]

	return append(b,
		day[0], day[1], day[2], ',', ' ',
		byte('0'+dd/10), byte('0'+dd%10), ' ',
		mon[0], mon[1], mon[2], ' ',
		byte('0'+yy/1000), byte('0'+(yy/100)%10), byte('0'+(yy/10)%10), byte('0'+yy%10), ' ',
		byte('0'+hh/10), byte('0'+hh%10), ':',
		byte('0'+mn/10), byte('0'+mn%10), ':',
		byte('0'+ss/10), byte('0'+ss%10), ' ',
		'G', 'M', 'T')
}

var errTooLarge = errors.New("http: request too large")

// Read next request from connection.
func (c *conn) readRequest() (w *response, err error) {
	if c.hijacked() {
		return nil, ErrHijacked
	}

	if d := c.server.ReadTimeout; d != 0 {
		c.rwc.SetReadDeadline(time.Now().Add(d))
	}
	if d := c.server.WriteTimeout; d != 0 {
		defer func() {
			c.rwc.SetWriteDeadline(time.Now().Add(d))
		}()
	}

	c.lr.N = c.server.initialLimitedReaderSize()
	if c.lastMethod == "POST" {
		// RFC 2616 section 4.1 tolerance for old buggy clients.
		peek, _ := c.buf.Reader.Peek(4) // ReadRequest will get err below
		c.buf.Reader.Discard(numLeadingCRorLF(peek))
	}
	var req *Request
	if req, err = ReadRequest(c.buf.Reader); err != nil {
		if c.lr.N == 0 {
			return nil, errTooLarge
		}
		return nil, err
	}
	c.lr.N = noLimit
	c.lastMethod = req.Method

	req.RemoteAddr = c.remoteAddr
	req.TLS = c.tlsState
	if body, ok := req.Body.(*body); ok {
		body.doEarlyClose = true
	}

	w = &response{
		conn:          c,
		req:           req,
		handlerHeader: make(Header),
		contentLength: -1,
	}
	w.cw.res = w
	w.w = newBufioWriterSize(&w.cw, bufferBeforeChunkingSize)
	return w, nil
}

func (w *response) Header() Header {
	if w.cw.header == nil && w.wroteHeader && !w.cw.wroteHeader {
		// Accessing the header between logically writing it
		// and physically writing it means we need to allocate
		// a clone to snapshot the logically written state.
		w.cw.header = w.handlerHeader.clone()
	}
	w.calledHeader = true
	return w.handlerHeader
}

// maxPostHandlerReadBytes is the max number of Request.Body bytes not
// consumed by a handler that the server will read from the client
// in order to keep a connection alive.  If there are more bytes than
// this then the server to be paranoid instead sends a "Connection:
// close" response.
//
// This number is approximately what a typical machine's TCP buffer
// size is anyway.  (if we have the bytes on the machine, we might as
// well read them)
const maxPostHandlerReadBytes = 256 << 10

func (w *response) WriteHeader(code int) {
	if w.conn.hijacked() {
		w.conn.server.logf("http: response.WriteHeader on hijacked connection")
		return
	}
	if w.wroteHeader {
		w.conn.server.logf("http: multiple response.WriteHeader calls")
		return
	}
	w.wroteHeader = true
	w.status = code

	if w.calledHeader && w.cw.header == nil {
		w.cw.header = w.handlerHeader.clone()
	}

	if cl := w.handlerHeader.get("Content-Length"); cl != "" {
		v, err := strconv.ParseInt(cl, 10, 64)
		if err == nil && v >= 0 {
			w.contentLength = v
		} else {
			w.conn.server.logf("http: invalid Content-Length of %q", cl)
			w.handlerHeader.Del("Content-Length")
		}
	}
}

// extraHeader is the set of headers sometimes added by chunkWriter.writeHeader.
// This type is used to avoid extra allocations from cloning and/or populating
// the response Header map and all its 1-element slices.
type extraHeader struct {
	contentType      string
	connection       string
	transferEncoding string
	date             []byte // written if not nil
	contentLength    []byte // written if not nil
}

// Sorted the same as extraHeader.Write's loop.
var extraHeaderKeys = [][]byte{
	[]byte("Content-Type"),
	[]byte("Connection"),
	[]byte("Transfer-Encoding"),
}

var (
	headerContentLength = []byte("Content-Length: ")
	headerDate          = []byte("Date: ")
)

// Write writes the headers described in h to w.
//
// This method has a value receiver, despite the somewhat large size
// of h, because it prevents an allocation. The escape analysis isn't
// smart enough to realize this function doesn't mutate h.
func (h extraHeader) Write(w *bufio.Writer) {
	if h.date != nil {
		w.Write(headerDate)
		w.Write(h.date)
		w.Write(crlf)
	}
	if h.contentLength != nil {
		w.Write(headerContentLength)
		w.Write(h.contentLength)
		w.Write(crlf)
	}
	for i, v := range []string{h.contentType, h.connection, h.transferEncoding} {
		if v != "" {
			w.Write(extraHeaderKeys[i])
			w.Write(colonSpace)
			w.WriteString(v)
			w.Write(crlf)
		}
	}
}

// writeHeader finalizes the header sent to the client and writes it
// to cw.res.conn.buf.
//
// p is not written by writeHeader, but is the first chunk of the body
// that will be written.  It is sniffed for a Content-Type if none is
// set explicitly.  It's also used to set the Content-Length, if the
// total body size was small and the handler has already finished
// running.
func (cw *chunkWriter) writeHeader(p []byte) {
	if cw.wroteHeader {
		return
	}
	cw.wroteHeader = true

	w := cw.res
	keepAlivesEnabled := w.conn.server.doKeepAlives()
	isHEAD := w.req.Method == "HEAD"

	// header is written out to w.conn.buf below. Depending on the
	// state of the handler, we either own the map or not. If we
	// don't own it, the exclude map is created lazily for
	// WriteSubset to remove headers. The setHeader struct holds
	// headers we need to add.
	header := cw.header
	owned := header != nil
	if !owned {
		header = w.handlerHeader
	}
	var excludeHeader map[string]bool
	delHeader := func(key string) {
		if owned {
			header.Del(key)
			return
		}
		if _, ok := header[key]; !ok {
			return
		}
		if excludeHeader == nil {
			excludeHeader = make(map[string]bool)
		}
		excludeHeader[key] = true
	}
	var setHeader extraHeader

	trailers := false
	for _, v := range cw.header["Trailer"] {
		trailers = true
		foreachHeaderElement(v, cw.res.declareTrailer)
	}

	te := header.get("Transfer-Encoding")
	hasTE := te != ""

	// If the handler is done but never sent a Content-Length
	// response header and this is our first (and last) write, set
	// it, even to zero. This helps HTTP/1.0 clients keep their
	// "keep-alive" connections alive.
	// Exceptions: 304/204/1xx responses never get Content-Length, and if
	// it was a HEAD request, we don't know the difference between
	// 0 actual bytes and 0 bytes because the handler noticed it
	// was a HEAD request and chose not to write anything.  So for
	// HEAD, the handler should either write the Content-Length or
	// write non-zero bytes.  If it's actually 0 bytes and the
	// handler never looked at the Request.Method, we just don't
	// send a Content-Length header.
	// Further, we don't send an automatic Content-Length if they
	// set a Transfer-Encoding, because they're generally incompatible.
	if w.handlerDone && !trailers && !hasTE && bodyAllowedForStatus(w.status) && header.get("Content-Length") == "" && (!isHEAD || len(p) > 0) {
		w.contentLength = int64(len(p))
		setHeader.contentLength = strconv.AppendInt(cw.res.clenBuf[:0], int64(len(p)), 10)
	}

	// If this was an HTTP/1.0 request with keep-alive and we sent a
	// Content-Length back, we can make this a keep-alive response ...
	if w.req.wantsHttp10KeepAlive() && keepAlivesEnabled {
		sentLength := header.get("Content-Length") != ""
		if sentLength && header.get("Connection") == "keep-alive" {
			w.closeAfterReply = false
		}
	}

	// Check for a explicit (and valid) Content-Length header.
	hasCL := w.contentLength != -1

	if w.req.wantsHttp10KeepAlive() && (isHEAD || hasCL) {
		_, connectionHeaderSet := header["Connection"]
		if !connectionHeaderSet {
			setHeader.connection = "keep-alive"
		}
	} else if !w.req.ProtoAtLeast(1, 1) || w.req.wantsClose() {
		w.closeAfterReply = true
	}

	if header.get("Connection") == "close" || !keepAlivesEnabled {
		w.closeAfterReply = true
	}

	// If the client wanted a 100-continue but we never sent it to
	// them (or, more strictly: we never finished reading their
	// request body), don't reuse this connection because it's now
	// in an unknown state: we might be sending this response at
	// the same time the client is now sending its request body
	// after a timeout.  (Some HTTP clients send Expect:
	// 100-continue but knowing that some servers don't support
	// it, the clients set a timer and send the body later anyway)
	// If we haven't seen EOF, we can't skip over the unread body
	// because we don't know if the next bytes on the wire will be
	// the body-following-the-timer or the subsequent request.
	// See Issue 11549.
	if ecr, ok := w.req.Body.(*expectContinueReader); ok && !ecr.sawEOF {
		w.closeAfterReply = true
	}

	// Per RFC 2616, we should consume the request body before
	// replying, if the handler hasn't already done so.  But we
	// don't want to do an unbounded amount of reading here for
	// DoS reasons, so we only try up to a threshold.
	if w.req.ContentLength != 0 && !w.closeAfterReply {
		var discard, tooBig bool

		switch bdy := w.req.Body.(type) {
		case *expectContinueReader:
			if bdy.resp.wroteContinue {
				discard = true
			}
		case *body:
			bdy.mu.Lock()
			switch {
			case bdy.closed:
				if !bdy.sawEOF {
					// Body was closed in handler with non-EOF error.
					w.closeAfterReply = true
				}
			case bdy.unreadDataSizeLocked() >= maxPostHandlerReadBytes:
				tooBig = true
			default:
				discard = true
			}
			bdy.mu.Unlock()
		default:
			discard = true
		}

		if discard {
			_, err := io.CopyN(ioutil.Discard, w.req.Body, maxPostHandlerReadBytes+1)
			switch err {
			case nil:
				// There must be even more data left over.
				tooBig = true
			case ErrBodyReadAfterClose:
				// Body was already consumed and closed.
			case io.EOF:
				// The remaining body was just consumed, close it.
				err = w.req.Body.Close()
				if err != nil {
					w.closeAfterReply = true
				}
			default:
				// Some other kind of error occured, like a read timeout, or
				// corrupt chunked encoding. In any case, whatever remains
				// on the wire must not be parsed as another HTTP request.
				w.closeAfterReply = true
			}
		}

		if tooBig {
			w.requestTooLarge()
			delHeader("Connection")
			setHeader.connection = "close"
		}
	}

	code := w.status
	if bodyAllowedForStatus(code) {
		// If no content type, apply sniffing algorithm to body.
		_, haveType := header["Content-Type"]
		if !haveType && !hasTE {
			setHeader.contentType = DetectContentType(p)
		}
	} else {
		for _, k := range suppressedHeaders(code) {
			delHeader(k)
		}
	}

	if _, ok := header["Date"]; !ok {
		setHeader.date = appendTime(cw.res.dateBuf[:0], time.Now())
	}

	if hasCL && hasTE && te != "identity" {
		// TODO: return an error if WriteHeader gets a return parameter
		// For now just ignore the Content-Length.
		w.conn.server.logf("http: WriteHeader called with both Transfer-Encoding of %q and a Content-Length of %d",
			te, w.contentLength)
		delHeader("Content-Length")
		hasCL = false
	}

	if w.req.Method == "HEAD" || !bodyAllowedForStatus(code) {
		// do nothing
	} else if code == StatusNoContent {
		delHeader("Transfer-Encoding")
	} else if hasCL {
		delHeader("Transfer-Encoding")
	} else if w.req.ProtoAtLeast(1, 1) {
		// HTTP/1.1 or greater: Transfer-Encoding has been set to identity,  and no
		// content-length has been provided. The connection must be closed after the
		// reply is written, and no chunking is to be done. This is the setup
		// recommended in the Server-Sent Events candidate recommendation 11,
		// section 8.
		if hasTE && te == "identity" {
			cw.chunking = false
			w.closeAfterReply = true
		} else {
			// HTTP/1.1 or greater: use chunked transfer encoding
			// to avoid closing the connection at EOF.
			cw.chunking = true
			setHeader.transferEncoding = "chunked"
		}
	} else {
		// HTTP version < 1.1: cannot do chunked transfer
		// encoding and we don't know the Content-Length so
		// signal EOF by closing connection.
		w.closeAfterReply = true
		delHeader("Transfer-Encoding") // in case already set
	}

	// Cannot use Content-Length with non-identity Transfer-Encoding.
	if cw.chunking {
		delHeader("Content-Length")
	}
	if !w.req.ProtoAtLeast(1, 0) {
		return
	}

	if w.closeAfterReply && (!keepAlivesEnabled || !hasToken(cw.header.get("Connection"), "close")) {
		delHeader("Connection")
		if w.req.ProtoAtLeast(1, 1) {
			setHeader.connection = "close"
		}
	}

	w.conn.buf.WriteString(statusLine(w.req, code))
	cw.header.WriteSubset(w.conn.buf, excludeHeader)
	setHeader.Write(w.conn.buf.Writer)
	w.conn.buf.Write(crlf)
}

// foreachHeaderElement splits v according to the "#rule" construction
// in RFC 2616 section 2.1 and calls fn for each non-empty element.
func foreachHeaderElement(v string, fn func(string)) {
	v = textproto.TrimString(v)
	if v == "" {
		return
	}
	if !strings.Contains(v, ",") {
		fn(v)
		return
	}
	for _, f := range strings.Split(v, ",") {
		if f = textproto.TrimString(f); f != "" {
			fn(f)
		}
	}
}

// statusLines is a cache of Status-Line strings, keyed by code (for
// HTTP/1.1) or negative code (for HTTP/1.0). This is faster than a
// map keyed by struct of two fields. This map's max size is bounded
// by 2*len(statusText), two protocol types for each known official
// status code in the statusText map.
var (
	statusMu    sync.RWMutex
	statusLines = make(map[int]string)
)

// statusLine returns a response Status-Line (RFC 2616 Section 6.1)
// for the given request and response status code.
func statusLine(req *Request, code int) string {
	// Fast path:
	key := code
	proto11 := req.ProtoAtLeast(1, 1)
	if !proto11 {
		key = -key
	}
	statusMu.RLock()
	line, ok := statusLines[key]
	statusMu.RUnlock()
	if ok {
		return line
	}

	// Slow path:
	proto := "HTTP/1.0"
	if proto11 {
		proto = "HTTP/1.1"
	}
	codestring := strconv.Itoa(code)
	text, ok := statusText[code]
	if !ok {
		text = "status code " + codestring
	}
	line = proto + " " + codestring + " " + text + "\r\n"
	if ok {
		statusMu.Lock()
		defer statusMu.Unlock()
		statusLines[key] = line
	}
	return line
}

// bodyAllowed reports whether a Write is allowed for this response type.
// It's illegal to call this before the header has been flushed.
func (w *response) bodyAllowed() bool {
	if !w.wroteHeader {
		panic("")
	}
	return bodyAllowedForStatus(w.status)
}

// The Life Of A Write is like this:
//
// Handler starts. No header has been sent. The handler can either
// write a header, or just start writing.  Writing before sending a header
// sends an implicitly empty 200 OK header.
//
// If the handler didn't declare a Content-Length up front, we either
// go into chunking mode or, if the handler finishes running before
// the chunking buffer size, we compute a Content-Length and send that
// in the header instead.
//
// Likewise, if the handler didn't set a Content-Type, we sniff that
// from the initial chunk of output.
//
// The Writers are wired together like:
//
// 1. *response (the ResponseWriter) ->
// 2. (*response).w, a *bufio.Writer of bufferBeforeChunkingSize bytes
// 3. chunkWriter.Writer (whose writeHeader finalizes Content-Length/Type)
//    and which writes the chunk headers, if needed.
// 4. conn.buf, a bufio.Writer of default (4kB) bytes, writing to ->
// 5. checkConnErrorWriter{c}, which notes any non-nil error on Write
//    and populates c.werr with it if so. but otherwise writes to:
// 6. the rwc, the net.Conn.
//
// TODO(bradfitz): short-circuit some of the buffering when the
// initial header contains both a Content-Type and Content-Length.
// Also short-circuit in (1) when the header's been sent and not in
// chunking mode, writing directly to (4) instead, if (2) has no
// buffered data.  More generally, we could short-circuit from (1) to
// (3) even in chunking mode if the write size from (1) is over some
// threshold and nothing is in (2).  The answer might be mostly making
// bufferBeforeChunkingSize smaller and having bufio's fast-paths deal
// with this instead.
func (w *response) Write(data []byte) (n int, err error) {
	return w.write(len(data), data, "")
}

func (w *response) WriteString(data string) (n int, err error) {
	return w.write(len(data), nil, data)
}

// either dataB or dataS is non-zero.
func (w *response) write(lenData int, dataB []byte, dataS string) (n int, err error) {
	if w.conn.hijacked() {
		w.conn.server.logf("http: response.Write on hijacked connection")
		return 0, ErrHijacked
	}
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	if lenData == 0 {
		return 0, nil
	}
	if !w.bodyAllowed() {
		return 0, ErrBodyNotAllowed
	}

	w.written += int64(lenData) // ignoring errors, for errorKludge
	if w.contentLength != -1 && w.written > w.contentLength {
		return 0, ErrContentLength
	}
	if dataB != nil {
		return w.w.Write(dataB)
	} else {
		return w.w.WriteString(dataS)
	}
}

func (w *response) finishRequest() {
	w.handlerDone = true

	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}

	w.w.Flush()
	putBufioWriter(w.w)
	w.cw.close()
	w.conn.buf.Flush()

	// Close the body (regardless of w.closeAfterReply) so we can
	// re-use its bufio.Reader later safely.
	w.req.Body.Close()

	if w.req.MultipartForm != nil {
		w.req.MultipartForm.RemoveAll()
	}
}

// shouldReuseConnection reports whether the underlying TCP connection can be reused.
// It must only be called after the handler is done executing.
func (w *response) shouldReuseConnection() bool {
	if w.closeAfterReply {
		// The request or something set while executing the
		// handler indicated we shouldn't reuse this
		// connection.
		return false
	}

	if w.req.Method != "HEAD" && w.contentLength != -1 && w.bodyAllowed() && w.contentLength != w.written {
		// Did not write enough. Avoid getting out of sync.
		return false
	}

	// There was some error writing to the underlying connection
	// during the request, so don't re-use this conn.
	if w.conn.werr != nil {
		return false
	}

	if w.closedRequestBodyEarly() {
		return false
	}

	return true
}

func (w *response) closedRequestBodyEarly() bool {
	body, ok := w.req.Body.(*body)
	return ok && body.didEarlyClose()
}

func (w *response) Flush() {
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	w.w.Flush()
	w.cw.flush()
}

func (c *conn) finalFlush() {
	if c.buf != nil {
		c.buf.Flush()

		// Steal the bufio.Reader (~4KB worth of memory) and its associated
		// reader for a future connection.
		putBufioReader(c.buf.Reader)

		// Steal the bufio.Writer (~4KB worth of memory) and its associated
		// writer for a future connection.
		putBufioWriter(c.buf.Writer)

		c.buf = nil
	}
}

// Close the connection.
func (c *conn) close() {
	c.finalFlush()
	if c.rwc != nil {
		c.rwc.Close()
		c.rwc = nil
	}
}

// rstAvoidanceDelay is the amount of time we sleep after closing the
// write side of a TCP connection before closing the entire socket.
// By sleeping, we increase the chances that the client sees our FIN
// and processes its final data before they process the subsequent RST
// from closing a connection with known unread data.
// This RST seems to occur mostly on BSD systems. (And Windows?)
// This timeout is somewhat arbitrary (~latency around the planet).
const rstAvoidanceDelay = 500 * time.Millisecond

type closeWriter interface {
	CloseWrite() error
}

var _ closeWriter = (*net.TCPConn)(nil)

// closeWrite flushes any outstanding data and sends a FIN packet (if
// client is connected via TCP), signalling that we're done.  We then
// pause for a bit, hoping the client processes it before any
// subsequent RST.
//
// See https://golang.org/issue/3595
func (c *conn) closeWriteAndWait() {
	c.finalFlush()
	if tcp, ok := c.rwc.(closeWriter); ok {
		tcp.CloseWrite()
	}
	time.Sleep(rstAvoidanceDelay)
}

// validNPN reports whether the proto is not a blacklisted Next
// Protocol Negotiation protocol.  Empty and built-in protocol types
// are blacklisted and can't be overridden with alternate
// implementations.
func validNPN(proto string) bool {
	switch proto {
	case "", "http/1.1", "http/1.0":
		return false
	}
	return true
}

func (c *conn) setState(nc net.Conn, state ConnState) {
	if hook := c.server.ConnState; hook != nil {
		hook(nc, state)
	}
}

// Serve a new connection.
func (c *conn) serve() {
	origConn := c.rwc // copy it before it's set nil on Close or Hijack
	defer func() {
		if err := recover(); err != nil {
			const size = 64 << 10
			buf := make([]byte, size)
			buf = buf[:runtime.Stack(buf, false)]
			c.server.logf("http: panic serving %v: %v\n%s", c.remoteAddr, err, buf)
		}
		if !c.hijacked() {
			c.close()
			c.setState(origConn, StateClosed)
		}
	}()

	if tlsConn, ok := c.rwc.(*tls.Conn); ok {
		if d := c.server.ReadTimeout; d != 0 {
			c.rwc.SetReadDeadline(time.Now().Add(d))
		}
		if d := c.server.WriteTimeout; d != 0 {
			c.rwc.SetWriteDeadline(time.Now().Add(d))
		}
		if err := tlsConn.Handshake(); err != nil {
			c.server.logf("http: TLS handshake error from %s: %v", c.rwc.RemoteAddr(), err)
			return
		}
		c.tlsState = new(tls.ConnectionState)
		*c.tlsState = tlsConn.ConnectionState()
		if proto := c.tlsState.NegotiatedProtocol; validNPN(proto) {
			if fn := c.server.TLSNextProto[proto]; fn != nil {
				h := initNPNRequest{tlsConn, serverHandler{c.server}}
				fn(c.server, tlsConn, h)
			}
			return
		}
	}

	for {
		w, err := c.readRequest()
		if c.lr.N != c.server.initialLimitedReaderSize() {
			// If we read any bytes off the wire, we're active.
			c.setState(c.rwc, StateActive)
		}
		if err != nil {
			if err == errTooLarge {
				// Their HTTP client may or may not be
				// able to read this if we're
				// responding to them and hanging up
				// while they're still writing their
				// request.  Undefined behavior.
				io.WriteString(c.rwc, "HTTP/1.1 413 Request Entity Too Large\r\n\r\n")
				c.closeWriteAndWait()
				break
			} else if err == io.EOF {
				break // Don't reply
			} else if neterr, ok := err.(net.Error); ok && neterr.Timeout() {
				break // Don't reply
			}
			io.WriteString(c.rwc, "HTTP/1.1 400 Bad Request\r\n\r\n")
			break
		}

		// Expect 100 Continue support
		req := w.req
		if req.expectsContinue() {
			if req.ProtoAtLeast(1, 1) && req.ContentLength != 0 {
				// Wrap the Body reader with one that replies on the connection
				req.Body = &expectContinueReader{readCloser: req.Body, resp: w}
			}
			req.Header.Del("Expect")
		} else if req.Header.get("Expect") != "" {
			w.sendExpectationFailed()
			break
		}

		// HTTP cannot have multiple simultaneous active requests.[*]
		// Until the server replies to this request, it can't read another,
		// so we might as well run the handler in this goroutine.
		// [*] Not strictly true: HTTP pipelining.  We could let them all process
		// in parallel even if their responses need to be serialized.
		serverHandler{c.server}.ServeHTTP(w, w.req)
		if c.hijacked() {
			return
		}
		w.finishRequest()
		if !w.shouldReuseConnection() {
			if w.requestBodyLimitHit || w.closedRequestBodyEarly() {
				c.closeWriteAndWait()
			}
			break
		}
		c.setState(c.rwc, StateIdle)
	}
}

func (w *response) sendExpectationFailed() {
	// TODO(bradfitz): let ServeHTTP handlers handle
	// requests with non-standard expectation[s]? Seems
	// theoretical at best, and doesn't fit into the
	// current ServeHTTP model anyway.  We'd need to
	// make the ResponseWriter an optional
	// "ExpectReplier" interface or something.
	//
	// For now we'll just obey RFC 2616 14.20 which says
	// "If a server receives a request containing an
	// Expect field that includes an expectation-
	// extension that it does not support, it MUST
	// respond with a 417 (Expectation Failed) status."
	w.Header().Set("Connection", "close")
	w.WriteHeader(StatusExpectationFailed)
	w.finishRequest()
}

// Hijack implements the Hijacker.Hijack method. Our response is both a ResponseWriter
// and a Hijacker.
func (w *response) Hijack() (rwc net.Conn, buf *bufio.ReadWriter, err error) {
	if w.wroteHeader {
		w.cw.flush()
	}
	// Release the bufioWriter that writes to the chunk writer, it is not
	// used after a connection has been hijacked.
	rwc, buf, err = w.conn.hijack()
	if err == nil {
		putBufioWriter(w.w)
		w.w = nil
	}
	return rwc, buf, err
}

func (w *response) CloseNotify() <-chan bool {
	return w.conn.closeNotify()
}

// The HandlerFunc type is an adapter to allow the use of
// ordinary functions as HTTP handlers.  If f is a function
// with the appropriate signature, HandlerFunc(f) is a
// Handler object that calls f.
type HandlerFunc func(ResponseWriter, *Request)

// ServeHTTP calls f(w, r).
func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
	f(w, r)
}

// Helper handlers

// Error replies to the request with the specified error message and HTTP code.
// The error message should be plain text.
func Error(w ResponseWriter, error string, code int) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(code)
	fmt.Fprintln(w, error)
}

// NotFound replies to the request with an HTTP 404 not found error.
func NotFound(w ResponseWriter, r *Request) { Error(w, "404 page not found", StatusNotFound) }

// NotFoundHandler returns a simple request handler
// that replies to each request with a ``404 page not found'' reply.
func NotFoundHandler() Handler { return HandlerFunc(NotFound) }

// StripPrefix returns a handler that serves HTTP requests
// by removing the given prefix from the request URL's Path
// and invoking the handler h. StripPrefix handles a
// request for a path that doesn't begin with prefix by
// replying with an HTTP 404 not found error.
func StripPrefix(prefix string, h Handler) Handler {
	if prefix == "" {
		return h
	}
	return HandlerFunc(func(w ResponseWriter, r *Request) {
		if p := strings.TrimPrefix(r.URL.Path, prefix); len(p) < len(r.URL.Path) {
			r.URL.Path = p
			h.ServeHTTP(w, r)
		} else {
			NotFound(w, r)
		}
	})
}

// Redirect replies to the request with a redirect to url,
// which may be a path relative to the request path.
func Redirect(w ResponseWriter, r *Request, urlStr string, code int) {
	if u, err := url.Parse(urlStr); err == nil {
		// If url was relative, make absolute by
		// combining with request path.
		// The browser would probably do this for us,
		// but doing it ourselves is more reliable.

		// NOTE(rsc): RFC 2616 says that the Location
		// line must be an absolute URI, like
		// "http://www.google.com/redirect/",
		// not a path like "/redirect/".
		// Unfortunately, we don't know what to
		// put in the host name section to get the
		// client to connect to us again, so we can't
		// know the right absolute URI to send back.
		// Because of this problem, no one pays attention
		// to the RFC; they all send back just a new path.
		// So do we.
		oldpath := r.URL.Path
		if oldpath == "" { // should not happen, but avoid a crash if it does
			oldpath = "/"
		}
		if u.Scheme == "" {
			// no leading http://server
			if urlStr == "" || urlStr[0] != '/' {
				// make relative path absolute
				olddir, _ := path.Split(oldpath)
				urlStr = olddir + urlStr
			}

			var query string
			if i := strings.Index(urlStr, "?"); i != -1 {
				urlStr, query = urlStr[:i], urlStr[i:]
			}

			// clean up but preserve trailing slash
			trailing := strings.HasSuffix(urlStr, "/")
			urlStr = path.Clean(urlStr)
			if trailing && !strings.HasSuffix(urlStr, "/") {
				urlStr += "/"
			}
			urlStr += query
		}
	}

	w.Header().Set("Location", urlStr)
	w.WriteHeader(code)

	// RFC2616 recommends that a short note "SHOULD" be included in the
	// response because older user agents may not understand 301/307.
	// Shouldn't send the response for POST or HEAD; that leaves GET.
	if r.Method == "GET" {
		note := "<a href=\"" + htmlEscape(urlStr) + "\">" + statusText[code] + "</a>.\n"
		fmt.Fprintln(w, note)
	}
}

var htmlReplacer = strings.NewReplacer(
	"&", "&amp;",
	"<", "&lt;",
	">", "&gt;",
	// "&#34;" is shorter than "&quot;".
	`"`, "&#34;",
	// "&#39;" is shorter than "&apos;" and apos was not in HTML until HTML5.
	"'", "&#39;",
)

func htmlEscape(s string) string {
	return htmlReplacer.Replace(s)
}

// Redirect to a fixed URL
type redirectHandler struct {
	url  string
	code int
}

func (rh *redirectHandler) ServeHTTP(w ResponseWriter, r *Request) {
	Redirect(w, r, rh.url, rh.code)
}

// RedirectHandler returns a request handler that redirects
// each request it receives to the given url using the given
// status code.
func RedirectHandler(url string, code int) Handler {
	return &redirectHandler{url, code}
}

// ServeMux is an HTTP request multiplexer.
// It matches the URL of each incoming request against a list of registered
// patterns and calls the handler for the pattern that
// most closely matches the URL.
//
// Patterns name fixed, rooted paths, like "/favicon.ico",
// or rooted subtrees, like "/images/" (note the trailing slash).
// Longer patterns take precedence over shorter ones, so that
// if there are handlers registered for both "/images/"
// and "/images/thumbnails/", the latter handler will be
// called for paths beginning "/images/thumbnails/" and the
// former will receive requests for any other paths in the
// "/images/" subtree.
//
// Note that since a pattern ending in a slash names a rooted subtree,
// the pattern "/" matches all paths not matched by other registered
// patterns, not just the URL with Path == "/".
//
// Patterns may optionally begin with a host name, restricting matches to
// URLs on that host only.  Host-specific patterns take precedence over
// general patterns, so that a handler might register for the two patterns
// "/codesearch" and "codesearch.google.com/" without also taking over
// requests for "http://www.google.com/".
//
// ServeMux also takes care of sanitizing the URL request path,
// redirecting any request containing . or .. elements to an
// equivalent .- and ..-free URL.
type ServeMux struct {
	mu    sync.RWMutex
	m     map[string]muxEntry
	hosts bool // whether any patterns contain hostnames
}

type muxEntry struct {
	explicit bool
	h        Handler
	pattern  string
}

// NewServeMux allocates and returns a new ServeMux.
func NewServeMux() *ServeMux { return &ServeMux{m: make(map[string]muxEntry)} }

// DefaultServeMux is the default ServeMux used by Serve.
var DefaultServeMux = NewServeMux()

// Does path match pattern?
func pathMatch(pattern, path string) bool {
	if len(pattern) == 0 {
		// should not happen
		return false
	}
	n := len(pattern)
	if pattern[n-1] != '/' {
		return pattern == path
	}
	return len(path) >= n && path[0:n] == pattern
}

// Return the canonical path for p, eliminating . and .. elements.
func cleanPath(p string) string {
	if p == "" {
		return "/"
	}
	if p[0] != '/' {
		p = "/" + p
	}
	np := path.Clean(p)
	// path.Clean removes trailing slash except for root;
	// put the trailing slash back if necessary.
	if p[len(p)-1] == '/' && np != "/" {
		np += "/"
	}
	return np
}

// Find a handler on a handler map given a path string
// Most-specific (longest) pattern wins
func (mux *ServeMux) match(path string) (h Handler, pattern string) {
	var n = 0
	for k, v := range mux.m {
		if !pathMatch(k, path) {
			continue
		}
		if h == nil || len(k) > n {
			n = len(k)
			h = v.h
			pattern = v.pattern
		}
	}
	return
}

// Handler returns the handler to use for the given request,
// consulting r.Method, r.Host, and r.URL.Path. It always returns
// a non-nil handler. If the path is not in its canonical form, the
// handler will be an internally-generated handler that redirects
// to the canonical path.
//
// Handler also returns the registered pattern that matches the
// request or, in the case of internally-generated redirects,
// the pattern that will match after following the redirect.
//
// If there is no registered handler that applies to the request,
// Handler returns a ``page not found'' handler and an empty pattern.
func (mux *ServeMux) Handler(r *Request) (h Handler, pattern string) {
	if r.Method != "CONNECT" {
		if p := cleanPath(r.URL.Path); p != r.URL.Path {
			_, pattern = mux.handler(r.Host, p)
			url := *r.URL
			url.Path = p
			return RedirectHandler(url.String(), StatusMovedPermanently), pattern
		}
	}

	return mux.handler(r.Host, r.URL.Path)
}

// handler is the main implementation of Handler.
// The path is known to be in canonical form, except for CONNECT methods.
func (mux *ServeMux) handler(host, path string) (h Handler, pattern string) {
	mux.mu.RLock()
	defer mux.mu.RUnlock()

	// Host-specific pattern takes precedence over generic ones
	if mux.hosts {
		h, pattern = mux.match(host + path)
	}
	if h == nil {
		h, pattern = mux.match(path)
	}
	if h == nil {
		h, pattern = NotFoundHandler(), ""
	}
	return
}

// ServeHTTP dispatches the request to the handler whose
// pattern most closely matches the request URL.
func (mux *ServeMux) ServeHTTP(w ResponseWriter, r *Request) {
	if r.RequestURI == "*" {
		if r.ProtoAtLeast(1, 1) {
			w.Header().Set("Connection", "close")
		}
		w.WriteHeader(StatusBadRequest)
		return
	}
	h, _ := mux.Handler(r)
	h.ServeHTTP(w, r)
}

// Handle registers the handler for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (mux *ServeMux) Handle(pattern string, handler Handler) {
	mux.mu.Lock()
	defer mux.mu.Unlock()

	if pattern == "" {
		panic("http: invalid pattern " + pattern)
	}
	if handler == nil {
		panic("http: nil handler")
	}
	if mux.m[pattern].explicit {
		panic("http: multiple registrations for " + pattern)
	}

	mux.m[pattern] = muxEntry{explicit: true, h: handler, pattern: pattern}

	if pattern[0] != '/' {
		mux.hosts = true
	}

	// Helpful behavior:
	// If pattern is /tree/, insert an implicit permanent redirect for /tree.
	// It can be overridden by an explicit registration.
	n := len(pattern)
	if n > 0 && pattern[n-1] == '/' && !mux.m[pattern[0:n-1]].explicit {
		// If pattern contains a host name, strip it and use remaining
		// path for redirect.
		path := pattern
		if pattern[0] != '/' {
			// In pattern, at least the last character is a '/', so
			// strings.Index can't be -1.
			path = pattern[strings.Index(pattern, "/"):]
		}
		url := &url.URL{Path: path}
		mux.m[pattern[0:n-1]] = muxEntry{h: RedirectHandler(url.String(), StatusMovedPermanently), pattern: pattern}
	}
}

// HandleFunc registers the handler function for the given pattern.
func (mux *ServeMux) HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	mux.Handle(pattern, HandlerFunc(handler))
}

// Handle registers the handler for the given pattern
// in the DefaultServeMux.
// The documentation for ServeMux explains how patterns are matched.
func Handle(pattern string, handler Handler) { DefaultServeMux.Handle(pattern, handler) }

// HandleFunc registers the handler function for the given pattern
// in the DefaultServeMux.
// The documentation for ServeMux explains how patterns are matched.
func HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	DefaultServeMux.HandleFunc(pattern, handler)
}

// Serve accepts incoming HTTP connections on the listener l,
// creating a new service goroutine for each.  The service goroutines
// read requests and then call handler to reply to them.
// Handler is typically nil, in which case the DefaultServeMux is used.
func Serve(l net.Listener, handler Handler) error {
	srv := &Server{Handler: handler}
	return srv.Serve(l)
}

// A Server defines parameters for running an HTTP server.
// The zero value for Server is a valid configuration.
type Server struct {
	Addr           string        // TCP address to listen on, ":http" if empty
	Handler        Handler       // handler to invoke, http.DefaultServeMux if nil
	ReadTimeout    time.Duration // maximum duration before timing out read of the request
	WriteTimeout   time.Duration // maximum duration before timing out write of the response
	MaxHeaderBytes int           // maximum size of request headers, DefaultMaxHeaderBytes if 0
	TLSConfig      *tls.Config   // optional TLS config, used by ListenAndServeTLS

	// TLSNextProto optionally specifies a function to take over
	// ownership of the provided TLS connection when an NPN
	// protocol upgrade has occurred.  The map key is the protocol
	// name negotiated. The Handler argument should be used to
	// handle HTTP requests and will initialize the Request's TLS
	// and RemoteAddr if not already set.  The connection is
	// automatically closed when the function returns.
	TLSNextProto map[string]func(*Server, *tls.Conn, Handler)

	// ConnState specifies an optional callback function that is
	// called when a client connection changes state. See the
	// ConnState type and associated constants for details.
	ConnState func(net.Conn, ConnState)

	// ErrorLog specifies an optional logger for errors accepting
	// connections and unexpected behavior from handlers.
	// If nil, logging goes to os.Stderr via the log package's
	// standard logger.
	ErrorLog *log.Logger

	disableKeepAlives int32 // accessed atomically.
}

// A ConnState represents the state of a client connection to a server.
// It's used by the optional Server.ConnState hook.
type ConnState int

const (
	// StateNew represents a new connection that is expected to
	// send a request immediately. Connections begin at this
	// state and then transition to either StateActive or
	// StateClosed.
	StateNew ConnState = iota

	// StateActive represents a connection that has read 1 or more
	// bytes of a request. The Server.ConnState hook for
	// StateActive fires before the request has entered a handler
	// and doesn't fire again until the request has been
	// handled. After the request is handled, the state
	// transitions to StateClosed, StateHijacked, or StateIdle.
	StateActive

	// StateIdle represents a connection that has finished
	// handling a request and is in the keep-alive state, waiting
	// for a new request. Connections transition from StateIdle
	// to either StateActive or StateClosed.
	StateIdle

	// StateHijacked represents a hijacked connection.
	// This is a terminal state. It does not transition to StateClosed.
	StateHijacked

	// StateClosed represents a closed connection.
	// This is a terminal state. Hijacked connections do not
	// transition to StateClosed.
	StateClosed
)

var stateName = map[ConnState]string{
	StateNew:      "new",
	StateActive:   "active",
	StateIdle:     "idle",
	StateHijacked: "hijacked",
	StateClosed:   "closed",
}

func (c ConnState) String() string {
	return stateName[c]
}

// serverHandler delegates to either the server's Handler or
// DefaultServeMux and also handles "OPTIONS *" requests.
type serverHandler struct {
	srv *Server
}

func (sh serverHandler) ServeHTTP(rw ResponseWriter, req *Request) {
	handler := sh.srv.Handler
	if handler == nil {
		handler = DefaultServeMux
	}
	if req.RequestURI == "*" && req.Method == "OPTIONS" {
		handler = globalOptionsHandler{}
	}
	handler.ServeHTTP(rw, req)
}

// ListenAndServe listens on the TCP network address srv.Addr and then
// calls Serve to handle requests on incoming connections.  If
// srv.Addr is blank, ":http" is used.
func (srv *Server) ListenAndServe() error {
	addr := srv.Addr
	if addr == "" {
		addr = ":http"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return srv.Serve(tcpKeepAliveListener{ln.(*net.TCPListener)})
}

// Serve accepts incoming connections on the Listener l, creating a
// new service goroutine for each.  The service goroutines read requests and
// then call srv.Handler to reply to them.
func (srv *Server) Serve(l net.Listener) error {
	defer l.Close()
	var tempDelay time.Duration // how long to sleep on accept failure
	for {
		rw, e := l.Accept()
		if e != nil {
			if ne, ok := e.(net.Error); ok && ne.Temporary() {
				if tempDelay == 0 {
					tempDelay = 5 * time.Millisecond
				} else {
					tempDelay *= 2
				}
				if max := 1 * time.Second; tempDelay > max {
					tempDelay = max
				}
				srv.logf("http: Accept error: %v; retrying in %v", e, tempDelay)
				time.Sleep(tempDelay)
				continue
			}
			return e
		}
		tempDelay = 0
		c, err := srv.newConn(rw)
		if err != nil {
			continue
		}
		c.setState(c.rwc, StateNew) // before Serve can return
		go c.serve()
	}
}

func (s *Server) doKeepAlives() bool {
	return atomic.LoadInt32(&s.disableKeepAlives) == 0
}

// SetKeepAlivesEnabled controls whether HTTP keep-alives are enabled.
// By default, keep-alives are always enabled. Only very
// resource-constrained environments or servers in the process of
// shutting down should disable them.
func (srv *Server) SetKeepAlivesEnabled(v bool) {
	if v {
		atomic.StoreInt32(&srv.disableKeepAlives, 0)
	} else {
		atomic.StoreInt32(&srv.disableKeepAlives, 1)
	}
}

func (s *Server) logf(format string, args ...interface{}) {
	if s.ErrorLog != nil {
		s.ErrorLog.Printf(format, args...)
	} else {
		log.Printf(format, args...)
	}
}

// ListenAndServe listens on the TCP network address addr
// and then calls Serve with handler to handle requests
// on incoming connections.  Handler is typically nil,
// in which case the DefaultServeMux is used.
//
// A trivial example server is:
//
//	package main
//
//	import (
//		"io"
//		"net/http"
//		"log"
//	)
//
//	// hello world, the web server
//	func HelloServer(w http.ResponseWriter, req *http.Request) {
//		io.WriteString(w, "hello, world!\n")
//	}
//
//	func main() {
//		http.HandleFunc("/hello", HelloServer)
//		err := http.ListenAndServe(":12345", nil)
//		if err != nil {
//			log.Fatal("ListenAndServe: ", err)
//		}
//	}
func ListenAndServe(addr string, handler Handler) error {
	server := &Server{Addr: addr, Handler: handler}
	return server.ListenAndServe()
}

// ListenAndServeTLS acts identically to ListenAndServe, except that it
// expects HTTPS connections. Additionally, files containing a certificate and
// matching private key for the server must be provided. If the certificate
// is signed by a certificate authority, the certFile should be the concatenation
// of the server's certificate, any intermediates, and the CA's certificate.
//
// A trivial example server is:
//
//	import (
//		"log"
//		"net/http"
//	)
//
//	func handler(w http.ResponseWriter, req *http.Request) {
//		w.Header().Set("Content-Type", "text/plain")
//		w.Write([]byte("This is an example server.\n"))
//	}
//
//	func main() {
//		http.HandleFunc("/", handler)
//		log.Printf("About to listen on 10443. Go to https://127.0.0.1:10443/")
//		err := http.ListenAndServeTLS(":10443", "cert.pem", "key.pem", nil)
//		if err != nil {
//			log.Fatal(err)
//		}
//	}
//
// One can use generate_cert.go in crypto/tls to generate cert.pem and key.pem.
func ListenAndServeTLS(addr string, certFile string, keyFile string, handler Handler) error {
	server := &Server{Addr: addr, Handler: handler}
	return server.ListenAndServeTLS(certFile, keyFile)
}

// ListenAndServeTLS listens on the TCP network address srv.Addr and
// then calls Serve to handle requests on incoming TLS connections.
//
// Filenames containing a certificate and matching private key for the
// server must be provided if the Server's TLSConfig.Certificates is
// not populated. If the certificate is signed by a certificate
// authority, the certFile should be the concatenation of the server's
// certificate, any intermediates, and the CA's certificate.
//
// If srv.Addr is blank, ":https" is used.
func (srv *Server) ListenAndServeTLS(certFile, keyFile string) error {
	addr := srv.Addr
	if addr == "" {
		addr = ":https"
	}
	config := cloneTLSConfig(srv.TLSConfig)
	if config.NextProtos == nil {
		config.NextProtos = []string{"http/1.1"}
	}

	if len(config.Certificates) == 0 || certFile != "" || keyFile != "" {
		var err error
		config.Certificates = make([]tls.Certificate, 1)
		config.Certificates[0], err = tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return err
		}
	}

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	tlsListener := tls.NewListener(tcpKeepAliveListener{ln.(*net.TCPListener)}, config)
	return srv.Serve(tlsListener)
}

// TimeoutHandler returns a Handler that runs h with the given time limit.
//
// The new Handler calls h.ServeHTTP to handle each request, but if a
// call runs for longer than its time limit, the handler responds with
// a 503 Service Unavailable error and the given message in its body.
// (If msg is empty, a suitable default message will be sent.)
// After such a timeout, writes by h to its ResponseWriter will return
// ErrHandlerTimeout.
func TimeoutHandler(h Handler, dt time.Duration, msg string) Handler {
	f := func() <-chan time.Time {
		return time.After(dt)
	}
	return &timeoutHandler{h, f, msg}
}

// ErrHandlerTimeout is returned on ResponseWriter Write calls
// in handlers which have timed out.
var ErrHandlerTimeout = errors.New("http: Handler timeout")

type timeoutHandler struct {
	handler Handler
	timeout func() <-chan time.Time // returns channel producing a timeout
	body    string
}

func (h *timeoutHandler) errorBody() string {
	if h.body != "" {
		return h.body
	}
	return "<html><head><title>Timeout</title></head><body><h1>Timeout</h1></body></html>"
}

func (h *timeoutHandler) ServeHTTP(w ResponseWriter, r *Request) {
	done := make(chan bool, 1)
	tw := &timeoutWriter{w: w}
	go func() {
		h.handler.ServeHTTP(tw, r)
		done <- true
	}()
	select {
	case <-done:
		return
	case <-h.timeout():
		tw.mu.Lock()
		defer tw.mu.Unlock()
		if !tw.wroteHeader {
			tw.w.WriteHeader(StatusServiceUnavailable)
			tw.w.Write([]byte(h.errorBody()))
		}
		tw.timedOut = true
	}
}

type timeoutWriter struct {
	w ResponseWriter

	mu          sync.Mutex
	timedOut    bool
	wroteHeader bool
}

func (tw *timeoutWriter) Header() Header {
	return tw.w.Header()
}

func (tw *timeoutWriter) Write(p []byte) (int, error) {
	tw.mu.Lock()
	defer tw.mu.Unlock()
	tw.wroteHeader = true // implicitly at least
	if tw.timedOut {
		return 0, ErrHandlerTimeout
	}
	return tw.w.Write(p)
}

func (tw *timeoutWriter) WriteHeader(code int) {
	tw.mu.Lock()
	defer tw.mu.Unlock()
	if tw.timedOut || tw.wroteHeader {
		return
	}
	tw.wroteHeader = true
	tw.w.WriteHeader(code)
}

// tcpKeepAliveListener sets TCP keep-alive timeouts on accepted
// connections. It's used by ListenAndServe and ListenAndServeTLS so
// dead TCP connections (e.g. closing laptop mid-download) eventually
// go away.
type tcpKeepAliveListener struct {
	*net.TCPListener
}

func (ln tcpKeepAliveListener) Accept() (c net.Conn, err error) {
	tc, err := ln.AcceptTCP()
	if err != nil {
		return
	}
	tc.SetKeepAlive(true)
	tc.SetKeepAlivePeriod(3 * time.Minute)
	return tc, nil
}

// globalOptionsHandler responds to "OPTIONS *" requests.
type globalOptionsHandler struct{}

func (globalOptionsHandler) ServeHTTP(w ResponseWriter, r *Request) {
	w.Header().Set("Content-Length", "0")
	if r.ContentLength != 0 {
		// Read up to 4KB of OPTIONS body (as mentioned in the
		// spec as being reserved for future use), but anything
		// over that is considered a waste of server resources
		// (or an attack) and we abort and close the connection,
		// courtesy of MaxBytesReader's EOF behavior.
		mb := MaxBytesReader(w, r.Body, 4<<10)
		io.Copy(ioutil.Discard, mb)
	}
}

type eofReaderWithWriteTo struct{}

func (eofReaderWithWriteTo) WriteTo(io.Writer) (int64, error) { return 0, nil }
func (eofReaderWithWriteTo) Read([]byte) (int, error)         { return 0, io.EOF }

// eofReader is a non-nil io.ReadCloser that always returns EOF.
// It has a WriteTo method so io.Copy won't need a buffer.
var eofReader = &struct {
	eofReaderWithWriteTo
	io.Closer
}{
	eofReaderWithWriteTo{},
	ioutil.NopCloser(nil),
}

// Verify that an io.Copy from an eofReader won't require a buffer.
var _ io.WriterTo = eofReader

// initNPNRequest is an HTTP handler that initializes certain
// uninitialized fields in its *Request. Such partially-initialized
// Requests come from NPN protocol handlers.
type initNPNRequest struct {
	c *tls.Conn
	h serverHandler
}

func (h initNPNRequest) ServeHTTP(rw ResponseWriter, req *Request) {
	if req.TLS == nil {
		req.TLS = &tls.ConnectionState{}
		*req.TLS = h.c.ConnectionState()
	}
	if req.Body == nil {
		req.Body = eofReader
	}
	if req.RemoteAddr == "" {
		req.RemoteAddr = h.c.RemoteAddr().String()
	}
	h.h.ServeHTTP(rw, req)
}

// loggingConn is used for debugging.
type loggingConn struct {
	name string
	net.Conn
}

var (
	uniqNameMu   sync.Mutex
	uniqNameNext = make(map[string]int)
)

func newLoggingConn(baseName string, c net.Conn) net.Conn {
	uniqNameMu.Lock()
	defer uniqNameMu.Unlock()
	uniqNameNext[baseName]++
	return &loggingConn{
		name: fmt.Sprintf("%s-%d", baseName, uniqNameNext[baseName]),
		Conn: c,
	}
}

func (c *loggingConn) Write(p []byte) (n int, err error) {
	log.Printf("%s.Write(%d) = ....", c.name, len(p))
	n, err = c.Conn.Write(p)
	log.Printf("%s.Write(%d) = %d, %v", c.name, len(p), n, err)
	return
}

func (c *loggingConn) Read(p []byte) (n int, err error) {
	log.Printf("%s.Read(%d) = ....", c.name, len(p))
	n, err = c.Conn.Read(p)
	log.Printf("%s.Read(%d) = %d, %v", c.name, len(p), n, err)
	return
}

func (c *loggingConn) Close() (err error) {
	log.Printf("%s.Close() = ...", c.name)
	err = c.Conn.Close()
	log.Printf("%s.Close() = %v", c.name, err)
	return
}

// checkConnErrorWriter writes to c.rwc and records any write errors to c.werr.
// It only contains one field (and a pointer field at that), so it
// fits in an interface value without an extra allocation.
type checkConnErrorWriter struct {
	c *conn
}

func (w checkConnErrorWriter) Write(p []byte) (n int, err error) {
	n, err = w.c.w.Write(p) // c.w == c.rwc, except after a hijack, when rwc is nil.
	if err != nil && w.c.werr == nil {
		w.c.werr = err
	}
	return
}

func numLeadingCRorLF(v []byte) (n int) {
	for _, b := range v {
		if b == '\r' || b == '\n' {
			n++
			continue
		}
		break
	}
	return

}
