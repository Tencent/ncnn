// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Memory allocator, based on tcmalloc.
// http://goog-perftools.sourceforge.net/doc/tcmalloc.html

// The main allocator works in runs of pages.
// Small allocation sizes (up to and including 32 kB) are
// rounded to one of about 100 size classes, each of which
// has its own free list of objects of exactly that size.
// Any free page of memory can be split into a set of objects
// of one size class, which are then managed using free list
// allocators.
//
// The allocator's data structures are:
//
//	FixAlloc: a free-list allocator for fixed-size objects,
//		used to manage storage used by the allocator.
//	MHeap: the malloc heap, managed at page (4096-byte) granularity.
//	MSpan: a run of pages managed by the MHeap.
//	MCentral: a shared free list for a given size class.
//	MCache: a per-thread (in Go, per-P) cache for small objects.
//	MStats: allocation statistics.
//
// Allocating a small object proceeds up a hierarchy of caches:
//
//	1. Round the size up to one of the small size classes
//	   and look in the corresponding MCache free list.
//	   If the list is not empty, allocate an object from it.
//	   This can all be done without acquiring a lock.
//
//	2. If the MCache free list is empty, replenish it by
//	   taking a bunch of objects from the MCentral free list.
//	   Moving a bunch amortizes the cost of acquiring the MCentral lock.
//
//	3. If the MCentral free list is empty, replenish it by
//	   allocating a run of pages from the MHeap and then
//	   chopping that memory into a objects of the given size.
//	   Allocating many objects amortizes the cost of locking
//	   the heap.
//
//	4. If the MHeap is empty or has no page runs large enough,
//	   allocate a new group of pages (at least 1MB) from the
//	   operating system.  Allocating a large run of pages
//	   amortizes the cost of talking to the operating system.
//
// Freeing a small object proceeds up the same hierarchy:
//
//	1. Look up the size class for the object and add it to
//	   the MCache free list.
//
//	2. If the MCache free list is too long or the MCache has
//	   too much memory, return some to the MCentral free lists.
//
//	3. If all the objects in a given span have returned to
//	   the MCentral list, return that span to the page heap.
//
//	4. If the heap has too much memory, return some to the
//	   operating system.
//
//	TODO(rsc): Step 4 is not implemented.
//
// Allocating and freeing a large object uses the page heap
// directly, bypassing the MCache and MCentral free lists.
//
// The small objects on the MCache and MCentral free lists
// may or may not be zeroed.  They are zeroed if and only if
// the second word of the object is zero.  A span in the
// page heap is zeroed unless s->needzero is set. When a span
// is allocated to break into small objects, it is zeroed if needed
// and s->needzero is set. There are two main benefits to delaying the
// zeroing this way:
//
//	1. stack frames allocated from the small object lists
//	   or the page heap can avoid zeroing altogether.
//	2. the cost of zeroing when reusing a small object is
//	   charged to the mutator, not the garbage collector.
//
// This C code was written with an eye toward translating to Go
// in the future.  Methods have the form Type_Method(Type *t, ...).

typedef struct MCentral	MCentral;
typedef struct MHeap	MHeap;
typedef struct MSpan	MSpan;
typedef struct MStats	MStats;
typedef struct MLink	MLink;
typedef struct MTypes	MTypes;
typedef struct GCStats	GCStats;

enum
{
	PageShift	= 13,
	PageSize	= 1<<PageShift,
	PageMask	= PageSize - 1,
};
typedef	uintptr	PageID;		// address >> PageShift

enum
{
	// Computed constant.  The definition of MaxSmallSize and the
	// algorithm in msize.c produce some number of different allocation
	// size classes.  NumSizeClasses is that number.  It's needed here
	// because there are static arrays of this length; when msize runs its
	// size choosing algorithm it double-checks that NumSizeClasses agrees.
	NumSizeClasses = 67,

	// Tunable constants.
	MaxSmallSize = 32<<10,

	// Tiny allocator parameters, see "Tiny allocator" comment in malloc.goc.
	TinySize = 16,
	TinySizeClass = 2,

	FixAllocChunk = 16<<10,		// Chunk size for FixAlloc
	MaxMHeapList = 1<<(20 - PageShift),	// Maximum page length for fixed-size list in MHeap.
	HeapAllocChunk = 1<<20,		// Chunk size for heap growth

	// Number of bits in page to span calculations (4k pages).
	// On Windows 64-bit we limit the arena to 32GB or 35 bits (see below for reason).
	// On other 64-bit platforms, we limit the arena to 128GB, or 37 bits.
	// On 32-bit, we don't bother limiting anything, so we use the full 32-bit address.
#if __SIZEOF_POINTER__ == 8
#ifdef GOOS_windows
	// Windows counts memory used by page table into committed memory
	// of the process, so we can't reserve too much memory.
	// See http://golang.org/issue/5402 and http://golang.org/issue/5236.
	MHeapMap_Bits = 35 - PageShift,
#else
	MHeapMap_Bits = 37 - PageShift,
#endif
#else
	MHeapMap_Bits = 32 - PageShift,
#endif

	// Max number of threads to run garbage collection.
	// 2, 3, and 4 are all plausible maximums depending
	// on the hardware details of the machine.  The garbage
	// collector scales well to 8 cpus.
	MaxGcproc = 8,
};

// Maximum memory allocation size, a hint for callers.
// This must be a #define instead of an enum because it
// is so large.
#if __SIZEOF_POINTER__ == 8
#define	MaxMem	(1ULL<<(MHeapMap_Bits+PageShift))	/* 128 GB or 32 GB */
#else
#define	MaxMem	((uintptr)-1)
#endif

// A generic linked list of blocks.  (Typically the block is bigger than sizeof(MLink).)
struct MLink
{
	MLink *next;
};

// SysAlloc obtains a large chunk of zeroed memory from the
// operating system, typically on the order of a hundred kilobytes
// or a megabyte.
// NOTE: SysAlloc returns OS-aligned memory, but the heap allocator
// may use larger alignment, so the caller must be careful to realign the
// memory obtained by SysAlloc.
//
// SysUnused notifies the operating system that the contents
// of the memory region are no longer needed and can be reused
// for other purposes.
// SysUsed notifies the operating system that the contents
// of the memory region are needed again.
//
// SysFree returns it unconditionally; this is only used if
// an out-of-memory error has been detected midway through
// an allocation.  It is okay if SysFree is a no-op.
//
// SysReserve reserves address space without allocating memory.
// If the pointer passed to it is non-nil, the caller wants the
// reservation there, but SysReserve can still choose another
// location if that one is unavailable.  On some systems and in some
// cases SysReserve will simply check that the address space is
// available and not actually reserve it.  If SysReserve returns
// non-nil, it sets *reserved to true if the address space is
// reserved, false if it has merely been checked.
// NOTE: SysReserve returns OS-aligned memory, but the heap allocator
// may use larger alignment, so the caller must be careful to realign the
// memory obtained by SysAlloc.
//
// SysMap maps previously reserved address space for use.
// The reserved argument is true if the address space was really
// reserved, not merely checked.
//
// SysFault marks a (already SysAlloc'd) region to fault
// if accessed.  Used only for debugging the runtime.

void*	runtime_SysAlloc(uintptr nbytes, uint64 *stat);
void	runtime_SysFree(void *v, uintptr nbytes, uint64 *stat);
void	runtime_SysUnused(void *v, uintptr nbytes);
void	runtime_SysUsed(void *v, uintptr nbytes);
void	runtime_SysMap(void *v, uintptr nbytes, bool reserved, uint64 *stat);
void*	runtime_SysReserve(void *v, uintptr nbytes, bool *reserved);
void	runtime_SysFault(void *v, uintptr nbytes);

// FixAlloc is a simple free-list allocator for fixed size objects.
// Malloc uses a FixAlloc wrapped around SysAlloc to manages its
// MCache and MSpan objects.
//
// Memory returned by FixAlloc_Alloc is not zeroed.
// The caller is responsible for locking around FixAlloc calls.
// Callers can keep state in the object but the first word is
// smashed by freeing and reallocating.
struct FixAlloc
{
	uintptr	size;
	void	(*first)(void *arg, byte *p);	// called first time p is returned
	void*	arg;
	MLink*	list;
	byte*	chunk;
	uint32	nchunk;
	uintptr	inuse;	// in-use bytes now
	uint64*	stat;
};

void	runtime_FixAlloc_Init(FixAlloc *f, uintptr size, void (*first)(void*, byte*), void *arg, uint64 *stat);
void*	runtime_FixAlloc_Alloc(FixAlloc *f);
void	runtime_FixAlloc_Free(FixAlloc *f, void *p);


// Statistics.
// Shared with Go: if you edit this structure, also edit type MemStats in mem.go.
struct MStats
{
	// General statistics.
	uint64	alloc;		// bytes allocated and still in use
	uint64	total_alloc;	// bytes allocated (even if freed)
	uint64	sys;		// bytes obtained from system (should be sum of xxx_sys below, no locking, approximate)
	uint64	nlookup;	// number of pointer lookups
	uint64	nmalloc;	// number of mallocs
	uint64	nfree;  // number of frees

	// Statistics about malloc heap.
	// protected by mheap.Lock
	uint64	heap_alloc;	// bytes allocated and still in use
	uint64	heap_sys;	// bytes obtained from system
	uint64	heap_idle;	// bytes in idle spans
	uint64	heap_inuse;	// bytes in non-idle spans
	uint64	heap_released;	// bytes released to the OS
	uint64	heap_objects;	// total number of allocated objects

	// Statistics about allocation of low-level fixed-size structures.
	// Protected by FixAlloc locks.
	uint64	stacks_inuse;	// bootstrap stacks
	uint64	stacks_sys;
	uint64	mspan_inuse;	// MSpan structures
	uint64	mspan_sys;
	uint64	mcache_inuse;	// MCache structures
	uint64	mcache_sys;
	uint64	buckhash_sys;	// profiling bucket hash table
	uint64	gc_sys;
	uint64	other_sys;

	// Statistics about garbage collector.
	// Protected by mheap or stopping the world during GC.
	uint64	next_gc;	// next GC (in heap_alloc time)
	uint64  last_gc;	// last GC (in absolute time)
	uint64	pause_total_ns;
	uint64	pause_ns[256];
	uint64	pause_end[256];
	uint32	numgc;
	float64	gc_cpu_fraction;
	bool	enablegc;
	bool	debuggc;

	// Statistics about allocation size classes.
	struct {
		uint32 size;
		uint64 nmalloc;
		uint64 nfree;
	} by_size[NumSizeClasses];
};

extern MStats mstats
  __asm__ (GOSYM_PREFIX "runtime.memStats");
void	runtime_updatememstats(GCStats *stats);

// Size classes.  Computed and initialized by InitSizes.
//
// SizeToClass(0 <= n <= MaxSmallSize) returns the size class,
//	1 <= sizeclass < NumSizeClasses, for n.
//	Size class 0 is reserved to mean "not small".
//
// class_to_size[i] = largest size in class i
// class_to_allocnpages[i] = number of pages to allocate when
//	making new objects in class i

int32	runtime_SizeToClass(int32);
uintptr	runtime_roundupsize(uintptr);
extern	int32	runtime_class_to_size[NumSizeClasses];
extern	int32	runtime_class_to_allocnpages[NumSizeClasses];
extern	int8	runtime_size_to_class8[1024/8 + 1];
extern	int8	runtime_size_to_class128[(MaxSmallSize-1024)/128 + 1];
extern	void	runtime_InitSizes(void);


typedef struct MCacheList MCacheList;
struct MCacheList
{
	MLink *list;
	uint32 nlist;
};

// Per-thread (in Go, per-P) cache for small objects.
// No locking needed because it is per-thread (per-P).
struct MCache
{
	// The following members are accessed on every malloc,
	// so they are grouped here for better caching.
	int32 next_sample;		// trigger heap sample after allocating this many bytes
	intptr local_cachealloc;	// bytes allocated (or freed) from cache since last lock of heap
	// Allocator cache for tiny objects w/o pointers.
	// See "Tiny allocator" comment in malloc.goc.
	byte*	tiny;
	uintptr	tinysize;
	// The rest is not accessed on every malloc.
	MSpan*	alloc[NumSizeClasses];	// spans to allocate from
	MCacheList free[NumSizeClasses];// lists of explicitly freed objects
	// Local allocator stats, flushed during GC.
	uintptr local_nlookup;		// number of pointer lookups
	uintptr local_largefree;	// bytes freed for large objects (>MaxSmallSize)
	uintptr local_nlargefree;	// number of frees for large objects (>MaxSmallSize)
	uintptr local_nsmallfree[NumSizeClasses];	// number of frees for small objects (<=MaxSmallSize)
};

MSpan*	runtime_MCache_Refill(MCache *c, int32 sizeclass);
void	runtime_MCache_Free(MCache *c, MLink *p, int32 sizeclass, uintptr size);
void	runtime_MCache_ReleaseAll(MCache *c);

// MTypes describes the types of blocks allocated within a span.
// The compression field describes the layout of the data.
//
// MTypes_Empty:
//     All blocks are free, or no type information is available for
//     allocated blocks.
//     The data field has no meaning.
// MTypes_Single:
//     The span contains just one block.
//     The data field holds the type information.
//     The sysalloc field has no meaning.
// MTypes_Words:
//     The span contains multiple blocks.
//     The data field points to an array of type [NumBlocks]uintptr,
//     and each element of the array holds the type of the corresponding
//     block.
// MTypes_Bytes:
//     The span contains at most seven different types of blocks.
//     The data field points to the following structure:
//         struct {
//             type  [8]uintptr       // type[0] is always 0
//             index [NumBlocks]byte
//         }
//     The type of the i-th block is: data.type[data.index[i]]
enum
{
	MTypes_Empty = 0,
	MTypes_Single = 1,
	MTypes_Words = 2,
	MTypes_Bytes = 3,
};
struct MTypes
{
	byte	compression;	// one of MTypes_*
	uintptr	data;
};

enum
{
	KindSpecialFinalizer = 1,
	KindSpecialProfile = 2,
	// Note: The finalizer special must be first because if we're freeing
	// an object, a finalizer special will cause the freeing operation
	// to abort, and we want to keep the other special records around
	// if that happens.
};

typedef struct Special Special;
struct Special
{
	Special*	next;	// linked list in span
	uint16		offset;	// span offset of object
	byte		kind;	// kind of Special
};

// The described object has a finalizer set for it.
typedef struct SpecialFinalizer SpecialFinalizer;
struct SpecialFinalizer
{
	Special		special;
	FuncVal*	fn;
	const FuncType*	ft;
	const PtrType*	ot;
};

// The described object is being heap profiled.
typedef struct Bucket Bucket; // from mprof.goc
typedef struct SpecialProfile SpecialProfile;
struct SpecialProfile
{
	Special	special;
	Bucket*	b;
};

// An MSpan is a run of pages.
enum
{
	MSpanInUse = 0,
	MSpanFree,
	MSpanListHead,
	MSpanDead,
};
struct MSpan
{
	MSpan	*next;		// in a span linked list
	MSpan	*prev;		// in a span linked list
	PageID	start;		// starting page number
	uintptr	npages;		// number of pages in span
	MLink	*freelist;	// list of free objects
	// sweep generation:
	// if sweepgen == h->sweepgen - 2, the span needs sweeping
	// if sweepgen == h->sweepgen - 1, the span is currently being swept
	// if sweepgen == h->sweepgen, the span is swept and ready to use
	// h->sweepgen is incremented by 2 after every GC
	uint32	sweepgen;
	uint16	ref;		// capacity - number of objects in freelist
	uint8	sizeclass;	// size class
	bool	incache;	// being used by an MCache
	uint8	state;		// MSpanInUse etc
	uint8	needzero;	// needs to be zeroed before allocation
	uintptr	elemsize;	// computed from sizeclass or from npages
	int64   unusedsince;	// First time spotted by GC in MSpanFree state
	uintptr npreleased;	// number of pages released to the OS
	byte	*limit;		// end of data in span
	MTypes	types;		// types of allocated objects in this span
	Lock	specialLock;	// guards specials list
	Special	*specials;	// linked list of special records sorted by offset.
	MLink	*freebuf;	// objects freed explicitly, not incorporated into freelist yet
};

void	runtime_MSpan_Init(MSpan *span, PageID start, uintptr npages);
void	runtime_MSpan_EnsureSwept(MSpan *span);
bool	runtime_MSpan_Sweep(MSpan *span);

// Every MSpan is in one doubly-linked list,
// either one of the MHeap's free lists or one of the
// MCentral's span lists.  We use empty MSpan structures as list heads.
void	runtime_MSpanList_Init(MSpan *list);
bool	runtime_MSpanList_IsEmpty(MSpan *list);
void	runtime_MSpanList_Insert(MSpan *list, MSpan *span);
void	runtime_MSpanList_InsertBack(MSpan *list, MSpan *span);
void	runtime_MSpanList_Remove(MSpan *span);	// from whatever list it is in


// Central list of free objects of a given size.
struct MCentral
{
	Lock  lock;
	int32 sizeclass;
	MSpan nonempty;	// list of spans with a free object
	MSpan empty;	// list of spans with no free objects (or cached in an MCache)
	int32 nfree;	// # of objects available in nonempty spans
};

void	runtime_MCentral_Init(MCentral *c, int32 sizeclass);
MSpan*	runtime_MCentral_CacheSpan(MCentral *c);
void	runtime_MCentral_UncacheSpan(MCentral *c, MSpan *s);
bool	runtime_MCentral_FreeSpan(MCentral *c, MSpan *s, int32 n, MLink *start, MLink *end);
void	runtime_MCentral_FreeList(MCentral *c, MLink *start); // TODO: need this?

// Main malloc heap.
// The heap itself is the "free[]" and "large" arrays,
// but all the other global data is here too.
struct MHeap
{
	Lock lock;
	MSpan free[MaxMHeapList];	// free lists of given length
	MSpan freelarge;		// free lists length >= MaxMHeapList
	MSpan busy[MaxMHeapList];	// busy lists of large objects of given length
	MSpan busylarge;		// busy lists of large objects length >= MaxMHeapList
	MSpan **allspans;		// all spans out there
	MSpan **sweepspans;		// copy of allspans referenced by sweeper
	uint32	nspan;
	uint32	nspancap;
	uint32	sweepgen;		// sweep generation, see comment in MSpan
	uint32	sweepdone;		// all spans are swept

	// span lookup
	MSpan**	spans;
	uintptr	spans_mapped;

	// range of addresses we might see in the heap
	byte *bitmap;
	uintptr bitmap_mapped;
	byte *arena_start;
	byte *arena_used;
	byte *arena_end;
	bool arena_reserved;

	// central free lists for small size classes.
	// the padding makes sure that the MCentrals are
	// spaced CacheLineSize bytes apart, so that each MCentral.Lock
	// gets its own cache line.
	struct {
		MCentral mcentral;
		byte pad[64];
	} central[NumSizeClasses];

	FixAlloc spanalloc;	// allocator for Span*
	FixAlloc cachealloc;	// allocator for MCache*
	FixAlloc specialfinalizeralloc;	// allocator for SpecialFinalizer*
	FixAlloc specialprofilealloc;	// allocator for SpecialProfile*
	Lock speciallock; // lock for sepcial record allocators.

	// Malloc stats.
	uint64 largefree;	// bytes freed for large objects (>MaxSmallSize)
	uint64 nlargefree;	// number of frees for large objects (>MaxSmallSize)
	uint64 nsmallfree[NumSizeClasses];	// number of frees for small objects (<=MaxSmallSize)
};
extern MHeap runtime_mheap;

void	runtime_MHeap_Init(MHeap *h);
MSpan*	runtime_MHeap_Alloc(MHeap *h, uintptr npage, int32 sizeclass, bool large, bool needzero);
void	runtime_MHeap_Free(MHeap *h, MSpan *s, int32 acct);
MSpan*	runtime_MHeap_Lookup(MHeap *h, void *v);
MSpan*	runtime_MHeap_LookupMaybe(MHeap *h, void *v);
void	runtime_MGetSizeClassInfo(int32 sizeclass, uintptr *size, int32 *npages, int32 *nobj);
void*	runtime_MHeap_SysAlloc(MHeap *h, uintptr n);
void	runtime_MHeap_MapBits(MHeap *h);
void	runtime_MHeap_MapSpans(MHeap *h);
void	runtime_MHeap_Scavenger(void*);
void	runtime_MHeap_SplitSpan(MHeap *h, MSpan *s);

void*	runtime_mallocgc(uintptr size, uintptr typ, uint32 flag);
void*	runtime_persistentalloc(uintptr size, uintptr align, uint64 *stat);
int32	runtime_mlookup(void *v, byte **base, uintptr *size, MSpan **s);
void	runtime_gc(int32 force);
uintptr	runtime_sweepone(void);
void	runtime_markscan(void *v);
void	runtime_marknogc(void *v);
void	runtime_checkallocated(void *v, uintptr n);
void	runtime_markfreed(void *v);
void	runtime_checkfreed(void *v, uintptr n);
extern	int32	runtime_checking;
void	runtime_markspan(void *v, uintptr size, uintptr n, bool leftover);
void	runtime_unmarkspan(void *v, uintptr size);
void	runtime_purgecachedstats(MCache*);
void*	runtime_cnew(const Type*);
void*	runtime_cnewarray(const Type*, intgo);
void	runtime_tracealloc(void*, uintptr, uintptr);
void	runtime_tracefree(void*, uintptr);
void	runtime_tracegc(void);

uintptr	runtime_gettype(void*);

enum
{
	// flags to malloc
	FlagNoScan	= 1<<0,	// GC doesn't have to scan object
	FlagNoProfiling	= 1<<1,	// must not profile
	FlagNoGC	= 1<<2,	// must not free or scan for pointers
	FlagNoZero	= 1<<3, // don't zero memory
	FlagNoInvokeGC	= 1<<4, // don't invoke GC
};

typedef struct Obj Obj;
struct Obj
{
	byte	*p;	// data pointer
	uintptr	n;	// size of data in bytes
	uintptr	ti;	// type info
};

void	runtime_MProf_Malloc(void*, uintptr);
void	runtime_MProf_Free(Bucket*, uintptr, bool);
void	runtime_MProf_GC(void);
void	runtime_iterate_memprof(void (*callback)(Bucket*, uintptr, Location*, uintptr, uintptr, uintptr));
int32	runtime_gcprocs(void);
void	runtime_helpgc(int32 nproc);
void	runtime_gchelper(void);
void	runtime_createfing(void);
G*	runtime_wakefing(void);
extern bool	runtime_fingwait;
extern bool	runtime_fingwake;

void	runtime_setprofilebucket(void *p, Bucket *b);

struct __go_func_type;
struct __go_ptr_type;
bool	runtime_addfinalizer(void *p, FuncVal *fn, const struct __go_func_type*, const struct __go_ptr_type*);
void	runtime_removefinalizer(void*);
void	runtime_queuefinalizer(void *p, FuncVal *fn, const struct __go_func_type *ft, const struct __go_ptr_type *ot);

void	runtime_freeallspecials(MSpan *span, void *p, uintptr size);
bool	runtime_freespecial(Special *s, void *p, uintptr size, bool freed);

enum
{
	TypeInfo_SingleObject = 0,
	TypeInfo_Array = 1,
	TypeInfo_Chan = 2,

	// Enables type information at the end of blocks allocated from heap	
	DebugTypeAtBlockEnd = 0,
};

// Information from the compiler about the layout of stack frames.
typedef struct BitVector BitVector;
struct BitVector
{
	int32 n; // # of bits
	uint32 *data;
};
typedef struct StackMap StackMap;
struct StackMap
{
	int32 n; // number of bitmaps
	int32 nbit; // number of bits in each bitmap
	uint32 data[];
};
enum {
	// Pointer map
	BitsPerPointer = 2,
	BitsDead = 0,
	BitsScalar = 1,
	BitsPointer = 2,
	BitsMultiWord = 3,
	// BitsMultiWord will be set for the first word of a multi-word item.
	// When it is set, one of the following will be set for the second word.
	BitsString = 0,
	BitsSlice = 1,
	BitsIface = 2,
	BitsEface = 3,
};
// Returns pointer map data for the given stackmap index
// (the index is encoded in PCDATA_StackMapIndex).
BitVector	runtime_stackmapdata(StackMap *stackmap, int32 n);

// defined in mgc0.go
void	runtime_gc_m_ptr(Eface*);
void	runtime_gc_g_ptr(Eface*);
void	runtime_gc_itab_ptr(Eface*);

void	runtime_memorydump(void);
int32	runtime_setgcpercent(int32);

// Value we use to mark dead pointers when GODEBUG=gcdead=1.
#define PoisonGC ((uintptr)0xf969696969696969ULL)
#define PoisonStack ((uintptr)0x6868686868686868ULL)

struct Workbuf;
void	runtime_MProf_Mark(struct Workbuf**, void (*)(struct Workbuf**, Obj));
void	runtime_proc_scan(struct Workbuf**, void (*)(struct Workbuf**, Obj));
void	runtime_time_scan(struct Workbuf**, void (*)(struct Workbuf**, Obj));
void	runtime_netpoll_scan(struct Workbuf**, void (*)(struct Workbuf**, Obj));
