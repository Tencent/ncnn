/* -----------------------------------------------------------------------
   ffiw64.c - Copyright (c) 2014 Red Hat, Inc.

   x86 win64 Foreign Function Interface

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   ``Software''), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED ``AS IS'', WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
   ----------------------------------------------------------------------- */

#include <ffi.h>
#include <ffi_common.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef X86_WIN64

struct win64_call_frame
{
  UINT64 rbp;		/* 0 */
  UINT64 retaddr;	/* 8 */
  UINT64 fn;		/* 16 */
  UINT64 flags;		/* 24 */
  UINT64 rvalue;	/* 32 */
};

extern void ffi_call_win64 (void *stack, struct win64_call_frame *,
			    void *closure) FFI_HIDDEN;

ffi_status
ffi_prep_cif_machdep (ffi_cif *cif)
{
  int flags, n;

  if (cif->abi != FFI_WIN64)
    return FFI_BAD_ABI;

  flags = cif->rtype->type;
  switch (flags)
    {
    default:
      break;
    case FFI_TYPE_LONGDOUBLE:
      flags = FFI_TYPE_STRUCT;
      break;
    case FFI_TYPE_COMPLEX:
      flags = FFI_TYPE_STRUCT;
      /* FALLTHRU */
    case FFI_TYPE_STRUCT:
      switch (cif->rtype->size)
	{
	case 8:
	  flags = FFI_TYPE_UINT64;
	  break;
	case 4:
	  flags = FFI_TYPE_SMALL_STRUCT_4B;
	  break;
	case 2:
	  flags = FFI_TYPE_SMALL_STRUCT_2B;
	  break;
	case 1:
	  flags = FFI_TYPE_SMALL_STRUCT_1B;
	  break;
	}
      break;
    }
  cif->flags = flags;

  /* Each argument either fits in a register, an 8 byte slot, or is
     passed by reference with the pointer in the 8 byte slot.  */
  n = cif->nargs;
  n += (flags == FFI_TYPE_STRUCT);
  if (n < 4)
    n = 4;
  cif->bytes = n * 8;

  return FFI_OK;
}

static void
ffi_call_int (ffi_cif *cif, void (*fn)(void), void *rvalue,
	      void **avalue, void *closure)
{
  int i, j, n, flags;
  UINT64 *stack;
  size_t rsize;
  struct win64_call_frame *frame;

  FFI_ASSERT(cif->abi == FFI_WIN64);

  flags = cif->flags;
  rsize = 0;

  /* If we have no return value for a structure, we need to create one.
     Otherwise we can ignore the return type entirely.  */
  if (rvalue == NULL)
    {
      if (flags == FFI_TYPE_STRUCT)
	rsize = cif->rtype->size;
      else
	flags = FFI_TYPE_VOID;
    }

  stack = alloca(cif->bytes + sizeof(struct win64_call_frame) + rsize);
  frame = (struct win64_call_frame *)((char *)stack + cif->bytes);
  if (rsize)
    rvalue = frame + 1;

  frame->fn = (uintptr_t)fn;
  frame->flags = flags;
  frame->rvalue = (uintptr_t)rvalue;

  j = 0;
  if (flags == FFI_TYPE_STRUCT)
    {
      stack[0] = (uintptr_t)rvalue;
      j = 1;
    }

  for (i = 0, n = cif->nargs; i < n; ++i, ++j)
    {
      switch (cif->arg_types[i]->size)
	{
	case 8:
	  stack[j] = *(UINT64 *)avalue[i];
	  break;
	case 4:
	  stack[j] = *(UINT32 *)avalue[i];
	  break;
	case 2:
	  stack[j] = *(UINT16 *)avalue[i];
	  break;
	case 1:
	  stack[j] = *(UINT8 *)avalue[i];
	  break;
	default:
	  stack[j] = (uintptr_t)avalue[i];
	  break;
	}
    }

  ffi_call_win64 (stack, frame, closure);
}

void
ffi_call (ffi_cif *cif, void (*fn)(void), void *rvalue, void **avalue)
{
  ffi_call_int (cif, fn, rvalue, avalue, NULL);
}

void
ffi_call_go (ffi_cif *cif, void (*fn)(void), void *rvalue,
	     void **avalue, void *closure)
{
  ffi_call_int (cif, fn, rvalue, avalue, closure);
}


extern void ffi_closure_win64(void) FFI_HIDDEN;
extern void ffi_go_closure_win64(void) FFI_HIDDEN;

ffi_status
ffi_prep_closure_loc (ffi_closure* closure,
		      ffi_cif* cif,
		      void (*fun)(ffi_cif*, void*, void**, void*),
		      void *user_data,
		      void *codeloc)
{
  static const unsigned char trampoline[16] = {
    /* leaq  -0x7(%rip),%r10   # 0x0  */
    0x4c, 0x8d, 0x15, 0xf9, 0xff, 0xff, 0xff,
    /* jmpq  *0x3(%rip)        # 0x10 */
    0xff, 0x25, 0x03, 0x00, 0x00, 0x00,
    /* nopl  (%rax) */
    0x0f, 0x1f, 0x00
  };
  unsigned char *tramp = closure->tramp;

  if (cif->abi != FFI_WIN64)
    return FFI_BAD_ABI;

  memcpy (tramp, trampoline, sizeof(trampoline));
  *(UINT64 *)(tramp + 16) = (uintptr_t)ffi_closure_win64;

  closure->cif = cif;
  closure->fun = fun;
  closure->user_data = user_data;

  return FFI_OK;
}

ffi_status
ffi_prep_go_closure (ffi_go_closure* closure, ffi_cif* cif,
		     void (*fun)(ffi_cif*, void*, void**, void*))
{
  if (cif->abi != FFI_WIN64)
    return FFI_BAD_ABI;

  closure->tramp = ffi_go_closure_win64;
  closure->cif = cif;
  closure->fun = fun;

  return FFI_OK;
}

struct win64_closure_frame
{
  UINT64 rvalue[2];
  UINT64 fargs[4];
  UINT64 retaddr;
  UINT64 args[];
};

int FFI_HIDDEN
ffi_closure_win64_inner(ffi_cif *cif,
			void (*fun)(ffi_cif*, void*, void**, void*),
			void *user_data,
			struct win64_closure_frame *frame)
{
  void **avalue;
  void *rvalue;
  int i, n, nreg, flags;

  avalue = alloca(cif->nargs * sizeof(void *));
  rvalue = frame->rvalue;
  nreg = 0;

  /* When returning a structure, the address is in the first argument.
     We must also be prepared to return the same address in eax, so
     install that address in the frame and pretend we return a pointer.  */
  flags = cif->flags;
  if (flags == FFI_TYPE_STRUCT)
    {
      rvalue = (void *)(uintptr_t)frame->args[0];
      frame->rvalue[0] = frame->args[0];
      nreg = 1;
    }

  for (i = 0, n = cif->nargs; i < n; ++i, ++nreg)
    {
      size_t size = cif->arg_types[i]->size;
      size_t type = cif->arg_types[i]->type;
      void *a;

      if (type == FFI_TYPE_DOUBLE || type == FFI_TYPE_FLOAT)
	{
	  if (nreg < 4)
	    a = &frame->fargs[nreg];
	  else
	    a = &frame->args[nreg];
	}
      else if (size == 1 || size == 2 || size == 4 || size == 8)
	a = &frame->args[nreg];
      else
	a = (void *)(uintptr_t)frame->args[nreg];

      avalue[i] = a;
    }

  /* Invoke the closure.  */
  fun (cif, rvalue, avalue, user_data);
  return flags;
}

#endif /* X86_WIN64 */
