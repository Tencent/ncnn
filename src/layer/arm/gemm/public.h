.macro ASM_DELARE function_name
#ifdef __APPLE__
.globl _\function_name
_\function_name:
#else
.global \function_name
#ifdef __ELF__
.hidden \function_name
.type \function_name, %function
#endif
\function_name:
.endm
