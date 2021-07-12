..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid10_offset_smem_plain:

soffset
===========================

An offset added to the base address to get memory address.

* If offset is specified as a register, it supplies an unsigned byte offset.
* If offset is specified as a 21-bit immediate, it supplies a signed byte offset.

.. WARNING:: Assembler currently supports 20-bit unsigned offsets only. Use :ref:`uimm20<amdgpu_synid_uimm20>` instead of :ref:`simm21<amdgpu_synid_simm21>`.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`simm21<amdgpu_synid_simm21>`
