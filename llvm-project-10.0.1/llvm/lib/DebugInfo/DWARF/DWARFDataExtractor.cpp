//===- DWARFDataExtractor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"

using namespace llvm;

uint64_t DWARFDataExtractor::getRelocatedValue(uint32_t Size, uint64_t *Off,
                                               uint64_t *SecNdx,
                                               Error *Err) const {
  if (SecNdx)
    *SecNdx = object::SectionedAddress::UndefSection;
  if (!Section)
    return getUnsigned(Off, Size, Err);
  Optional<RelocAddrEntry> E = Obj->find(*Section, *Off);
  uint64_t A = getUnsigned(Off, Size, Err);
  if (!E)
    return A;
  if (SecNdx)
    *SecNdx = E->SectionIndex;
  uint64_t R = E->Resolver(E->Reloc, E->SymbolValue, A);
  if (E->Reloc2)
    R = E->Resolver(*E->Reloc2, E->SymbolValue2, R);
  return R;
}

Optional<uint64_t>
DWARFDataExtractor::getEncodedPointer(uint64_t *Offset, uint8_t Encoding,
                                      uint64_t PCRelOffset) const {
  if (Encoding == dwarf::DW_EH_PE_omit)
    return None;

  uint64_t Result = 0;
  uint64_t OldOffset = *Offset;
  // First get value
  switch (Encoding & 0x0F) {
  case dwarf::DW_EH_PE_absptr:
    switch (getAddressSize()) {
    case 2:
    case 4:
    case 8:
      Result = getUnsigned(Offset, getAddressSize());
      break;
    default:
      return None;
    }
    break;
  case dwarf::DW_EH_PE_uleb128:
    Result = getULEB128(Offset);
    break;
  case dwarf::DW_EH_PE_sleb128:
    Result = getSLEB128(Offset);
    break;
  case dwarf::DW_EH_PE_udata2:
    Result = getUnsigned(Offset, 2);
    break;
  case dwarf::DW_EH_PE_udata4:
    Result = getUnsigned(Offset, 4);
    break;
  case dwarf::DW_EH_PE_udata8:
    Result = getUnsigned(Offset, 8);
    break;
  case dwarf::DW_EH_PE_sdata2:
    Result = getSigned(Offset, 2);
    break;
  case dwarf::DW_EH_PE_sdata4:
    Result = getSigned(Offset, 4);
    break;
  case dwarf::DW_EH_PE_sdata8:
    Result = getSigned(Offset, 8);
    break;
  default:
    return None;
  }
  // Then add relative offset, if required
  switch (Encoding & 0x70) {
  case dwarf::DW_EH_PE_absptr:
    // do nothing
    break;
  case dwarf::DW_EH_PE_pcrel:
    Result += PCRelOffset;
    break;
  case dwarf::DW_EH_PE_datarel:
  case dwarf::DW_EH_PE_textrel:
  case dwarf::DW_EH_PE_funcrel:
  case dwarf::DW_EH_PE_aligned:
  default:
    *Offset = OldOffset;
    return None;
  }

  return Result;
}
