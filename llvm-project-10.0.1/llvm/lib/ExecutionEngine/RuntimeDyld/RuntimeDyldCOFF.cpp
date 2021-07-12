//===-- RuntimeDyldCOFF.cpp - Run-time dynamic linker for MC-JIT -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of COFF support for the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#include "RuntimeDyldCOFF.h"
#include "Targets/RuntimeDyldCOFFAArch64.h"
#include "Targets/RuntimeDyldCOFFI386.h"
#include "Targets/RuntimeDyldCOFFThumb.h"
#include "Targets/RuntimeDyldCOFFX86_64.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::object;

#define DEBUG_TYPE "dyld"

namespace {

class LoadedCOFFObjectInfo final
    : public LoadedObjectInfoHelper<LoadedCOFFObjectInfo,
                                    RuntimeDyld::LoadedObjectInfo> {
public:
  LoadedCOFFObjectInfo(
      RuntimeDyldImpl &RTDyld,
      RuntimeDyld::LoadedObjectInfo::ObjSectionToIDMap ObjSecToIDMap)
      : LoadedObjectInfoHelper(RTDyld, std::move(ObjSecToIDMap)) {}

  OwningBinary<ObjectFile>
  getObjectForDebug(const ObjectFile &Obj) const override {
    return OwningBinary<ObjectFile>();
  }
};
}

namespace llvm {

std::unique_ptr<RuntimeDyldCOFF>
llvm::RuntimeDyldCOFF::create(Triple::ArchType Arch,
                              RuntimeDyld::MemoryManager &MemMgr,
                              JITSymbolResolver &Resolver) {
  switch (Arch) {
  default: llvm_unreachable("Unsupported target for RuntimeDyldCOFF.");
  case Triple::x86:
    return std::make_unique<RuntimeDyldCOFFI386>(MemMgr, Resolver);
  case Triple::thumb:
    return std::make_unique<RuntimeDyldCOFFThumb>(MemMgr, Resolver);
  case Triple::x86_64:
    return std::make_unique<RuntimeDyldCOFFX86_64>(MemMgr, Resolver);
  case Triple::aarch64:
    return std::make_unique<RuntimeDyldCOFFAArch64>(MemMgr, Resolver);
  }
}

std::unique_ptr<RuntimeDyld::LoadedObjectInfo>
RuntimeDyldCOFF::loadObject(const object::ObjectFile &O) {
  if (auto ObjSectionToIDOrErr = loadObjectImpl(O)) {
    return std::make_unique<LoadedCOFFObjectInfo>(*this, *ObjSectionToIDOrErr);
  } else {
    HasError = true;
    raw_string_ostream ErrStream(ErrorStr);
    logAllUnhandledErrors(ObjSectionToIDOrErr.takeError(), ErrStream);
    return nullptr;
  }
}

uint64_t RuntimeDyldCOFF::getSymbolOffset(const SymbolRef &Sym) {
  // The value in a relocatable COFF object is the offset.
  return Sym.getValue();
}

bool RuntimeDyldCOFF::isCompatibleFile(const object::ObjectFile &Obj) const {
  return Obj.isCOFF();
}

} // namespace llvm
