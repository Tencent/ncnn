//===-- SymbolFileDWARFDwo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
#define SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_

#include "SymbolFileDWARF.h"

class SymbolFileDWARFDwo : public SymbolFileDWARF {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SymbolFileDWARF::isA(ClassID);
  }
  static bool classof(const SymbolFile *obj) { return obj->isA(&ID); }
  /// \}

  SymbolFileDWARFDwo(lldb::ObjectFileSP objfile, DWARFCompileUnit &dwarf_cu);

  ~SymbolFileDWARFDwo() override = default;

  lldb::CompUnitSP ParseCompileUnit(DWARFCompileUnit &dwarf_cu) override;

  DWARFCompileUnit *GetCompileUnit();

  DWARFUnit *
  GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit) override;

  size_t GetObjCMethodDIEOffsets(lldb_private::ConstString class_name,
                                 DIEArray &method_die_offsets) override;

  llvm::Expected<lldb_private::TypeSystem &>
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  DWARFDIE
  GetDIE(const DIERef &die_ref) override;

  DWARFCompileUnit *GetBaseCompileUnit() override { return &m_base_dwarf_cu; }

  llvm::Optional<uint32_t> GetDwoNum() override { return GetID() >> 32; }

protected:
  void LoadSectionData(lldb::SectionType sect_type,
                       lldb_private::DWARFDataExtractor &data) override;

  DIEToTypePtr &GetDIEToType() override;

  DIEToVariableSP &GetDIEToVariable() override;

  DIEToClangType &GetForwardDeclDieToClangType() override;

  ClangTypeToDIE &GetForwardDeclClangTypeToDie() override;

  UniqueDWARFASTTypeMap &GetUniqueDWARFASTTypeMap() override;

  lldb::TypeSP FindDefinitionTypeForDWARFDeclContext(
      const DWARFDeclContext &die_decl_ctx) override;

  lldb::TypeSP FindCompleteObjCDefinitionTypeForDIE(
      const DWARFDIE &die, lldb_private::ConstString type_name,
      bool must_be_implementation) override;

  SymbolFileDWARF &GetBaseSymbolFile();

  DWARFCompileUnit *ComputeCompileUnit();

  DWARFCompileUnit &m_base_dwarf_cu;
  DWARFCompileUnit *m_cu = nullptr;
};

#endif // SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
