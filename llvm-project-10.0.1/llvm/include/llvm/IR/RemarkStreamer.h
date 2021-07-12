//===- llvm/IR/RemarkStreamer.h - Remark Streamer ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the main interface for outputting remarks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_REMARKSTREAMER_H
#define LLVM_IR_REMARKSTREAMER_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace llvm {
/// Streamer for remarks.
class RemarkStreamer {
  /// The regex used to filter remarks based on the passes that emit them.
  Optional<Regex> PassFilter;
  /// The object used to serialize the remarks to a specific format.
  std::unique_ptr<remarks::RemarkSerializer> RemarkSerializer;
  /// The filename that the remark diagnostics are emitted to.
  const Optional<std::string> Filename;

  /// Convert diagnostics into remark objects.
  /// The lifetime of the members of the result is bound to the lifetime of
  /// the LLVM diagnostics.
  remarks::Remark toRemark(const DiagnosticInfoOptimizationBase &Diag);

public:
  RemarkStreamer(std::unique_ptr<remarks::RemarkSerializer> RemarkSerializer,
                 Optional<StringRef> Filename = None);
  /// Return the filename that the remark diagnostics are emitted to.
  Optional<StringRef> getFilename() const {
    return Filename ? Optional<StringRef>(*Filename) : None;
  }
  /// Return stream that the remark diagnostics are emitted to.
  raw_ostream &getStream() { return RemarkSerializer->OS; }
  /// Return the serializer used for this stream.
  remarks::RemarkSerializer &getSerializer() { return *RemarkSerializer; }
  /// Set a pass filter based on a regex \p Filter.
  /// Returns an error if the regex is invalid.
  Error setFilter(StringRef Filter);
  /// Emit a diagnostic through the streamer.
  void emit(const DiagnosticInfoOptimizationBase &Diag);
  /// Check if the remarks also need to have associated metadata in a section.
  bool needsSection() const;
};

template <typename ThisError>
struct RemarkSetupErrorInfo : public ErrorInfo<ThisError> {
  std::string Msg;
  std::error_code EC;

  RemarkSetupErrorInfo(Error E) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &EIB) {
      Msg = EIB.message();
      EC = EIB.convertToErrorCode();
    });
  }

  void log(raw_ostream &OS) const override { OS << Msg; }
  std::error_code convertToErrorCode() const override { return EC; }
};

struct RemarkSetupFileError : RemarkSetupErrorInfo<RemarkSetupFileError> {
  static char ID;
  using RemarkSetupErrorInfo<RemarkSetupFileError>::RemarkSetupErrorInfo;
};

struct RemarkSetupPatternError : RemarkSetupErrorInfo<RemarkSetupPatternError> {
  static char ID;
  using RemarkSetupErrorInfo<RemarkSetupPatternError>::RemarkSetupErrorInfo;
};

struct RemarkSetupFormatError : RemarkSetupErrorInfo<RemarkSetupFormatError> {
  static char ID;
  using RemarkSetupErrorInfo<RemarkSetupFormatError>::RemarkSetupErrorInfo;
};

/// Setup optimization remarks that output to a file.
Expected<std::unique_ptr<ToolOutputFile>>
setupOptimizationRemarks(LLVMContext &Context, StringRef RemarksFilename,
                         StringRef RemarksPasses, StringRef RemarksFormat,
                         bool RemarksWithHotness,
                         unsigned RemarksHotnessThreshold = 0);

/// Setup optimization remarks that output directly to a raw_ostream.
/// \p OS is managed by the caller and should be open for writing as long as \p
/// Context is streaming remarks to it.
Error setupOptimizationRemarks(LLVMContext &Context, raw_ostream &OS,
                               StringRef RemarksPasses, StringRef RemarksFormat,
                               bool RemarksWithHotness,
                               unsigned RemarksHotnessThreshold = 0);

} // end namespace llvm

#endif // LLVM_IR_REMARKSTREAMER_H
