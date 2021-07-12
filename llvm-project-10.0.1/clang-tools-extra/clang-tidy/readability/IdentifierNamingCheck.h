//===--- IdentifierNamingCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {

class MacroInfo;

namespace tidy {
namespace readability {

/// Checks for identifiers naming style mismatch.
///
/// This check will try to enforce coding guidelines on the identifiers naming.
/// It supports `lower_case`, `UPPER_CASE`, `camelBack` and `CamelCase` casing
/// and tries to convert from one to another if a mismatch is detected.
///
/// It also supports a fixed prefix and suffix that will be prepended or
/// appended to the identifiers, regardless of the casing.
///
/// Many configuration options are available, in order to be able to create
/// different rules for different kind of identifier. In general, the
/// rules are falling back to a more generic rule if the specific case is not
/// configured.
class IdentifierNamingCheck : public ClangTidyCheck {
public:
  IdentifierNamingCheck(StringRef Name, ClangTidyContext *Context);
  ~IdentifierNamingCheck();

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void onEndOfTranslationUnit() override;

  enum CaseType {
    CT_AnyCase = 0,
    CT_LowerCase,
    CT_CamelBack,
    CT_UpperCase,
    CT_CamelCase,
    CT_CamelSnakeCase,
    CT_CamelSnakeBack
  };

  struct NamingStyle {
    NamingStyle() = default;

    NamingStyle(llvm::Optional<CaseType> Case, const std::string &Prefix,
                const std::string &Suffix)
        : Case(Case), Prefix(Prefix), Suffix(Suffix) {}

    llvm::Optional<CaseType> Case;
    std::string Prefix;
    std::string Suffix;
  };

  /// This enum will be used in %select of the diagnostic message.
  /// Each value below IgnoreFailureThreshold should have an error message.
  enum class ShouldFixStatus {
    ShouldFix,
    ConflictsWithKeyword, /// The fixup will conflict with a language keyword,
                          /// so we can't fix it automatically.
    ConflictsWithMacroDefinition, /// The fixup will conflict with a macro
                                  /// definition, so we can't fix it
                                  /// automatically.

    /// Values pass this threshold will be ignored completely
    /// i.e no message, no fixup.
    IgnoreFailureThreshold,

    InsideMacro, /// If the identifier was used or declared within a macro we
                 /// won't offer a fixup for safety reasons.
  };

  /// Holds an identifier name check failure, tracking the kind of the
  /// identifier, its possible fixup and the starting locations of all the
  /// identifier usages.
  struct NamingCheckFailure {
    std::string KindName;
    std::string Fixup;

    /// Whether the failure should be fixed or not.
    ///
    /// ie: if the identifier was used or declared within a macro we won't offer
    /// a fixup for safety reasons.
    bool ShouldFix() const { return FixStatus == ShouldFixStatus::ShouldFix; }

    bool ShouldNotify() const {
      return FixStatus < ShouldFixStatus::IgnoreFailureThreshold;
    }

    ShouldFixStatus FixStatus = ShouldFixStatus::ShouldFix;

    /// A set of all the identifier usages starting SourceLocation, in
    /// their encoded form.
    llvm::DenseSet<unsigned> RawUsageLocs;

    NamingCheckFailure() = default;
  };

  typedef std::pair<SourceLocation, std::string> NamingCheckId;

  typedef llvm::DenseMap<NamingCheckId, NamingCheckFailure>
      NamingCheckFailureMap;

  /// Check Macros for style violations.
  void checkMacro(SourceManager &sourceMgr, const Token &MacroNameTok,
                  const MacroInfo *MI);

  /// Add a usage of a macro if it already has a violation.
  void expandMacro(const Token &MacroNameTok, const MacroInfo *MI);

private:
  std::vector<llvm::Optional<NamingStyle>> NamingStyles;
  bool IgnoreFailedSplit;
  NamingCheckFailureMap NamingCheckFailures;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERNAMINGCHECK_H
