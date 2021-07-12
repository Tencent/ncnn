//===-- NativeRegisterContextLinux_mips64.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__mips__)

#ifndef lldb_NativeRegisterContextLinux_mips64_h
#define lldb_NativeRegisterContextLinux_mips64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/RegisterContext_mips.h"
#include "Plugins/Process/Utility/lldb-mips-linux-register-enums.h"
#include <sys/uio.h>

#define MAX_NUM_WP 8

namespace lldb_private {
namespace process_linux {

class NativeProcessLinux;

class NativeRegisterContextLinux_mips64 : public NativeRegisterContextLinux {
public:
  NativeRegisterContextLinux_mips64(const ArchSpec &target_arch,
                                    NativeThreadProtocol &native_thread);

  uint32_t GetRegisterSetCount() const override;

  lldb::addr_t GetPCfromBreakpointLocation(
      lldb::addr_t fail_value = LLDB_INVALID_ADDRESS) override;

  lldb::addr_t GetWatchpointHitAddress(uint32_t wp_index) override;

  const RegisterSet *GetRegisterSet(uint32_t set_index) const override;

  Status ReadRegister(const RegisterInfo *reg_info,
                      RegisterValue &reg_value) override;

  Status WriteRegister(const RegisterInfo *reg_info,
                       const RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::DataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  Status ReadCP1();

  Status WriteCP1();

  uint8_t *ReturnFPOffset(uint8_t reg_index, uint32_t byte_offset);

  Status IsWatchpointHit(uint32_t wp_index, bool &is_hit) override;

  Status GetWatchpointHitIndex(uint32_t &wp_index,
                               lldb::addr_t trap_addr) override;

  Status IsWatchpointVacant(uint32_t wp_index, bool &is_vacant) override;

  bool ClearHardwareWatchpoint(uint32_t wp_index) override;

  Status ClearAllHardwareWatchpoints() override;

  Status SetHardwareWatchpointWithIndex(lldb::addr_t addr, size_t size,
                                        uint32_t watch_flags,
                                        uint32_t wp_index);

  uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                 uint32_t watch_flags) override;

  lldb::addr_t GetWatchpointAddress(uint32_t wp_index) override;

  uint32_t NumSupportedHardwareWatchpoints() override;

  static bool IsMSAAvailable();

protected:
  Status Read_SR_Config(uint32_t offset, const char *reg_name, uint32_t size,
                        RegisterValue &value);

  Status ReadRegisterRaw(uint32_t reg_index, RegisterValue &value) override;

  Status WriteRegisterRaw(uint32_t reg_index,
                          const RegisterValue &value) override;

  Status DoReadWatchPointRegisterValue(lldb::tid_t tid, void *watch_readback);

  Status DoWriteWatchPointRegisterValue(lldb::tid_t tid, void *watch_readback);

  bool IsFR0();

  bool IsFRE();

  bool IsFPR(uint32_t reg_index) const;

  bool IsMSA(uint32_t reg_index) const;

  void *GetGPRBuffer() override { return &m_gpr; }

  void *GetFPRBuffer() override { return &m_fpr; }

  size_t GetFPRSize() override { return sizeof(FPR_linux_mips); }

private:
  // Info about register ranges.
  struct RegInfo {
    uint32_t num_registers;
    uint32_t num_gpr_registers;
    uint32_t num_fpr_registers;

    uint32_t last_gpr;
    uint32_t first_fpr;
    uint32_t last_fpr;
    uint32_t first_msa;
    uint32_t last_msa;
  };

  RegInfo m_reg_info;

  GPR_linux_mips m_gpr;

  FPR_linux_mips m_fpr;

  MSA_linux_mips m_msa;

  lldb::addr_t hw_addr_map[MAX_NUM_WP];

  struct iovec m_iovec;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_mips64_h

#endif // defined (__mips__)
