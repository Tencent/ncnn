//===-- DomainSocket.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DomainSocket_h_
#define liblldb_DomainSocket_h_

#include "lldb/Host/Socket.h"

namespace lldb_private {
class DomainSocket : public Socket {
public:
  DomainSocket(bool should_close, bool child_processes_inherit);

  Status Connect(llvm::StringRef name) override;
  Status Listen(llvm::StringRef name, int backlog) override;
  Status Accept(Socket *&socket) override;

  std::string GetRemoteConnectionURI() const override;

protected:
  DomainSocket(SocketProtocol protocol, bool child_processes_inherit);

  virtual size_t GetNameOffset() const;
  virtual void DeleteSocketFile(llvm::StringRef name);
  std::string GetSocketName() const;

private:
  DomainSocket(NativeSocket socket, const DomainSocket &listen_socket);
};
}

#endif // ifndef liblldb_DomainSocket_h_
