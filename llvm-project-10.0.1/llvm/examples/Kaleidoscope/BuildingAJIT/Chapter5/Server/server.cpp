#include "../RemoteJITUtils.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/OrcRemoteTargetServer.h"
#include "llvm/ExecutionEngine/Orc/OrcABISupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <netinet/in.h>
#include <sys/socket.h>

using namespace llvm;
using namespace llvm::orc;

// Command line argument for TCP port.
cl::opt<uint32_t> Port("port",
                       cl::desc("TCP port to listen on"),
                       cl::init(20000));

ExitOnError ExitOnErr;

using MainFun = int (*)(int, const char*[]);

template <typename NativePtrT>
NativePtrT MakeNative(uint64_t P) {
  return reinterpret_cast<NativePtrT>(static_cast<uintptr_t>(P));
}

extern "C"
void printExprResult(double Val) {
  printf("Expression evaluated to: %f\n", Val);
}

// --- LAZY COMPILE TEST ---
int main(int argc, char* argv[]) {
  if (argc == 0)
    ExitOnErr.setBanner("jit_server: ");
  else
    ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  // --- Initialize LLVM ---
  cl::ParseCommandLineOptions(argc, argv, "LLVM lazy JIT example.\n");

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  if (sys::DynamicLibrary::LoadLibraryPermanently(nullptr)) {
    errs() << "Error loading program symbols.\n";
    return 1;
  }

  // --- Initialize remote connection ---

  int sockfd = socket(PF_INET, SOCK_STREAM, 0);
  sockaddr_in servAddr, clientAddr;
  socklen_t clientAddrLen = sizeof(clientAddr);
  memset(&servAddr, 0, sizeof(servAddr));
  servAddr.sin_family = PF_INET;
  servAddr.sin_family = INADDR_ANY;
  servAddr.sin_port = htons(Port);

  {
    // avoid "Address already in use" error.
    int yes = 1;
    if (setsockopt(sockfd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(int)) == -1) {
      errs() << "Error calling setsockopt.\n";
      return 1;
    }
  }

  if (bind(sockfd, reinterpret_cast<sockaddr*>(&servAddr),
           sizeof(servAddr)) < 0) {
    errs() << "Error on binding.\n";
    return 1;
  }
  listen(sockfd, 1);
  int newsockfd = accept(sockfd, reinterpret_cast<sockaddr*>(&clientAddr),
                         &clientAddrLen);

  auto SymbolLookup =
    [](const std::string &Name) {
      return RTDyldMemoryManager::getSymbolAddressInProcess(Name);
    };

  auto RegisterEHFrames =
    [](uint8_t *Addr, uint32_t Size) {
      RTDyldMemoryManager::registerEHFramesInProcess(Addr, Size);
    };

  auto DeregisterEHFrames =
    [](uint8_t *Addr, uint32_t Size) {
      RTDyldMemoryManager::deregisterEHFramesInProcess(Addr, Size);
    };

  FDRPCChannel TCPChannel(newsockfd, newsockfd);

  using MyServerT = remote::OrcRemoteTargetServer<FDRPCChannel, OrcX86_64_SysV>;

  MyServerT Server(TCPChannel, SymbolLookup, RegisterEHFrames, DeregisterEHFrames);

  while (!Server.receivedTerminate())
    ExitOnErr(Server.handleOne());

  return 0;
}
