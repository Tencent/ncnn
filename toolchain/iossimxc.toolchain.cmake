# Standard settings
# set(UNIX True)
# set(Darwin True)
# set(IOS True)
set (CMAKE_SYSTEM_NAME Darwin)
set (CMAKE_SYSTEM_VERSION 1)
set (UNIX True)
set (APPLE True)
set (IOS True)

# suppress -rdynamic
# set(CMAKE_SYSTEM_NAME Generic)

set(CMAKE_C_COMPILER i386-apple-darwin11-clang)
set(CMAKE_CXX_COMPILER i386-apple-darwin11-clang++)

set(_CMAKE_TOOLCHAIN_PREFIX i386-apple-darwin11-)

set(CMAKE_IOS_SDK_ROOT "/home/nihui/osd/cctools-port/usage_examples/ios_toolchain/target-sim/SDK/")

# Set the sysroot default to the most recent SDK
set(CMAKE_OSX_SYSROOT ${CMAKE_IOS_SDK_ROOT} CACHE PATH "Sysroot used for iOS Simulator support")

# set the architecture for iOS
# set(IOS_ARCH i386)
# set(IOS_ARCH x86_64)
set(IOS_ARCH i386;x86_64)

set(CMAKE_OSX_ARCHITECTURES ${IOS_ARCH} CACHE string "Build architecture for iOS Simulator")

# Set the find root to the iOS developer roots and to user defined paths
set(CMAKE_FIND_ROOT_PATH ${CMAKE_IOS_DEVELOPER_ROOT} ${CMAKE_IOS_SDK_ROOT} ${CMAKE_PREFIX_PATH} CACHE string "iOS Simulator find search path root")

# searching for frameworks only
set(CMAKE_FIND_FRAMEWORK FIRST)

# set up the default search directories for frameworks
set(CMAKE_SYSTEM_FRAMEWORK_PATH
    ${CMAKE_IOS_SDK_ROOT}/System/Library/Frameworks
)
