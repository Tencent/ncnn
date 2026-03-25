# FindVulkanHeaders.cmake
# Find Vulkan headers for improved cmake detection
#
# This module helps locate Vulkan SDK headers when building with Vulkan support
# Sets the following variables:
#   VULKAN_HEADERS_FOUND - True if Vulkan headers are found
#   VULKAN_HEADERS_INCLUDE_DIRS - Include directories for Vulkan headers
#   VULKAN_HEADERS_VERSION - Version of Vulkan headers if available
#
# Usage:
#   find_package(VulkanHeaders)
#   if(VULKAN_HEADERS_FOUND)
#       target_include_directories(mytarget PRIVATE ${VULKAN_HEADERS_INCLUDE_DIRS})
#   endif()

# Check for Vulkan SDK environment variable
if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
    if(EXISTS "${VULKAN_SDK_PATH}/include/vulkan/vulkan.h")
        set(VULKAN_HEADERS_INCLUDE_DIRS "${VULKAN_SDK_PATH}/include")
        set(VULKAN_HEADERS_FOUND TRUE)
        message(STATUS "Found Vulkan headers via VULKAN_SDK: ${VULKAN_HEADERS_INCLUDE_DIRS}")
    endif()
endif()

# Check common system paths if not found via SDK
if(NOT VULKAN_HEADERS_FOUND)
    find_path(VULKAN_HEADERS_INCLUDE_DIR
        NAMES vulkan/vulkan.h
        PATHS
            /usr/include
            /usr/local/include
            /opt/include
            ${CMAKE_PREFIX_PATH}/include
        DOC "Vulkan headers include directory"
    )

    if(VULKAN_HEADERS_INCLUDE_DIR)
        set(VULKAN_HEADERS_INCLUDE_DIRS ${VULKAN_HEADERS_INCLUDE_DIR})
        set(VULKAN_HEADERS_FOUND TRUE)
        message(STATUS "Found Vulkan headers: ${VULKAN_HEADERS_INCLUDE_DIRS}")
    endif()
endif()

# Try to extract version from vulkan_core.h if found
if(VULKAN_HEADERS_FOUND)
    set(VULKAN_CORE_HEADER "${VULKAN_HEADERS_INCLUDE_DIRS}/vulkan/vulkan_core.h")
    if(EXISTS "${VULKAN_CORE_HEADER}")
        file(READ "${VULKAN_CORE_HEADER}" VULKAN_CORE_CONTENT)
        string(REGEX MATCH "VK_HEADER_VERSION[ \t]+([0-9]+)" _ "${VULKAN_CORE_CONTENT}")
        if(CMAKE_MATCH_1)
            set(VULKAN_HEADERS_VERSION "${CMAKE_MATCH_1}")
            message(STATUS "Vulkan headers version: ${VULKAN_HEADERS_VERSION}")
        endif()
    endif()
endif()

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VulkanHeaders
    REQUIRED_VARS VULKAN_HEADERS_INCLUDE_DIRS
    VERSION_VAR VULKAN_HEADERS_VERSION
)

# Set cache variables
mark_as_advanced(
    VULKAN_HEADERS_INCLUDE_DIR
    VULKAN_HEADERS_INCLUDE_DIRS
    VULKAN_HEADERS_VERSION
)

# Create imported target for modern cmake usage
if(VULKAN_HEADERS_FOUND AND NOT TARGET VulkanHeaders::VulkanHeaders)
    add_library(VulkanHeaders::VulkanHeaders INTERFACE IMPORTED)
    set_target_properties(VulkanHeaders::VulkanHeaders PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${VULKAN_HEADERS_INCLUDE_DIRS}"
    )
endif()
