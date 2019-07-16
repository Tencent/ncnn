find_package(Vulkan REQUIRED)

add_library(Vulkan UNKNOWN IMPORTED)
set_target_properties(Vulkan PROPERTIES IMPORTED_LOCATION ${Vulkan_LIBRARY})
set_target_properties(Vulkan PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${Vulkan_INCLUDE_DIR})
