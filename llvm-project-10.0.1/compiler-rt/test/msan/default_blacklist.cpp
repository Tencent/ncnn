// Test that MSan uses the default blacklist from resource directory.
// RUN: %clangxx_msan -### %s 2>&1 | FileCheck %s
// CHECK: fsanitize-system-blacklist={{.*}}msan_blacklist.txt
