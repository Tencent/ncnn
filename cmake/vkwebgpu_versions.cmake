# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# This file is the single source of truth for the vkwebgpu dependency version set.
# Keep bootstrap scripts, CMake diagnostics, tests, and CI consumers derived from
# these variables instead of copying version strings.

set(VKWEBGPU_DAWN_COMMIT "0bc38adde72b79013536f8ce354b639ae19ae195")
set(VKWEBGPU_DAWN_RELEASE "v20260720.160313")
set(VKWEBGPU_DAWN_DEPS_SHA256 "237449559ddc1888fd5ba62a9fcc0b21a53f21afa061c60ff8f077c6f7364075")

set(VKWEBGPU_EMDAWNWEBGPU_PORT_FILENAME "emdawnwebgpu-v20260720.160313.remoteport.py")
set(VKWEBGPU_EMDAWNWEBGPU_PORT_URL "https://github.com/google/dawn/releases/download/v20260720.160313/emdawnwebgpu-v20260720.160313.remoteport.py")
set(VKWEBGPU_EMDAWNWEBGPU_PORT_SHA256 "ec2b01bab7f853e36ada4620821b1821baee9bbfab16f1350ecbb22856b25a0d")
set(VKWEBGPU_EMDAWNWEBGPU_PACKAGE_SHA512 "88f2f3de88652b145374c2b607888f2edb2ab30cd9c81b159bc24dbe06e70d6a093fa097965cf4bdb6953aec9990e99f568f9b46d8b71dacad49afcc4498b16c")

set(VKWEBGPU_EMSDK_VERSION "5.0.6")
set(VKWEBGPU_ASYNCIFY_STACK_SIZE "65536")
set(VKWEBGPU_WASM_STACK_SIZE "1048576")

set(VKWEBGPU_MINIMUM_CHROME_MAJOR "149")
set(VKWEBGPU_CHROME_VERSION "151.0.7922.47")
set(VKWEBGPU_CHROME_REVISION "1654411")
set(VKWEBGPU_CHROME_LINUX_URL "https://storage.googleapis.com/chrome-for-testing-public/151.0.7922.47/linux64/chrome-linux64.zip")
set(VKWEBGPU_CHROME_LINUX_SHA256 "14ac03a67e154e3f8bbc57e03ef03315fda8fedff8e045eee8b31500283a33f4")

set(VKWEBGPU_NODE_VERSION "24.18.0")
set(VKWEBGPU_PUPPETEER_CORE_VERSION "25.4.0")
