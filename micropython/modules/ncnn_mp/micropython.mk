NCNN_MP_MOD_DIR := $(USERMOD_DIR)
NCNN_INSTALL_PREFIX ?= $(NCNN_MP_MOD_DIR)/../../ncnn/build/install

SRC_USERMOD_C += $(NCNN_MP_MOD_DIR)/ncnn_mp.c

# Include directories
CFLAGS_USERMOD += -I$(NCNN_INSTALL_PREFIX)/include

# Other compiler flags
CFLAGS_USERMOD += -Wno-unused-function

# Linker flags
LDFLAGS_USERMOD += -L$(NCNN_INSTALL_PREFIX)/lib

ifeq ($(DEBUG), 1)
    NCNN_LIB_NAME := ncnnd
else
    NCNN_LIB_NAME := ncnn
endif

ifeq ($(USE_VULKAN), 1)
    LDFLAGS_USERMOD += -l$(NCNN_LIB_NAME) -lglslang
else
    LDFLAGS_USERMOD += -l$(NCNN_LIB_NAME)
endif

LDFLAGS_USERMOD += -lstdc++