# MicroPython ncnn module makefile

NCNN_MOD_DIR := $(USERMOD_DIR)

# Add all C files to SRC_USERMOD.
SRC_USERMOD += $(NCNN_MOD_DIR)/ncnn_module.c

# We can add our module folder to include paths if needed
CFLAGS_USERMOD += -I$(NCNN_MOD_DIR)

# Add ncnn library path - assuming ncnn is built in the main directory
NCNN_ROOT := $(NCNN_MOD_DIR)/../..
CFLAGS_USERMOD += -I$(NCNN_ROOT)/src
CFLAGS_USERMOD += -I$(NCNN_ROOT)/build/src
CFLAGS_USERMOD += -DNCNN_C_API=1

# Link with ncnn library
LDFLAGS_USERMOD += -L$(NCNN_ROOT)/build/src -lncnn -lstdc++

# Add any additional compiler flags for ncnn
CFLAGS_USERMOD += -std=c++11

# Enable required ncnn features
CFLAGS_USERMOD += -DNCNN_STDIO=1
CFLAGS_USERMOD += -DNCNN_STRING=1
CFLAGS_USERMOD += -DNCNN_PIXEL=1