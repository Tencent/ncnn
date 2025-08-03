NCNN_MP_MOD_DIR := $(USERMOD_DIR)
NCNN_DIR := $(NCNN_MP_MOD_DIR)/../../ncnn
NCNN_BUILD_DIR := $(NCNN_DIR)/build/src

SRC_USERMOD_C += $(NCNN_MP_MOD_DIR)/ncnn_mp.c

# # Add OpenMP flag for C and C++ compilers
# CFLAGS_USERMOD += -fopenmp
# CXXFLAGS_USERMOD += -fopenmp

# Include directories
CXXFLAGS_USERMOD += -I$(NCNN_DIR)/src
CFLAGS_USERMOD += -I$(NCNN_DIR)/src
CXXFLAGS_USERMOD += -I$(NCNN_BUILD_DIR)
CFLAGS_USERMOD += -I$(NCNN_BUILD_DIR)

# Other compiler flags
CXXFLAGS_USERMOD += -std=c++11
CFLAGS_USERMOD += -Wno-unused-function
CXXFLAGS_USERMOD += -Wno-unused-function

# Linker flags
LDFLAGS_USERMOD += -L$(NCNN_BUILD_DIR)
LDFLAGS_USERMOD += -lncnn
LDFLAGS_USERMOD += -lstdc++
# LDFLAGS_USERMOD += -fopenmp