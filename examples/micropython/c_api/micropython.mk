NCNN_MOD_DIR := $(USERMOD_DIR)

SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/ncnn_module.cpp

CFLAGS_USERMOD += -I$(NCNN_MOD_DIR) -I$(NCNN_MOD_DIR)/../../../build_micropython/install/include -fopenmp

NCNN_LIB_PATH := $(shell find $(NCNN_MOD_DIR)/../../../build_micropython/install -name "libncnn.*" -type f | head -1 | xargs dirname)

LDFLAGS_USERMOD += -L$(NCNN_LIB_PATH) -lncnn -fopenmp -lstdc++

CXXFLAGS_USERMOD += -I$(NCNN_MOD_DIR) -I$(NCNN_MOD_DIR)/../../../build_micropython/install/include -fopenmp