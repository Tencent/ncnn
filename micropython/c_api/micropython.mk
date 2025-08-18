NCNN_MOD_DIR := $(USERMOD_DIR)

NCNN_CFLAGS = -DNCNN_STRING=1 -DNCNN_STDIO=1 -DNCNN_PIXEL=1 -DNCNN_PIXEL_DRAWING=1

SRC_USERMOD += $(NCNN_MOD_DIR)/ncnn_module.c
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/c_api.cpp

CFLAGS_USERMOD += -I$(NCNN_MOD_DIR) -I$(NCNN_MOD_DIR)/../../build_micropython/install/include -fopenmp $(NCNN_CFLAGS)

NCNN_LIB_PATH := $(shell find $(NCNN_MOD_DIR)/../../build_micropython/install -name "libncnn.*" -type f | head -1 | xargs dirname)

LDFLAGS_USERMOD += -L$(NCNN_LIB_PATH) -lncnn -fopenmp -lstdc++

CXXFLAGS_USERMOD += -I$(NCNN_MOD_DIR) -I$(NCNN_MOD_DIR)/../../build_micropython/install/include -fopenmp $(NCNN_CFLAGS)