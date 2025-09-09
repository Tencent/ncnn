NCNN_MOD_DIR := $(USERMOD_DIR)

NCNN_INSTALL_DIR := $(NCNN_MOD_DIR)/../../build_micropython/install

NCNN_LIB_PATH := $(shell find $(NCNN_INSTALL_DIR) -name "libncnn.*" -type f | head -1 | xargs dirname)
NCNN_INCLUDE_PATH := $(NCNN_INSTALL_DIR)/include

SRC_USERMOD += $(NCNN_MOD_DIR)/src/core/ncnn_module.c
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/version.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/allocator.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/option.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/mat.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/mat_pixel.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/mat_pixel_drawing.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/mat_process.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/extractor.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/layer.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/net.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/datareader.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/paramdict.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/modelbin.cpp
SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/src/api/blob.cpp

CFLAGS_USERMOD += -I$(NCNN_MOD_DIR)/include -I$(NCNN_INCLUDE_PATH) $(NCNN_CFLAGS)

ifdef IDF_TARGET
    LDFLAGS_USERMOD += -Wl,--allow-multiple-definition
endif

LDFLAGS_USERMOD += -L$(NCNN_LIB_PATH) -lncnn -lstdc++

CXXFLAGS_USERMOD += -I$(NCNN_MOD_DIR)/include -I$(NCNN_INCLUDE_PATH) $(NCNN_CFLAGS)