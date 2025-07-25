NCNN_MOD_DIR := $(USERMOD_DIR)

SRC_USERMOD_CXX += $(NCNN_MOD_DIR)/ncnn_module.cpp

CFLAGS_USERMOD += -I$(NCNN_MOD_DIR) -I$(NCNN_MOD_DIR)/../../../build_micropython/install/include -fopenmp

LDFLAGS_USERMOD += -L$(NCNN_MOD_DIR)/../../../build_micropython/install/lib -lncnn -fopenmp  -lstdc++

CXXFLAGS_USERMOD += -I$(NCNN_MOD_DIR) -I$(NCNN_MOD_DIR)/../../../build_micropython/install/include -fopenmp