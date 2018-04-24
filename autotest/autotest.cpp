#include <stdio.h>

#include "blob.h"
#include "net.h"
#include "layer.h"
#include "mat.h"
#include "opencv.h"
#include "platform.h"

#include "test_convolution.h"
#include "test_innerproduct.h"
#include "gtest/gtest.h"

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
