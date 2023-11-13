mkdir -p ncnn_model
./pnnx output/3d_net.pt inputshape=[1,3,10,20,30] ncnnparam=ncnn_model/3d_net.param ncnnbin=ncnn_model/3d_net.bin
