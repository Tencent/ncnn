### the comprehensive model loading api table

|load from|alexnet.param|alexnet.param.bin|alexnet.bin|
|---|---|---|---|
|file path|load_param(const char*)|load_param_bin(const char*)|load_model(const char*)|
|file descriptor|load_param(FILE*)|load_param_bin(FILE*)|load_model(FILE*)|
|file memory|load_param_mem(const char*)|load_param(const unsigned char*)|load_model(const unsigned char*)|
|android asset|load_param(AAsset*)|load_param_bin(AAsset*)|load_model(AAsset*)|
|android asset path|load_param(AAssetManager*, const char*)|load_param_bin(AAssetManager*, const char*)|load_model(AAssetManager*, const char*)|
|custom IO reader|load_param(const DataReader&)|load_param_bin(const DataReader&)|load_mocel(const DataReader&)|

### points to note

1. Either of the following combination shall be enough for loading model
    * alexnet.param + alexnet.bin
    * alexnet.param.bin + alexnet.bin

2. Never modify Net opt member after loading

3. Most loading functions return 0 if success, except loading alexnet.param.bin and alexnet.bin from file memory, which returns the bytes consumed after loading
    * int Net::load_param(const unsigned char*)
    * int Net::load_model(const unsigned char*)

4. It is recommended to load model from Android asset directly to avoid copying them to sdcard on Android platform

5. The custom IO reader interface can be used to implement on-the-fly model decryption and loading
