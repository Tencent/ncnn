## current param load api
### Cons
#### long and awful code
#### three functions
#### not extensible
#### no default value
#### no variable length array
```
MyLayer  mylayer 1 1 in out 100 1.250000
```
```
binary 100
binary 1.250000
```
```cpp
#if NCNN_STDIO
#if NCNN_STRING
int MyLayer::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %f", &a, &b);
    if (nscan != 2)
    {
        fprintf(stderr, "MyLayer load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int MyLayer::load_param_bin(FILE* paramfp)
{
    fread(&a, sizeof(int), 1, paramfp);

    fread(&b, sizeof(float), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int MyLayer::load_param(const unsigned char*& mem)
{
    a = *(int*)(mem);
    mem += 4;

    b = *(float*)(mem);
    mem += 4;

    return 0;
}
```

## new param load api proposed
### Pros
#### clean and simple api
#### default value
#### extensible
#### variable length array
```
7767517
MyLayer  mylayer 1 1 in out 0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0
```
```
binary 0xDD857600(magic)

binary 0
binary 100
binary 1
binary 1.250000
binary -23303
binary 5
binary 0.1
binary 0.2
binary 0.4
binary 0.8
binary 1.0
binary -233(EOP)
```
```cpp
int MyLayer::load_param(const ParamDict& pd)
{
    // pd.get( param id (seq), default value );
    a = pd.get(0, 100);
    b = pd.get(1, 1.25f);

    // get default value for c if not specified in param file
    c = pd.get(2, 0.001);

    // get array
    d = pd.get(3, Mat(len, array));
    return 0;
}
```
