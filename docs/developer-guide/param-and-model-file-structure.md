## net.param
### example
```
7767517
3 3
Input         input    0 1 data 0=4 1=4 2=1
InnerProduct  ip       1 1 data fc 0=10 1=1 2=80
Softmax       softmax  1 1 fc prob 0=0
```
### overview
```
[magic]
```
* magic number : 7767517
```
[layer count] [blob count]
```
* layer count : count of the layer line follows, should be exactly the count of all layer names
* blob count : count of all blobs, usually greater than or equals to the layer count
### layer line

each layer must occupy exactly one physical line, including all blob names and parameters; do not split a layer across lines or put multiple layers on the same line

Custom `DataReader::scan()` implementations and C API `ncnn_datareader_t::scan` callbacks must follow scanf conversion and input consumption rules, including field widths and scansets. Parameter parsing uses `%1023[^\r\n]` to read up to 1023 characters without skipping leading whitespace or consuming CR/LF, repeating the scan for longer lines. A successful scanset conversion must append a null terminator and return 1. Returning 0 for an unsupported format can be interpreted as an empty parameter list and silently select default parameter values.

```
[layer type] [layer name] [input count] [output count] [input blobs] [output blobs] [layer specific params]
```
* layer type : type name, such as Convolution Softmax etc
* layer name : name of this layer, must be unique among all layer names
* input count : count of the blobs this layer needs as input
* output count : count of the blobs this layer produces as output
* input blobs : name list of all the input blob names, separated by space, must be unique among input blob names of all layers
* output blobs : name list of all the output blob names, separated by space, must be unique among output blob names of all layers
* layer specific params : key=value pair list, separated by space
### layer param
```
0=1 1=2.5 -23303=2,2.0,3.0
```
key index should be unique in each layer line, pair can be omitted if the default value used

the meaning of existing param key index can be looked up at [operation-param-weight-table](operation-param-weight-table)

* integer or float key : index 0 ~ 19
* integer value : int
* float value : float
* integer array or float array key : -23300 minus index 0 ~ 19
* integer array value : [array size],int,int,...,int
* float array value : [array size],float,float,...,float

Use a decimal point or exponent when generating floating-point scalar values, including integral values, for example `1=6.0`, `1=6e0`, or `1=0.0`. When loading text parameters, the float getter also converts integer spellings such as `1=6` and `1=0` to `6.0f` and `0.0f`. The int getter does not convert floating-point parameters to integers.

Keep floating-point spellings when converting models with `ncnn2mem`: binary scalar parameters do not retain integer/float type tags, and the converter writes integer spellings as integer bit patterns without this numeric conversion.

Use a decimal point or exponent for every element of a floating-point array, including integral values, for example `-23303=2,1.0,2.0`. Mixed integer and float element spellings within an array are not defined by the format.

A zero-length array such as `-23300=0` explicitly supplies an empty array. Array getters return that empty array even when a nonempty default is supplied; omitting the parameter returns the default.

In modern ncnn param file

* array could be represented as `3=2.0,3.0` that is much more human friendly
* string typed value: `4=hello` and the string is no longer than 255

## net.bin
```
  +---------+---------+---------+---------+---------+---------+
  | weight1 | weight2 | weight3 | weight4 | ....... | weightN |
  +---------+---------+---------+---------+---------+---------+
  ^         ^         ^         ^
  0x0      0x80      0x140     0x1C0
```
the model binary is the concatenation of all weight data, each weight buffer is aligned by 32bit

### weight buffer
```
[flag] (optional)
[raw data]
[padding] (optional)
```
* flag : unsigned int,  little-endian, indicating the weight storage type, 0 => float32, 0x01306B47 => float16, 0x01348B83 => bfloat16, otherwise => quantized int8, may be omitted if the layer implementation forced the storage type explicitly
* raw data : raw weight data, little-endian, float32 data or float16 data or bfloat16 data or quantized table and indexes depending on the storage type flag
* padding : padding space for 32bit alignment, may be omitted if already aligned
