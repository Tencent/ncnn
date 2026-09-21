Additional model and layer parameter validation is enabled by default through the `NCNN_VALIDATION` build option. Builds with `NCNN_VALIDATION=OFF` require prevalidated models and do not guarantee rejection of invalid parameters. Parameter formats, defaults and numeric conversions are unchanged. See [build minimal library](../how-to-use-and-FAQ/build-minimal-library.md#disable-ncnn_validation).

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

* integer or float key : index 0 ~ 31
* integer value : int
* float value : float
* integer array or float array key : -23300 minus index 0 ~ 31
* integer array value : [array size],int,int,...,int
* float array value : [array size],float,float,...,float

Use a decimal point or exponent when generating floating-point scalar values, including integral values, for example `1=6.0`, `1=6e0`, or `1=0.0`. When loading text parameters, the float getter also converts integer spellings such as `1=6` and `1=0` to `6.0f` and `0.0f`. The int getter does not convert floating-point parameters to integers.

Keep floating-point spellings when converting models with `ncnn2mem`: binary scalar parameters do not retain integer/float type tags, and the converter writes integer spellings as integer bit patterns without this numeric conversion.

Use a decimal point or exponent for every element of a floating-point array, including integral values, for example `-23303=2,1.0,2.0`. Mixed integer and float element spellings within an array are not defined by the format.

This also matters when using `ncnn2mem`. Some layers convert integer text arrays to floating-point values during `load_param`, but `ncnn2mem` writes the array element bits without that conversion. Binary arrays do not retain integer/float type tags, so the layer cannot recover the original text element type. For example, write activation parameters as `-23310=2,-1.0,2.0`, not `-23310=2,-1,2`, when converting to binary. The latter preserves integer bit patterns rather than the intended floating-point values.

Keep integer arrays, such as axes and slice indices, in integer form. Legacy YOLO mask arrays also use integer spellings to preserve float bit patterns. Converting every integer array numerically to floats would corrupt these parameters.

Layers that expect integer arrays, including axes, slice indices, and Einsum character codes, reject floating-point text arrays rather than converting their elements to integers. For example, write CopyTo starts as `-23309=2,0,1`, not `-23309=2,0.0,1.0`. The mixed spelling `-23309=2,0,1.0` is also invalid; the parser stores each element according to its spelling, so the bits for `1.0` would be read as the integer `1065353216`, not `1`.

A zero-length array such as `-23300=0` explicitly supplies an empty array. Array getters return that empty array even when a nonempty default is supplied; omitting the parameter returns the default.

In modern ncnn param file

* array could be represented as `3=2.0,3.0` that is much more human friendly
* string typed value: `4=hello` and the string is no longer than 255

### shape hints

An output blob shape may be declared inline in the layer line, so that consumers such as `ncnnoptimize`
do not have to re-derive it from real data. The hints are stored in param index `30`, therefore the text
spelling is the integer array `-23330` (`-23300 - 30`).

```
-23330=<count>,<dims>,<w>,<h>[,<d>],<c>[,<dims>,<w>,<h>[,<d>],<c>]...
```

* `<count>` is the total number of integers that follow, that is `output count * step`
* each output blob of the layer contributes exactly one record, in output blob order
* `step` is the number of integers per record: `4` or `5`, and `dims` 4 requires `step` 5.
  `ncnnoptimize` uses `step` 5 for every record of a layer
* `step` must be identical for every output of the same layer. A layer with both 3-dim and 4-dim outputs
  therefore uses `step` 5 for all of them, and the unused `<d>` slot of the 3-dim records is ignored
* `dims` 0 means "no hint for this output" and skips the whole record
* `w` is the innermost dimension, then `h`, then `d`, with `c` outermost, matching `ncnn::Mat(w,h,d,c)`.
  Hint shapes describe a single sample, so they do not carry the batch count (`Mat::n`)

```
Input         in    0 1 in0  -23330=5,4,7,6,5,4                # one 4-dim output, w=7 h=6 d=5 c=4
Input         in    0 2 in0 in1 -23330=10,3,7,6,1,5,4,4,3,2,32 # 3-dim w=7 h=6 c=5 and 4-dim w=4 h=3 d=2 c=32
```

Hints are optional. A model without them behaves exactly as before, and a runtime that does not
understand param index `30` stores and ignores it. `ncnnoptimize` derives the shapes with its
shape inference and writes the hints for every layer whose output shapes are all known; a layer with
any unknown output shape gets no hint at all. When `NCNN_VALIDATION` is enabled a malformed hint
(wrong element count, `step` other than 4 or 5, negative sizes) rejects the model, and param index
`30` must be an integer or float array; any other type is rejected.

## net.param.bin

Binary custom layer type indexes must include `LayerType::CustomBit`. Untagged unknown type indexes are rejected instead of falling back to a custom registry slot. `ncnn2mem` adds this tag automatically.

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
