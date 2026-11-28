# PNNX Architecture (PyTorch Neural Network eXchange)

PNNX converts PyTorch (TorchScript), ONNX, and TNN models to both a PNNX intermediate representation and the final ncnn format.

## Directory Structure

```
tools/pnnx/
├── CMakeLists.txt         # Build config (requires PyTorch, optional ONNX/Protobuf)
├── src/
│   ├── main.cpp           # Entry point
│   ├── ir.h / ir.cpp      # PNNX intermediate representation (Graph, Operator, Operand)
│   ├── load_torchscript.cpp   # TorchScript model loader
│   ├── load_onnx.cpp          # ONNX model loader
│   ├── save_ncnn.cpp          # ncnn param/bin exporter
│   ├── pass_level1/       # Level 1 passes (TorchScript graph cleanup)
│   ├── pass_level2/       # Level 2 passes (op recognition: F_relu.cpp, nn_Conv2d.cpp, ...)
│   ├── pass_level3/       # Level 3 passes (expression fusion, constant folding)
│   ├── pass_level4/       # Level 4 passes (canonicalization, dead code elimination)
│   ├── pass_level5/       # Level 5 passes (ncnn-specific optimizations, op fusion)
│   ├── pass_ncnn/         # Final ncnn lowering passes
│   └── pass_onnx/         # ONNX-specific passes
└── tests/                 # PNNX tests (Python + C++)
    ├── test_F_relu.py     # Python test script (defines model, exports, converts, compares)
    ├── ncnn/              # ncnn C++ test harnesses
    └── onnx/              # ONNX test harnesses
```

## PNNX IR (Intermediate Representation)

```cpp
namespace pnnx {

class Parameter {     // Scalar or array parameter (bool, int, float, string, arrays thereof)
    int type;         // 0=null 1=bool 2=int 3=float 4=string 5=int[] 6=float[] 7=string[] 10=complex 11=complex[]
    bool b; int i; float f; std::string s;
    std::vector<int> ai; std::vector<float> af; std::vector<std::string> as;
};

class Attribute {     // Tensor weight data
    int type;         // 1=f32 2=f64 3=f16 4=i32 ... 13=bf16
    std::vector<int> shape;
    std::vector<char> data;
};

class Operand {       // Edge in the graph
    Operator* producer;
    std::vector<Operator*> consumers;
    int type;
    std::vector<int> shape;
    std::string name;
};

class Operator {      // Node in the graph
    std::string type;   // e.g., "nn.Conv2d", "F.relu"
    std::string name;
    std::vector<Operand*> inputs;
    std::vector<Operand*> outputs;
    std::map<std::string, Parameter> params;
    std::map<std::string, Attribute> attrs;
};

class Graph {         // The computation graph
    std::vector<Operator*> ops;
    std::vector<Operand*> operands;
};

}
```

## Conversion Pipeline

```
Input Model (TorchScript / ONNX / TNN)
    │
    ▼
  Load → pnnx::Graph  (pass_level1: raw graph from framework)
    │
    ▼
  pass_level2  (op recognition: map framework ops to PNNX ops)
    │
    ▼
  pass_level3  (expression fusion, constant folding, cleanup)
    │
    ▼
  pass_level4  (canonicalization, dead code elimination)
    │
    ▼
  pass_level5  (ncnn-specific fusion: conv+relu, conv+bn, etc.)
    │
    ▼
  pass_ncnn    (lower PNNX ops to ncnn ops, adjust params)
    │
    ▼
  save_ncnn    (write .param and .bin files)
```

Outputs:
- `*.pnnx.param` / `*.pnnx.bin` — PNNX IR format
- `*.pnnx.py` — Python reconstruction code
- `*.pnnx.onnx` — ONNX export (if built with ONNX support)
- `*.ncnn.param` / `*.ncnn.bin` — ncnn format (ready for inference)
