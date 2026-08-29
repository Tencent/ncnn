// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

// ---------------------------------------------------------------------------
// emit：Pt2Graph（规范 pnnx IR）-> pnnx::Graph（纯建图）
// 方案2：aten→pnnx 映射、形状推理、参数补全、权重接线 都已在 Python 预处理完成并写入 JSON，
// C++ 这里只负责按 JSON 建 op/operand/param/attr，不做任何映射/推断逻辑。
// ---------------------------------------------------------------------------
static Parameter to_parameter(const Pt2Param& p)
{
    switch (p.type)
    {
    case 1: return Parameter(p.b);                                   // bool
    case 2: return Parameter((int)p.i);                              // int
    case 3: return Parameter((float)p.f);                            // float
    case 4: return Parameter(p.s);                                   // string
    case 5: return Parameter(std::vector<int>(p.ii.begin(), p.ii.end()));     // ints
    case 6: return Parameter(std::vector<float>(p.ff.begin(), p.ff.end()));   // floats
    default: return Parameter();                                     // null
    }
}

int emit_pt2_graph(const Pt2Graph& g, Graph& pnnx_graph)
{
    auto get_or_create_operand = [&](const std::string& name) -> Operand* {
        Operand* r = pnnx_graph.get_operand(name);
        if (!r)
            r = pnnx_graph.new_operand(name);
        return r;
    };

    // 1) 用户输入 -> pnnx.Input（shape/dtype 来自 Python 提取的 tensor_meta）
    for (size_t i = 0; i < g.inputs.size(); i++)
    {
        const Pt2IO& io = g.inputs[i];
        Operator* op = pnnx_graph.new_operator("pnnx.Input", io.name);
        Operand* r = get_or_create_operand(io.name);
        r->producer = op;
        op->outputs.push_back(r);
        if (!io.shape.empty())
        {
            r->shape = io.shape;
            r->type = 1; // f32
            // batch 在首轴（与 torchscript 路径约定一致，flatten/cat 等 pass 依赖此判定）
            r->params["__ncnn_batch_axis"] = Parameter(0);
        }
    }

    // 2) 各算子节点（规范 pnnx_type + params + 内联 attrs）
    for (size_t i = 0; i < g.nodes.size(); i++)
    {
        const Pt2Node& n = g.nodes[i];
        Operator* op = pnnx_graph.new_operator(n.pnnx_type, n.name);

        // 输入 operand（权重不在 inputs 里，在 attrs）
        for (size_t j = 0; j < n.inputs.size(); j++)
        {
            Operand* r = get_or_create_operand(n.inputs[j]);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
            op->inputnames.push_back(n.inputs[j]);
        }

        // 参数
        for (std::map<std::string, Pt2Param>::const_iterator it = n.params.begin(); it != n.params.end(); ++it)
        {
            op->params[it->first] = to_parameter(it->second);
        }

        // 内联权重属性（module 形态算子的 @weight/@bias 等）
        for (std::map<std::string, Pt2Attr>::const_iterator it = n.attrs.begin(); it != n.attrs.end(); ++it)
        {
            Attribute a;
            a.type = 1; // f32
            a.shape = it->second.shape;
            a.set_float32_data(it->second.data);
            op->attrs[it->first] = a;
        }

        // 输出 operand（shape 来自 tensor_meta）
        for (size_t j = 0; j < n.outputs.size(); j++)
        {
            Operand* r = get_or_create_operand(n.outputs[j]);
            r->producer = op;
            op->outputs.push_back(r);
            if (j < n.output_shapes.size() && !n.output_shapes[j].empty())
            {
                r->shape = n.output_shapes[j];
                r->type = 1; // f32
            }
        }
    }

    // 3) 用户输出 -> pnnx.Output
    for (size_t i = 0; i < g.outputs.size(); i++)
    {
        const Pt2IO& io = g.outputs[i];
        Operand* r = get_or_create_operand(io.name);
        std::string op_name = "pnnx_Output_" + io.name;
        Operator* op = pnnx_graph.new_operator("pnnx.Output", op_name);
        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// 预处理器：把真实 .pt2（torch.export.save 产出的 ExportedProgram）反序列化，
// 转成规范 pnnx IR 中间 zip（models/model.json + data/N 权重字节）。
// 方案2：Python 侧完成 aten→pnnx 映射 + tensor_meta 形状提取 + 参数补全 + 权重接线。
// torch.export.load 会剥掉 node.meta["tensor_meta"]，故从 node.meta["val"](FakeTensor) 取形状。
// ---------------------------------------------------------------------------
#if defined _WIN32
#define PNNX_POPEN _popen
#define PNNX_PCLOSE _pclose
#else
#define PNNX_POPEN popen
#define PNNX_PCLOSE pclose
#endif

static const char* kPt2PreprocessPy = R"PT2PYEOF(
import os, sys, json, zipfile
import torch

DTYPE_MAP = {torch.float32: 7, torch.float16: 5, torch.float64: 6, torch.int64: 4, torch.int32: 3, torch.bool: 1}

def die(msg):
    sys.stderr.write("pt2_preprocess: " + str(msg) + "\n"); sys.exit(1)

in_path = os.environ.get("PNNX_PT2_IN"); out_path = os.environ.get("PNNX_PT2_OUT")
if not in_path or not out_path: die("missing PNNX_PT2_IN / PNNX_PT2_OUT")
try: ep = torch.export.load(in_path)
except Exception as e: die("torch.export.load failed: %s" % e)

graph = ep.graph; sig = ep.graph_signature
sd_attr = ep.state_dict
sd = sd_attr if isinstance(sd_attr, dict) else dict(sd_attr())

def tm(node):
    # -> (sizes, dtype_int)；torch.export.load 剥掉 tensor_meta，退回 meta['val'](FakeTensor)
    meta = node.meta
    get = meta.get if hasattr(meta, "get") else (lambda k, d=None: meta[k] if k in meta else d)
    m = get("tensor_meta")
    if m is not None:
        sz = m.shape if hasattr(m, "shape") else (m.sizes if hasattr(m, "sizes") else None)
        sizes = list(sz) if sz is not None else None
        return sizes, DTYPE_MAP.get(m.dtype, 7)
    v = get("val")
    if v is not None and hasattr(v, "shape"):
        return list(v.shape), DTYPE_MAP.get(v.dtype, 7)
    # aten 多输出（tuple/namedtuple，如 max_pool*_with_indices / adaptive_max_pool*）：
    # 节点 val 是 tuple，值输出 = 第 0 元素（indices 是第 1 元素）。取不到 shape 会让
    # 输出 operand 元数据为空，下游 solve_batch_index / Reshape 插入全部走偏。
    if isinstance(v, tuple) and len(v) > 0 and hasattr(v[0], "shape"):
        return list(v[0].shape), DTYPE_MAP.get(v[0].dtype, 7)
    return None, None

def aten_name(target):
    s = str(target); p = s.rfind("aten.")
    op = s[p+5:] if p >= 0 else s
    d = op.rfind(".")
    return op[:d] if d >= 0 else op

# 参数节点名 -> state_dict 键
param_sd = {}
for spec in sig.input_specs:
    k = spec.kind; kn = k.name if hasattr(k, "name") else str(k)
    if kn in ("PARAMETER", "CONSTANT_TENSOR", "BUFFER"):
        try: param_sd[spec.arg.name] = spec.target
        except Exception: pass

wi = [0]; wentries = {}
def add_weight(node):
    sd_key = param_sd.get(node.name)
    if sd_key is None or sd_key not in sd: return None
    t = sd[sd_key]
    if t.dtype != torch.float32: return None
    entry = "data/%d" % wi[0]; wi[0] += 1
    wentries[entry] = t.detach().cpu().contiguous().numpy().astype("float32").tobytes()
    return {"data_path": entry, "shape": [int(x) for x in t.shape], "dtype": 7}

def is_weight(a): return isinstance(a, torch.fx.Node) and a.name in param_sd
def is_tensor(a): return isinstance(a, torch.fx.Node) and not is_weight(a)

def as_ints(v, n=None):
    """aten 的 int/int[] 参数规整成 list。
    None 或空 list -> None（pnnx 侧写作 None，如未给 stride）；标量 int 按 n 维展开。"""
    if v is None: return None
    if isinstance(v, (list, tuple)):
        if len(v) == 0: return None
        return [int(x) for x in v]
    if isinstance(v, bool): return [int(v)]
    if isinstance(v, int):
        return [v] if n is None else [v] * n
    try:
        return [int(v)]
    except Exception:
        return None

def as_bool(v, default=False):
    return default if v is None else bool(v)

def ndim_of(suffix):
    return 1 if suffix == "1d" else (3 if suffix == "3d" else 2)

def scalar_arg(node, idx, name, default):
    """取标量参数：优先 kwargs[name]，否则 args[idx]，都没有用 default。
    关键：不能用 is_tensor 把标量过滤掉——pnnx IR 里这些是算子的**命名参数**，
    丢了会导致 pass_ncnn 转换器匹配不上（如 F.leaky_relu 缺 negative_slope 就转不成 ReLU，
    F.celu 缺 alpha 转不成 CELU，算子会原样留在 ncnn param 里）。"""
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > idx and node.args[idx] is not None:
        return node.args[idx]
    return default

def nn_module_of(node):
    """node 最内层 nn 模块类名（如 ReLU6 / AdaptiveAvgPool2d）；非 nn 模块返回 None。
    torch.export.load **保留** meta["nn_module_stack"]，形如
    {'L__self__r6': ('r6', 'torch.nn.modules.activation.ReLU6')}。
    用途：torchscript 路径能看出算子来自 nn.X 模块还是 F.x 调用（两者 pnnx IR 形态与
    最终 ncnn op 名不同，如 nn.ReLU6 vs F.hardtanh、nn.AdaptiveAvgPool2d vs F.adaptive_avg_pool2d），
    而 aten 层面二者是同一个 op——只能靠 module stack 区分。"""
    get = node.meta.get if hasattr(node.meta, "get") else None
    if get is None:
        return None
    try:
        ms = get("nn_module_stack")
    except Exception:
        return None
    if not ms:
        return None
    try:
        items = list(ms.items())
        if not items:
            return None
        val = items[-1][1]
        path = val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else str(val)
        return path.rsplit(".", 1)[-1]
    except Exception:
        return None

def output_size_canonical(osz, in_node, d):
    """adaptive/max 池化的 output_size 规范化。
    torch.export 会把 output_size 里的 None 实例化成输入空间维尺寸(如 (None,3) -> (24,3)),
    而 torchscript 路径保留 (0,3) -> pass_ncnn 对 0 维写 -233 哨兵(该维自适应保持)。
    这里把"等于输入空间维尺寸"的元素还原成 0 对齐 ts 规范形态——数值上
    输出=输入维的 adaptive 池化就是恒等池化,与"保持该维"语义等价,安全。"""
    ishp, _ = tm(in_node) if in_node is not None else (None, None)
    if isinstance(osz, (list, tuple)):
        sz = [int(x) for x in osz][-d:]
        if ishp and len(ishp) >= 2:
            spatial = [int(x) for x in ishp[2:]][-d:]
            sz = [0 if i < len(spatial) and sz[i] == spatial[i] else sz[i] for i in range(len(sz))]
        return sz
    if isinstance(osz, int):
        return [osz] * d
    return None

def emit_node(node):
    op = aten_name(node.target)
    out_sizes, out_dt = tm(node)
    out_name = node.name
    out_shapes = [out_sizes] if out_sizes is not None else []
    params = {}; attrs = {}; inputs = []

    if op == "relu":
        pt = "nn.ReLU"; inputs = [a.name for a in node.args if is_tensor(a)]
    elif op == "relu6":
        pt = "F.relu6"; inputs = [a.name for a in node.args if is_tensor(a)]
    elif op == "relu_":
        # inplace relu 与 relu 等价（pnnx IR 里就是 nn.ReLU）
        pt = "nn.ReLU"; inputs = [a.name for a in node.args if is_tensor(a)]
    elif op == "log_sigmoid":
        pt = "F.logsigmoid"; inputs = [a.name for a in node.args if is_tensor(a)]
    elif op == "leaky_relu":
        pt = "F.leaky_relu"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["negative_slope"] = float(scalar_arg(node, 1, "negative_slope", 0.01))
    elif op == "elu":
        pt = "F.elu"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["alpha"] = float(scalar_arg(node, 1, "alpha", 1.0))
    elif op == "celu":
        pt = "F.celu"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["alpha"] = float(scalar_arg(node, 1, "alpha", 1.0))
    elif op == "hardtanh":
        # nn.ReLU6 会分解成 aten.hardtanh(0,6)；pnnx IR 侧二者形态不同：
        #   nn.ReLU6  -> `nn.ReLU6 <n> 1 1 in out`（模块形态，无参数）
        #   F.hardtanh-> `F.hardtanh ... min_val=-1.0 max_val=1.0`（函数形态，带参数）
        # 最终 ncnn op 名也不同（relu6 vs htanh），只能靠 nn_module_stack 区分。
        inputs = [a.name for a in node.args if is_tensor(a)]
        if nn_module_of(node) == "ReLU6":
            pt = "nn.ReLU6"
        else:
            pt = "F.hardtanh"
            params["min_val"] = float(scalar_arg(node, 1, "min_val", -1.0))
            params["max_val"] = float(scalar_arg(node, 2, "max_val", 1.0))
    elif op == "relu6_":
        # inplace relu6 同样按模块形态处理
        pt = "nn.ReLU6"; inputs = [a.name for a in node.args if is_tensor(a)]
    elif op == "gelu":
        pt = "F.gelu"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["approximate"] = str(scalar_arg(node, 1, "approximate", "none"))
    elif op == "softplus":
        pt = "F.softplus"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["beta"] = float(scalar_arg(node, 1, "beta", 1.0))
        params["threshold"] = float(scalar_arg(node, 2, "threshold", 20.0))
    elif op in ("softshrink", "hardshrink"):
        pt = "F." + op; inputs = [a.name for a in node.args if is_tensor(a)]
        params["lambd"] = float(scalar_arg(node, 1, "lambd", 0.5))
    elif op == "threshold":
        pt = "F.threshold"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["threshold"] = float(scalar_arg(node, 1, "threshold", 0.0))
        params["value"] = float(scalar_arg(node, 2, "value", 0.0))
    elif op in ("sigmoid","tanh","silu","selu","mish","hardsigmoid","hardswish","logsigmoid","softsign"):
        pt = "F." + op; inputs = [a.name for a in node.args if is_tensor(a)]
    elif op in ("softmax","log_softmax","softmin"):
        pt = "F." + op; inputs = [a.name for a in node.args if is_tensor(a)]
        for i,a in enumerate(node.args[1:],1):
            if isinstance(a,int): params["dim"]=a
        if "dim" in node.kwargs: params["dim"]=node.kwargs["dim"]
    elif op == "flatten":
        pt = "torch.flatten"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["start_dim"] = node.kwargs.get("start_dim", node.args[1] if len(node.args)>1 else 0)
        params["end_dim"] = node.kwargs.get("end_dim", node.args[2] if len(node.args)>2 else -1)
    elif op in ("cat","stack"):
        pt = "torch." + op
        lst = node.args[0]
        inputs = [x.name for x in lst if isinstance(x, torch.fx.Node)]
        dim = node.kwargs.get("dim", node.args[1] if len(node.args)>1 else 0)
        params["dim"] = dim
    elif op in ("add","sub","mul","div"):
        # ncnn BinaryOp：0=op码, 1=with_scalar, 2=b(标量值)
        # 标量操作数必须走 with_scalar/b 参数，不能当独立 operand（也不能直接丢掉）
        pt = "BinaryOp"; a = node.args
        opc = {"add":0,"sub":1,"mul":2,"div":3}[op]
        tens = [x.name for x in a if is_tensor(x)]
        scalar = None
        for x in a:
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                scalar = float(x)
                break
        if len(tens) == 2:
            inputs = tens
            params["0"] = opc
        elif len(tens) == 1 and scalar is not None:
            inputs = tens
            params["0"] = opc
            params["1"] = 1          # with_scalar
            params["2"] = scalar     # b
        else:
            return None
    elif op in ("view","reshape","contiguous"):
        pt = "Tensor.reshape"; inputs = [a.name for a in node.args if is_tensor(a)]
        if out_sizes is not None: params["shape"] = list(out_sizes)
    elif op == "permute":
        pt = "Tensor.permute"; inputs = [a.name for a in node.args if is_tensor(a)]
        # aten.permute(self, dims)：dims 可能是单个 list([0,2,3,1]) 或逐个 int(0,2,3,1)
        dims = []
        if len(node.args) > 1:
            a1 = node.args[1]
            if isinstance(a1, (list, tuple)):
                dims = [int(x) for x in a1]
            else:
                dims = [int(a) for a in node.args[1:] if isinstance(a, int)]
        if "dims" in node.kwargs: dims = list(node.kwargs["dims"])
        params["dims"] = dims
    elif op == "transpose":
        pt = "torch.transpose"; inputs = [a.name for a in node.args if is_tensor(a)]
        params["dim0"] = node.args[1] if len(node.args)>1 else node.kwargs.get("dim0",0)
        params["dim1"] = node.args[2] if len(node.args)>2 else node.kwargs.get("dim1",1)
    elif op in ("conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d"):
        a = node.args
        is_tp = op.startswith("conv_transpose")
        d = ndim_of(op[-2:])
        # 参数位置不同：
        #   conv*           -> (in, w, b, stride, padding, dilation, groups)
        #   conv_transpose* -> (in, w, b, stride, padding, output_padding, groups, dilation)
        stride = as_ints(a[3] if len(a)>3 else None)
        pad = a[4] if len(a)>4 else [0]*d
        if is_tp:
            outpad = as_ints(a[5] if len(a)>5 else None)
            groups = a[6] if len(a)>6 else 1
            dilation = as_ints(a[7] if len(a)>7 else None)
        else:
            outpad = None
            dilation = as_ints(a[5] if len(a)>5 else None)
            groups = a[6] if len(a)>6 else 1
        stride = stride or [1]*d
        dilation = dilation or [1]*d
        outpad = outpad or [0]*d
        # padding 可能是字符串 'same'/'valid'（ncnn 侧 4=-233 表示 same），必须原样透传，
        # 退化成 [0]*d 会丢语义（写成 4=0 valid），还会多出 14= 等 padding_h/w 参数导致匹配失败。
        pad_l = pad if isinstance(pad, str) else (list(pad) if isinstance(pad, (list, tuple)) else ([pad] * d if isinstance(pad, int) else [0] * d))
        outpad = (outpad if isinstance(outpad, str) else outpad) if is_tp else None
        # 形态判别：权重是注册参数 -> 模块形态 nn.Conv*d（内联 @attr）；
        #           权重是用户输入 -> 函数形态 F.conv*d（权重作 operand）
        if is_weight(a[1]):
            pt = ("nn.ConvTranspose%dd" % d) if is_tp else ("nn.Conv%dd" % d)
            inputs = [a[0].name]
            w = add_weight(a[1])
            if w: attrs["weight"] = w
            has_bias = (len(a) > 2 and a[2] is not None)
            if has_bias:
                b = add_weight(a[2])
                if b: attrs["bias"] = b
            ws = list(sd[param_sd[a[1].name]].shape)
            if is_tp:
                # 注意：nn.ConvTranspose*d 的规范形态**没有 padding_mode**（nn.Conv*d 才有），多设会匹配失败
                params.update(bias=has_bias, dilation=dilation, groups=groups,
                              in_channels=ws[0], kernel_size=ws[2:], out_channels=ws[1]*groups,
                              output_padding=outpad, padding=pad_l, stride=stride)
            else:
                params.update(bias=has_bias, dilation=dilation, groups=groups,
                              in_channels=ws[1]*groups, kernel_size=ws[2:], out_channels=ws[0],
                              padding=pad_l, padding_mode="zeros", stride=stride)
        else:
            pt = "F." + op
            inputs = [a[0].name, a[1].name]
            # 注意：函数形态的匹配图对参数集是精确的——
            #   无 bias（2 输入）那支带 `bias=None`；有 bias（3 输入）那支**不带** bias 参数。
            #   两种情况都设 bias 会让带 bias 那支多出参数而匹配失败（同 divisor_override 教训）。
            if len(a) > 2 and a[2] is not None:
                inputs.append(a[2].name)
                params.update(dilation=dilation, groups=groups, padding=pad_l, stride=stride)
            else:
                params.update(bias=None, dilation=dilation, groups=groups, padding=pad_l, stride=stride)
            if is_tp:
                params["output_padding"] = outpad
    elif op == "linear":
        pt = "nn.Linear"; a = node.args; inputs = [a[0].name]
        w = add_weight(a[1])
        if w: attrs["weight"] = w
        has_bias = (len(a)>2 and a[2] is not None)
        if has_bias:
            b = add_weight(a[2])
            if b: attrs["bias"] = b
        ws = list(sd[param_sd[a[1].name]].shape)
        params.update(bias=has_bias, in_features=ws[1], out_features=ws[0])
    elif op == "batch_norm":
        # 按输入 rank 选 BatchNorm1d/2d/3d：(N,C)/(N,C,L)->1d, (N,C,H,W)->2d, (N,C,D,H,W)->3d
        a = node.args; inputs = [a[0].name]
        ishp, _ = tm(a[0])
        rank = len(ishp) if ishp else 4
        pt = "nn.BatchNorm1d" if rank <= 3 else ("nn.BatchNorm3d" if rank >= 5 else "nn.BatchNorm2d")
        for nm,idx in (("weight",1),("bias",2),("running_mean",3),("running_var",4)):
            if len(a)>idx and a[idx] is not None:
                w = add_weight(a[idx])
                if w: attrs[nm] = w
        eps = a[7] if len(a)>7 else 1e-5
        # num_features：优先 weight，否则 running_mean/running_var，否则输入 shape[1]
        # （affine=False 时 weight 为 None，直接取 a[1].name 会 AttributeError）
        nf = None
        for idx in (1, 3, 4):
            if len(a)>idx and a[idx] is not None and getattr(a[idx], "name", None) in param_sd \
               and param_sd[a[idx].name] in sd:
                nf = list(sd[param_sd[a[idx].name]].shape)[0]
                break
        if nf is None and ishp and len(ishp) >= 2:
            nf = ishp[1]
        if nf is None:
            nf = 1
        params.update(affine=(len(a)>1 and a[1] is not None), eps=eps, num_features=nf)
    elif op == "layer_norm":
        pt = "nn.LayerNorm"; a = node.args; inputs = [a[0].name]
        nshape = list(a[1]) if len(a)>1 and isinstance(a[1],(list,tuple)) else []
        if len(a)>2 and a[2] is not None:
            w = add_weight(a[2]); 
            if w: attrs["weight"] = w
        if len(a)>3 and a[3] is not None:
            b = add_weight(a[3])
            if b: attrs["bias"] = b
        eps = a[4] if len(a)>4 else 1e-5
        params.update(elementwise_affine=(len(a)>2 and a[2] is not None), eps=eps, normalized_shape=nshape)
    elif op == "clone":
        # pnnx IR 保留 torch.clone 为离散算子（不消除，与 dropout 不同）
        pt = "torch.clone"; inputs = [x.name for x in node.args if is_tensor(x)]
    elif op in ("dropout","alpha_dropout","feature_dropout","feature_alpha_dropout"):
        # eval 模式下恒等，pnnx IR 会完全消除该算子 -> 产出别名而非节点
        ins = [x.name for x in node.args if is_tensor(x)]
        return ("ALIAS", ins[0])
    elif "getitem" in op:
        # operator.getitem：消费多输出节点（aten 返回 tuple，如 max_pool*_with_indices /
        # adaptive_max_pool*）。torch.export **不剪**无消费者的 getitem(indices)——
        # 它们留在图里但 users 为空（实测 test_F_adaptive_max_pool2d 的
        # getitem_1/3/5/7/9）。分派：idx=0 值输出消除为别名；无消费者的 indices 丢弃
        # （ts 侧对应 1→2 节点的死端输出，pass_ncnn 前同样被剪）；有消费者的 indices
        # 才需要真正的多输出 emit（待办，显式失败好过静默错接）。
        src = node.args[0] if node.args else None
        idx = node.args[1] if len(node.args) > 1 else node.kwargs.get("index")
        if not isinstance(src, torch.fx.Node):
            return None
        if idx == 0 or idx is None:
            return ("ALIAS", src.name)
        if not getattr(node, "users", None):
            return ("DROP",)
        return None
    elif op in ("max_pool1d","max_pool2d","max_pool3d"):
        d = ndim_of(op[-2:]); a = node.args
        inputs = [a[0].name]
        # nn.MaxPool*d 模块调用 vs F.max_pool*d 函数调用：ts canonical 保留模块形态
        # nn.MaxPool*d（sw_test_nn_MaxPool2d_ts 实证），且 pass_ncnn 的
        # insert_reshape_pooling 只对 nn.MaxPool*d（输入 rank 少 1 时）插 3D↔4D Reshape 对——
        # 模块形态误发 F.max_pool*d 会少这批 Reshape 导致 DIFF。
        mod = nn_module_of(node)
        pt = ("nn.MaxPool%dd" % d) if mod in ("MaxPool1d","MaxPool2d","MaxPool3d") else ("F." + op)
        params["kernel_size"] = as_ints(a[1] if len(a)>1 else None, d)
        params["stride"] = as_ints(a[2] if len(a)>2 else None, d)
        params["padding"] = as_ints(a[3] if len(a)>3 else [0]*d, d)
        params["dilation"] = as_ints(a[4] if len(a)>4 else [1]*d, d)
        params["ceil_mode"] = as_bool(a[5] if len(a)>5 else False)
        params["return_indices"] = False
    elif op in ("max_pool1d_with_indices","max_pool2d_with_indices","max_pool3d_with_indices"):
        # nn.MaxPool*d(return_indices=True) 模块调用分解而来（F.max_pool*d 的
        # return_indices=True 用例在 ncnn 测试里全部被注释）。indices 解包后未被消费
        # （现存测试全部如此，getitem(.,1) 被 DCE 剪掉）→ ts 规范形态为单输出
        # nn.MaxPool*d 1→1 return_indices=False（sw_test_nn_MaxPool2d_ts pool_5 实证）。
        # 注意 op[-2:] 对 "max_pool3d_with_indices" 取到 "es"，必须先剥后缀
        d = ndim_of(op.replace("_with_indices", "")[-2:]); a = node.args
        inputs = [a[0].name]
        mod = nn_module_of(node)
        pt = ("nn.MaxPool%dd" % d) if mod in ("MaxPool1d","MaxPool2d","MaxPool3d") else ("F.max_pool%dd" % d)
        params["kernel_size"] = as_ints(a[1] if len(a)>1 else None, d)
        params["stride"] = as_ints(a[2] if len(a)>2 else None, d)
        params["padding"] = as_ints(a[3] if len(a)>3 else [0]*d, d)
        params["dilation"] = as_ints(a[4] if len(a)>4 else [1]*d, d)
        params["ceil_mode"] = as_bool(a[5] if len(a)>5 else False)
        params["return_indices"] = False
    elif op in ("avg_pool1d","avg_pool2d","avg_pool3d"):
        d = ndim_of(op[-2:]); pt = "F." + op; a = node.args
        inputs = [a[0].name]
        params["kernel_size"] = as_ints(a[1] if len(a)>1 else None, d)
        params["stride"] = as_ints(a[2] if len(a)>2 else None, d)
        params["padding"] = as_ints(a[3] if len(a)>3 else [0]*d, d)
        params["ceil_mode"] = as_bool(a[4] if len(a)>4 else False)
        params["count_include_pad"] = as_bool(a[5] if len(a)>5 else True, default=True)
        # 仅 2d/3d 有 divisor_override；1d 的规范形态没有该参数，多设会导致转换器匹配失败
        if d >= 2:
            params["divisor_override"] = None
    elif op in ("adaptive_avg_pool1d","adaptive_avg_pool2d","adaptive_avg_pool3d"):
        d = ndim_of(op[-2:]); a = node.args
        inputs = [a[0].name]
        mod = nn_module_of(node)
        # nn.AdaptiveAvgPool*d 模块形态 vs F.adaptive_avg_pool*d 函数形态（ncnn op 名不同：aap vs aap1d/...）
        pt = ("nn." + mod) if mod in ("AdaptiveAvgPool1d","AdaptiveAvgPool2d","AdaptiveAvgPool3d") else ("F." + op)
        # torch.export 把 output_size 的 None 实例化成输入维尺寸（ts 保留 0 → pass_ncnn 写 -233），
        # output_size_canonical 还原 0 对齐（DIFF 病根：18=24/64 具体值 vs 18=-233 哨兵）
        params["output_size"] = output_size_canonical(a[1] if len(a)>1 else None, a[0], d)
    elif op in ("adaptive_max_pool1d","adaptive_max_pool2d","adaptive_max_pool3d"):
        # 与 adaptive_avg_pool 同构，但 aten 返回 (values, indices) tuple：未消费的
        # indices 输出会被 torch.export 的 DCE 剪掉（全部现存测试如此），ts 规范形态
        # 为单输出 return_indices=False（1→1），pass_ncnn 的 F_adaptive_max_pool*d /
        # nn_AdaptiveMaxPool*d 两支模式（output_size + return_indices）均按此匹配。
        d = ndim_of(op[-2:]); a = node.args
        inputs = [a[0].name]
        mod = nn_module_of(node)
        pt = ("nn." + mod) if mod in ("AdaptiveMaxPool1d","AdaptiveMaxPool2d","AdaptiveMaxPool3d") else ("F." + op)
        params["output_size"] = output_size_canonical(a[1] if len(a)>1 else None, a[0], d)
        params["return_indices"] = False
    elif op in ("abs","acos","acosh","asin","asinh","atan","atanh","ceil","cos","cosh",
                "erf","exp","expm1","floor","log","log10","log1p","neg","reciprocal",
                "round","rsqrt","sign","sin","sinh","sqrt","square","tan","trunc"):
        # pnnx IR 里一元逐元素算子是 **pnnx.Expression**（不是 F.<op>/torch.<op>），
        # 形如 `pnnx.Expression <n> 1 1 in out expr=exp(@0)`；@0 指第一个输入 operand。
        # 链式（如 torch.abs(x-0.5)）会被 torchscript 侧融合成 expr=abs(sub(@0,0.5))，
        # 这里先发单算子形态，链式融合交给后续 pass / 下一批。
        pt = "pnnx.Expression"
        inputs = [a.name for a in node.args if is_tensor(a)][:1]
        params["expr"] = "%s(@0)" % op
    elif op in ("upsample_nearest1d", "upsample_nearest2d"):
        # 规范形态：`F.upsample_nearest <n> 1 1 in out size=(..)` 或 `scale_factor=(..)`
        # size 与 scale_factor **互斥**——给了哪个就只写哪个，两个都写会匹配失败。
        a = node.args
        pt = "F.upsample_nearest"
        inputs = [a[0].name]
        sz = as_ints(a[1] if len(a) > 1 else None)
        sc = a[2] if len(a) > 2 else None
        if sz:
            params["size"] = sz
        elif sc is not None:
            params["scale_factor"] = [float(x) for x in sc] if isinstance(sc, (list, tuple)) else [float(sc)]
    elif op == "upsample_bilinear2d":
        # 规范形态：`F.upsample <n> 1 1 in out align_corners=True mode=bilinear size=(..)`
        # 注意 bilinear 走统一的 F.upsample + mode 参数（不是 F.upsample_bilinear）。
        # aten: upsample_bilinear2d(input, output_size, align_corners, scale_factors)
        a = node.args
        pt = "F.upsample"
        inputs = [a[0].name]
        params["mode"] = "bilinear"
        params["align_corners"] = bool(a[2]) if (len(a) > 2 and a[2] is not None) else False
        sz = as_ints(a[1] if len(a) > 1 else None)
        sc = a[3] if len(a) > 3 else None
        if sz:
            params["size"] = sz
        elif sc is not None:
            params["scale_factor"] = [float(x) for x in sc] if isinstance(sc, (list, tuple)) else [float(sc)]
    else:
        return None

    return {"name": "pt2_" + out_name, "pnnx_type": pt, "inputs": inputs,
            "outputs": [out_name], "output_shapes": out_shapes,
            "params": params, "attrs": attrs}

inputs_json = []
for spec in sig.input_specs:
    k = spec.kind; kn = k.name if hasattr(k,"name") else str(k)
    if kn in ("USER_INPUT",1):
        name = spec.arg.name
        ph = next((n for n in graph.nodes if n.name == name), None)
        s, dt = tm(ph) if ph else (None,7)
        inputs_json.append({"name": name, "shape": s, "dtype": dt})

outputs_json = []
for spec in sig.output_specs:
    name = spec.arg.name
    ph = next((n for n in graph.nodes if n.name == name), None)
    s, dt = tm(ph) if ph else (None,7)
    outputs_json.append({"name": name, "shape": s, "dtype": dt})

nodes_json = []
# 恒等算子（dropout 族）在 pnnx IR 里被消除，其输出名需别名到输入名
alias = {}
def resolve(nm):
    seen = set()
    while nm in alias and nm not in seen:
        seen.add(nm)
        nm = alias[nm]
    return nm

for node in graph.nodes:
    if node.op != "call_function": continue
    rec = emit_node(node)
    if rec is None:
        die("unsupported aten op: %s" % node.target)
    if isinstance(rec, tuple) and rec[0] == "ALIAS":
        alias[node.name] = resolve(rec[1])
        continue
    if isinstance(rec, tuple) and rec[0] == "DROP":
        # 死端节点（无消费者的多输出残余，如未被消费的 indices）：不建节点也不进
        # alias——若日后有东西引用它的名字，get_operand 会显式失败而非静默错接。
        continue
    rec["inputs"] = [resolve(x) for x in rec["inputs"]]
    nodes_json.append(rec)

# 输出名也可能落在被消除的恒等节点上
for o in outputs_json:
    o["name"] = resolve(o["name"])

model_json = {"inputs": inputs_json, "outputs": outputs_json, "nodes": nodes_json}

try:
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("models/model.json", json.dumps(model_json))
        for name, data in wentries.items():
            zf.writestr(name, data)
except Exception as e:
    die("failed to write intermediate zip: %s" % str(e))

sys.stderr.write("pt2_preprocess: wrote %s (%d nodes, %d weights)\n" % (out_path, len(nodes_json), len(wentries)))
)PT2PYEOF";

static int run_pt2_preprocess(const std::string& in_path, const std::string& out_path)
{
    const char* py = getenv("PNNX_PYTHON");
    if (!py || py[0] == '\0')
        py = "python3";

#if defined _WIN32
    _putenv_s("PNNX_PT2_IN", in_path.c_str());
    _putenv_s("PNNX_PT2_OUT", out_path.c_str());
#else
    setenv("PNNX_PT2_IN", in_path.c_str(), 1);
    setenv("PNNX_PT2_OUT", out_path.c_str(), 1);
#endif

    std::string cmd = std::string(py) + " -";
    FILE* pp = PNNX_POPEN(cmd.c_str(), "w");
    if (!pp)
    {
        fprintf(stderr, "load_pt2: failed to launch python (%s)\n", py);
        return -1;
    }
    fputs(kPt2PreprocessPy, pp);
    int st = PNNX_PCLOSE(pp);
    if (st != 0)
    {
        fprintf(stderr, "load_pt2: pt2 preprocess python failed (exit %d).\n"
                        "        Ensure PNNX_PYTHON points to a python with torch installed\n"
                        "        (e.g. the ncnn-env venv), or activate that venv before running pnnx.\n", st);
        return -1;
    }
    return 0;
}

// 真实 .pt2 的 zip 内含 archive/data...（pickle 的 ExportedProgram 入口），据此与 torchscript/中间格式区分。
static bool is_real_pt2(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    fseek(fp, 0, SEEK_END);
    long fs = ftell(fp);
    long scan = fs;
    if (scan > 8 * 1024 * 1024)
        scan = 8 * 1024 * 1024;
    if (scan <= 0)
    {
        fclose(fp);
        return false;
    }
    std::vector<char> buf((size_t)scan);
    fseek(fp, fs - scan, SEEK_SET);
    size_t rd = fread(buf.data(), 1, (size_t)scan, fp);
    fclose(fp);
    const char* mk = "archive/data";
    size_t mklen = strlen(mk);
    if (mklen > rd)
        return false;
    for (size_t i = 0; i + mklen <= rd; i++)
    {
        if (memcmp(buf.data() + i, mk, mklen) == 0)
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// 公开入口
// ---------------------------------------------------------------------------
int load_pt2(const std::string& pt2path, Graph& g,
             const std::string& /*device*/,
             const std::vector<std::vector<int64_t> >& /*input_shapes*/,   // 形状已由 Python 从 tensor_meta 写入 JSON，此处不再需要
             const std::vector<std::string>& /*input_types*/,
             const std::vector<std::vector<char> >& /*input_contents*/,
             const std::vector<std::vector<int64_t> >& /*input_shapes2*/,
             const std::vector<std::string>& /*input_types2*/,
             const std::vector<std::vector<char> >& /*input_contents2*/,
             const std::vector<std::string>& /*customop_modules*/,
             const std::vector<std::string>& /*module_operators*/,
             const std::string& /*foldable_constants_zippath*/,
             std::set<std::string>& /*foldable_constants*/)
{
    std::string to_parse = pt2path;
    std::string tmp_path;

    if (is_real_pt2(pt2path))
    {
        tmp_path = pt2path + ".pnnx_intermediate.zip";
        if (run_pt2_preprocess(pt2path, tmp_path) != 0)
            return -1;
        to_parse = tmp_path;
    }

    Pt2Graph pg;
    if (parse_pt2_zip(to_parse, pg) != 0)
    {
        if (!tmp_path.empty())
            remove(tmp_path.c_str());
        return -1;
    }
    if (emit_pt2_graph(pg, g) != 0)
    {
        if (!tmp_path.empty())
            remove(tmp_path.c_str());
        return -1;
    }
    if (!tmp_path.empty())
        remove(tmp_path.c_str());
    return 0;
}

} // namespace pnnx
