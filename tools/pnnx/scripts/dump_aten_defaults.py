#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线生成 aten 参数默认值静态表(dump_aten_defaults.py)

遍历 torch._C._jit_get_all_schemas(),把 aten 算子形参的默认值 dump 成
C++ 静态表(pnnx/src/aten_defaults_table.h)。pnnx 的 pt2 loader 据此补全
torch.export 省略的实参(等于默认值的参数会被 torch.export 从图里抹掉,
如 cat 的 dim=0、flatten 的 end_dim=-1)。

这是 pnnx pt2 前端的差异化设计:默认值知识在离线阶段固化、进仓库可审计、
可随时重跑再生成,loader 侧零 torch 运行时依赖。

用法(conda NCNN 环境,需 torch,但不需要 gpu/libtorch):
    # 只收录真实 .pt2 图里出现过的算子(推荐,按需重跑扩充)
    python scripts/dump_aten_defaults.py --scan ../tests/ncnn --out src/aten_defaults_table.h
    # 显式追加算子(全名,overload 可省略,缺省取 .default)
    python scripts/dump_aten_defaults.py --ops aten::cat aten::flatten.using_ints ...
    # 全量 aten(表很大,仅兜底用)
    python scripts/dump_aten_defaults.py --all

值编码(与 src/aten_defaults_table.h 头注释一致):
    NO_DEFAULT=-1  无默认值(占位行,保证形参顺序可用)
    NONE=0         ""
    INT=1          十进制整数(Scalar 默认值按 IValue 实际类型归 int/float)
    FLOAT=2        strtod 可解析(repr 输出,含 inf/-inf/nan)
    BOOL=3         "0"/"1"
    STRING=4       原文(C 字符串转义后)
    INTS=5         逗号分隔
    FLOATS=6       逗号分隔
    STRINGS=7      逗号分隔(元素含逗号的 op 拒绝,防歧义)
    DEVICE=8       ""=None,否则 "cpu"/"cuda:0" 形态
    UNSUPPORTED=9  bool 列表/嵌套列表/Tensor 等 builder 无法表达的默认值,
                   该参数不参与补全(告警留痕,不影响同 op 其他参数)
"""

import argparse
import datetime
import json
import sys
import zipfile
from pathlib import Path

import torch

# type 标签常量(必须与 C++ 枚举 Pt2DefaultType 一致)
T_NO_DEFAULT = -1
T_NONE = 0
T_INT = 1
T_FLOAT = 2
T_BOOL = 3
T_STRING = 4
T_INTS = 5
T_FLOATS = 6
T_STRINGS = 7
T_DEVICE = 8
T_UNSUPPORTED = 9

TYPE_NAMES = {
    T_NO_DEFAULT: "NO_DEFAULT",
    T_NONE: "NONE",
    T_INT: "INT",
    T_FLOAT: "FLOAT",
    T_BOOL: "BOOL",
    T_STRING: "STRING",
    T_INTS: "INTS",
    T_FLOATS: "FLOATS",
    T_STRINGS: "STRINGS",
    T_DEVICE: "DEVICE",
    T_UNSUPPORTED: "UNSUPPORTED",
}


class UnsupportedDefault(Exception):
    """单个默认值无法用静态表编码(不致命,该参数降级为 UNSUPPORTED)"""


def _float_repr(v):
    # repr 保真且 strtod 可解析:1.0 / 0.5 / 1e-07 / inf / nan
    return repr(float(v))


def encode_default(dv):
    """IValue 默认值 → (type 标签, 字符串值)。不支持时抛 UnsupportedDefault。"""
    if dv is None:
        return T_NONE, ""
    # bool 必须先于 int 判(python bool 是 int 子类)
    if isinstance(dv, bool):
        return T_BOOL, "1" if dv else "0"
    if isinstance(dv, int):
        return T_INT, str(dv)
    if isinstance(dv, float):
        return T_FLOAT, _float_repr(dv)
    if isinstance(dv, str):
        return T_STRING, dv
    if isinstance(dv, torch.device):
        # "" 表示 None;device(type=cpu)按 str 输出
        return T_DEVICE, str(dv) if dv.type else ""
    if isinstance(dv, list):
        if not dv:
            # 空列表:元素类型未知,按 int 空列表编码(builder 产出 (0,) 形态?不,产出空 ai)
            return T_INTS, ""
        if all(isinstance(x, bool) for x in dv):
            raise UnsupportedDefault("bool list(pnnx Parameter 无 bool 列表表达)")
        if all(isinstance(x, int) for x in dv):
            return T_INTS, ",".join(str(x) for x in dv)
        if all(isinstance(x, float) for x in dv):
            return T_FLOATS, ",".join(_float_repr(x) for x in dv)
        if all(isinstance(x, str) for x in dv):
            if any("," in x for x in dv):
                raise UnsupportedDefault("string list 元素含逗号,平铺编码有歧义")
            return T_STRINGS, ",".join(dv)
        raise UnsupportedDefault(f"列表元素类型混合或张量: {dv!r}")
    raise UnsupportedDefault(f"类型 {type(dv).__name__}: {dv!r}")


def collect_ops_from_pt2(path):
    """从单个 .pt2 的 models/model.json 提取 torch.ops.* target 全名。

    纯 zipfile + json,不需要反序列化 torch 对象。
    """
    ops = set()
    with zipfile.ZipFile(path) as zf:
        entries = [n for n in zf.namelist() if n.endswith("models/model.json")]
        if not entries:
            print(f"  SKIP(非 pt2): {path}", file=sys.stderr)
            return ops
        data = json.loads(zf.read(entries[0]))
    for node in data.get("graph_module", {}).get("graph", {}).get("nodes", []):
        target = node.get("target", "")
        if not target.startswith("torch.ops."):
            continue
        # "torch.ops.aten.conv2d.default" → "aten::conv2d.default"
        rest = target[len("torch.ops."):]
        ns, sep, tail = rest.partition(".")
        if not sep:
            continue
        ops.add(ns + "::" + tail)
    return ops


def scan_dirs(dirs):
    ops = set()
    for d in dirs:
        pt2s = sorted(Path(d).rglob("*.pt2"))
        print(f"scan {d}: {len(pt2s)} 个 .pt2", file=sys.stderr)
        for p in pt2s:
            try:
                ops |= collect_ops_from_pt2(p)
            except Exception as e:  # noqa: BLE001 单文件损坏不拖垮全量
                print(f"  WARN 读取失败 {p}: {e}", file=sys.stderr)
    return ops


def build_schema_index():
    """registry 全量 schema 按 pt2 全名(aten::op.overload)建索引。

    注意:default overload 的 schema.overload_name 是空串,而 pt2 target
    用 ".default" —— 这里统一映射成 pt2 全名,builder 侧免转换。
    """
    index = {}
    for s in torch._C._jit_get_all_schemas():
        overload = s.overload_name or "default"
        index[f"{s.name}.{overload}"] = s
    return index


def resolve_op_names(requested, schema_index):
    """补全省略的 overload(aten::cat → aten::cat.default),返回 (全名, 报错) 列表。"""
    resolved = []
    for name in requested:
        name = name.strip()
        if not name:
            continue
        if name in schema_index:
            resolved.append(name)
            continue
        # overload 省略:收集所有同名候选
        cands = [k for k in schema_index if k.startswith(name + ".")]
        if not cands:
            print(f"WARN schema 未收录: {name}", file=sys.stderr)
            continue
        if f"{name}.default" in cands:
            resolved.append(f"{name}.default")
        else:
            print(f"WARN {name} 无 .default overload,候选 {cands} 未收录(请用全名显式指定)",
                  file=sys.stderr)
    return resolved


def escape_cpp_string(s):
    """C 字符串字面量转义(aten 字符串默认值均为 ASCII,防患于未然)。"""
    out = []
    for ch in s:
        if ch in ('\\', '"'):
            out.append("\\" + ch)
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("\\%03o" % ord(ch))  # 3 位八进制自终止,不会被后续数字吞并
    return "".join(out)


def mangle_op_name(full_name):
    return full_name.replace("::", "_").replace(".", "_")


def dump_op(full_name, schema):
    """schema → (行列表 [(形参名, type, 编码值)], 跳过原因/None)"""
    rows = []
    for arg in schema.arguments:
        if arg.has_default_value():
            try:
                t, v = encode_default(arg.default_value)
            except UnsupportedDefault as e:
                print(f"  WARN {full_name}.{arg.name}: 默认值不可编码({e}),降级 UNSUPPORTED",
                      file=sys.stderr)
                t, v = T_UNSUPPORTED, ""
        else:
            t, v = T_NO_DEFAULT, ""
        rows.append((arg.name, t, v))
    return rows


HEADER_TEMPLATE = """\
// Copyright {year} Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// aten 参数默认值静态表(离线生成,勿手改)。
//
// torch.export 会把等于默认值的实参从图里省略(cat 的 dim=0、flatten 的
// end_dim=-1、conv2d 的 dilation/groups 等);pt2 builder 据本表把省略的
// 实参补全为完整 schema 形态,使 pt2 图与 torchscript 图同构、下游
// pass_level2 形态分支(torch_cat / torch_flatten / F_conv2d_1 ...)零改动复用。
//
// 再生成:{cmd}
// 来源:torch {torch_version} 的 torch._C._jit_get_all_schemas()({count_schemas} 个 schema)
// 生成时间:{date}
// 收录算子:{count_ops} 个
//
// 值编码(type 标签 + 字符串值):
//   NO_DEFAULT=-1  无默认值的必填参数(占位,保证形参顺序)
//   NONE=0         ""
//   INT=1          十进制整数
//   FLOAT=2        strtod 可解析(含 inf/-inf/nan)
//   BOOL=3         "0"/"1"
//   STRING=4       原文
//   INTS/FLOATS/STRINGS=5/6/7  逗号分隔平铺;值为 "" 表示空列表,builder 转
//                  type 0(None)—— ts 侧空列表实参物化为 None 常量(如
//                  max_pool2d 的 stride=()),下游转换器按 type 0 解释
//   DEVICE=8       ""=None,否则 "cpu"/"cuda:0" 形态(builder 转 STRING)
//   UNSUPPORTED=9  bool 列表/嵌套列表/Tensor 等 builder 无法表达的默认值,
//                  不参与补全(生成时告警留痕)
//
// 限制:覆盖随测试语料增长按需重跑扩充;表未收录的算子 builder 保持
// torch.export 原样转写(缺参不补,stderr 告警)。

#ifndef PNNX_ATEN_DEFAULTS_TABLE_H
#define PNNX_ATEN_DEFAULTS_TABLE_H

#include <stddef.h>
#include <string.h>

namespace pnnx {{

enum Pt2DefaultType
{{
    PT2_D_NO_DEFAULT = -1,
    PT2_D_NONE = 0,
    PT2_D_INT = 1,
    PT2_D_FLOAT = 2,
    PT2_D_BOOL = 3,
    PT2_D_STRING = 4,
    PT2_D_INTS = 5,
    PT2_D_FLOATS = 6,
    PT2_D_STRINGS = 7,
    PT2_D_DEVICE = 8,
    PT2_D_UNSUPPORTED = 9
}};

struct Pt2ArgDefault
{{
    const char* name;
    Pt2DefaultType type;
    const char* value;
}};

struct Pt2DefaultsEntry
{{
    const char* op; // 全名含 overload,如 "aten::conv2d.default"
    const Pt2ArgDefault* args;
    size_t arg_count;
}};

// 按 pt2 target 全名(如 "aten::flatten.using_ints")查参数默认值表。
// 未收录返回 0。
inline const Pt2DefaultsEntry* find_pt2_aten_defaults(const char* op)
{{
{entries}
    return 0;
}}

}} // namespace pnnx

#endif // PNNX_ATEN_DEFAULTS_TABLE_H
"""


def generate_header(op_rows, cmd, out_path):
    year = datetime.date.today().year
    count_schemas = len(torch._C._jit_get_all_schemas())

    entries_lines = []
    for full_name, rows in op_rows:
        mangled = mangle_op_name(full_name)
        entries_lines.append(f"    static const Pt2ArgDefault args_{mangled}[] = {{")
        for name, t, v in rows:
            entries_lines.append("        {\"%s\", PT2_D_%s, \"%s\"}," % (escape_cpp_string(name), TYPE_NAMES[t], escape_cpp_string(v)))
        entries_lines.append("    };")
        entries_lines.append(
            "    static const Pt2DefaultsEntry entry_%s = {\"%s\", args_%s, %d};"
            % (mangled, escape_cpp_string(full_name), mangled, len(rows))
        )
        entries_lines.append("    if (strcmp(op, entry_%s.op) == 0) return &entry_%s;" % (mangled, mangled))
        entries_lines.append("")

    header = HEADER_TEMPLATE.format(
        year=year,
        cmd=cmd,
        torch_version=torch.__version__,
        count_schemas=count_schemas,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        count_ops=len(op_rows),
        entries="\n".join(entries_lines).rstrip("\n"),
    )

    Path(out_path).write_text(header, encoding="utf-8", newline="\n")
    print(f"生成 {out_path}: {len(op_rows)} 个算子", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", nargs="*", default=[], help="递归扫描目录里的 *.pt2,收录其中出现过的 aten op")
    ap.add_argument("--ops", nargs="*", default=[], help="显式追加 op(全名或省略 overload)")
    ap.add_argument("--all", action="store_true", help="收录全部 aten schema(含所有默认值 op,表很大)")
    ap.add_argument("--out", default="src/aten_defaults_table.h", help="输出头文件路径")
    args = ap.parse_args()

    if not (args.scan or args.ops or args.all):
        ap.error("至少提供 --scan / --ops / --all 之一")

    schema_index = build_schema_index()

    requested = set(args.ops)
    if args.scan:
        scanned = scan_dirs(args.scan)
        print(f"scan 命中 {len(scanned)} 个 op", file=sys.stderr)
        requested |= scanned
    if args.all:
        requested |= {k for k in schema_index if k.startswith("aten::")}

    op_names = resolve_op_names(sorted(requested), schema_index)

    # 排序保证生成结果可复现(不随 registry 顺序漂移)
    op_names = sorted(set(op_names))

    op_rows = []
    for full in op_names:
        rows = dump_op(full, schema_index[full])
        op_rows.append((full, rows))

    cmd = "python scripts/dump_aten_defaults.py"
    if args.scan:
        cmd += " --scan " + " ".join(args.scan)
    if args.ops:
        cmd += " --ops " + " ".join(sorted(set(args.ops)))
    if args.all:
        cmd += " --all"

    generate_header(op_rows, cmd, args.out)


if __name__ == "__main__":
    main()
