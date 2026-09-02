// Tencent 2026
// SPDX-License-Identifier: BSD-3-Clause

// N4 整改回归样例(独立 harness,零 libtorch)。
// 编译(在 tests/ 目录下)。注意 test_pt2_regress.cpp 必须放在链接行最后:
// 它 include 的 F_pt2.cpp 含 pass 注册对象,需晚于 pass_level2.cpp 里的
// 全局注册表 map 构造执行,否则静态初始化顺序未定导致启动段错误。
//     g++ -O2 -std=c++11 -Wall -Wextra -I../src ../src/pass_level2.cpp ../src/ir.cpp ../src/utils.cpp ../src/storezip.cpp ../src/pt2_schema.cpp ../src/model_stat.cpp ../src/pass_level2/functionize.cpp ../src/pass_level2/eliminate_contiguous.cpp ../src/pass_level2/eliminate_size_numtotensor_int.cpp ../src/pass_level2/fuse_constantlist.cpp test_pt2_regress.cpp -o test_pt2_regress
//
// 覆盖 docs/15 P1.3 / P2.2:
// 1) F_pt2_fold_ones_like:合法形态(f32 + 静态正 shape + 标量 other/alpha)
//    折叠出带 data 的 pnnx.Attribute;非 f32 / 空 shape / 非正维 / 非标量
//    other 一律保持原图(match 拒绝,write 不再静默留空属性)。
// 2) load_pt2 argument_to_constant:DEVICE 保留 "type:index" 编码,cuda:1
//    不降级为 cuda;无 index 为裸 "type";空 device 编 None。
//
// 白盒说明:#include 产品 .cpp 以访问 static 函数(argument_to_constant)
// 与文件内 pass 类(F_pt2_fold_ones_like);两者均零 libtorch 依赖。

#include "load_pt2.cpp"
#include "pass_level2/F_pt2.cpp"

#include <stdio.h>
#include <string.h>

#include <map>
#include <string>

using namespace pnnx;

static int g_failed = 0;

#define CHECK(cond, msg)                \
    do                                  \
    {                                   \
        if (cond)                       \
            printf("ok   %s\n", (msg)); \
        else                            \
        {                               \
            printf("FAIL %s\n", (msg)); \
            g_failed++;                 \
        }                               \
    } while (0)

static Operator* find_op(Graph& g, const char* type)
{
    for (size_t i = 0; i < g.ops.size(); i++)
    {
        if (g.ops[i]->type == type)
            return g.ops[i];
    }
    return 0;
}

static void build_ones_like_graph(Graph& g, bool string_other)
{
    // clang-format off
    g.parse(R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
torch.ones_like         op_0        1 1 input ones_out dtype=0
prim::Constant          op_c        0 1 other value=0.5
prim::Constant          op_a        0 1 alpha value=1
aten::add               op_1        3 1 ones_out other alpha out
pnnx.Output             output      1 0 out
)PNNXIR");
        // clang-format on

        if (string_other)
    {
        // other 为字符串常量 → 非标量形态,必须整体不匹配
        for (size_t i = 0; i < g.ops.size(); i++)
        {
            if (g.ops[i]->name == "op_c")
                g.ops[i]->params["value"] = Parameter("abc");
        }
    }
}

static void run_ones_like_pass(Graph& g)
{
    F_pt2_fold_ones_like pass;
    int opindex = 0;
    pnnx_graph_rewrite(g, &pass, opindex);
}

static void test_ones_like_fold()
{
    // 合法形态:f32 静态正 shape → 折叠,attr 带完整 data
    {
        Graph g;
        build_ones_like_graph(g, false);
        Operator* add = find_op(g, "aten::add");
        add->outputs[0]->type = 1; // f32
        add->outputs[0]->shape.push_back(2);
        add->outputs[0]->shape.push_back(3);

        run_ones_like_pass(g);

        const Operator* fold = find_op(g, "pnnx.Attribute");
        CHECK(fold != 0, "ones_like: f32 static shape folded to pnnx.Attribute");
        CHECK(find_op(g, "torch.ones_like") == 0 && find_op(g, "aten::add") == 0,
              "ones_like: matched subgraph consumed");
        if (fold != 0)
        {
            std::map<std::string, Attribute>::const_iterator it = fold->attrs.find("data");
            CHECK(it != fold->attrs.end() && it->second.data.size() == 6 * sizeof(float),
                  "ones_like: attr data = 6 floats");
            CHECK(it != fold->attrs.end() && it->second.shape.size() == 2 && it->second.shape[0] == 2
                      && it->second.shape[1] == 3,
                  "ones_like: attr shape = (2,3)");
            float v0 = 0.f;
            if (it != fold->attrs.end() && it->second.data.size() >= 4)
                memcpy(&v0, it->second.data.data(), 4);
            CHECK(v0 == 1.5f, "ones_like: folded value = 1+1*0.5 = 1.5");
        }
    }

    // 拒绝形态:每种都保持原图,不产生 pnnx.Attribute
    {
        Graph g;
        build_ones_like_graph(g, false);
        find_op(g, "aten::add")->outputs[0]->type = 5; // i64
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: non-f32 output keeps original graph");
    }
    {
        Graph g;
        build_ones_like_graph(g, false);
        find_op(g, "aten::add")->outputs[0]->type = 1; // shape 缺失(动态/未知)
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: missing shape keeps original graph");
    }
    {
        Graph g;
        build_ones_like_graph(g, false);
        Operator* add = find_op(g, "aten::add");
        add->outputs[0]->type = 1;
        add->outputs[0]->shape.push_back(0); // 非正维
        add->outputs[0]->shape.push_back(3);
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: non-positive dim keeps original graph");
    }
    {
        Graph g;
        build_ones_like_graph(g, true); // other 为字符串,非标量
        Operator* add = find_op(g, "aten::add");
        add->outputs[0]->type = 1;
        add->outputs[0]->shape.push_back(2);
        add->outputs[0]->shape.push_back(3);
        run_ones_like_pass(g);
        CHECK(find_op(g, "pnnx.Attribute") == 0 && find_op(g, "aten::add") != 0 && g.ops.size() == 6,
              "ones_like: non-scalar other keeps original graph");
    }
}

static void test_device_argument()
{
    Parameter v;
    Pt2Argument a;
    a.type = Pt2Argument::DEVICE;

    // cuda:1 → "cuda:1"(不得降级为 cuda)
    a.device_type = "cuda";
    a.device_index = 1;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda:1",
          "device: cuda:1 encoded as cuda:1");

    // cuda:0 → "cuda:0"(与无 index 可区分)
    a.device_index = 0;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda:0",
          "device: cuda:0 encoded as cuda:0");

    // 无 index(-1)→ 裸 "cuda"
    a.device_index = -1;
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cuda",
          "device: cuda with null index encoded as cuda");

    // cpu 无 index → "cpu"
    a.device_type = "cpu";
    CHECK(argument_to_constant(a, v) && v.type == 4 && v.s == "cpu",
          "device: cpu with null index encoded as cpu");

    // 空 device_type → None
    a.device_type = "";
    CHECK(argument_to_constant(a, v) && v.type == 0,
          "device: empty device encoded as None");
}

int main()
{
    test_ones_like_fold();
    test_device_argument();

    if (g_failed == 0)
    {
        printf("RESULT: all pass\n");
        return 0;
    }
    printf("RESULT: %d failed\n", g_failed);
    return 1;
}
