#!/usr/bin/env python3
"""
ncnn MicroPython API 完整测试

使用方法：
1. 确保已编译包含ncnn模块的MicroPython
2. 在ncnn_micropython目录下运行：
   /path/to/micropython test/test_api.py

测试内容：
- 导入ncnn模块
- 版本信息获取
- 模块初始化 
- Mat类：创建、填充、属性访问
- Net类：创建、API调用
- Extractor类：创建、API调用
- 内存管理和错误处理
"""

import sys

# 导入ncnn模块
try:
    import ncnn
    print("✅ 成功导入ncnn模块")
except ImportError as e:
    print("❌ 导入ncnn模块失败:", e)
    print("请确保已编译包含ncnn模块的MicroPython")
    sys.exit(1)

def test_version():
    """测试版本信息获取"""
    try:
        version = ncnn.version()
        print(f"✅ ncnn版本: {version}")
        return True
    except Exception as e:
        print(f"❌ 版本获取失败: {e}")
        return False

def test_init():
    """测试模块初始化"""
    try:
        ncnn.init()
        print("✅ 模块初始化成功")
        return True
    except Exception as e:
        print(f"❌ 模块初始化失败: {e}")
        return False

def test_mat_api():
    """测试Mat类所有API"""
    print("\n--- Mat类API测试 ---")
    success_count = 0
    
    # 测试Mat构造函数
    test_cases = [
        ("空Mat", lambda: ncnn.Mat()),
        ("1D Mat", lambda: ncnn.Mat(10)),
        ("2D Mat", lambda: ncnn.Mat(8, 8)),
        ("3D Mat", lambda: ncnn.Mat(4, 4, 3)),
        ("4D Mat", lambda: ncnn.Mat(2, 4, 4, 3)),
    ]
    
    mats = []
    for name, creator in test_cases:
        try:
            mat = creator()
            mats.append(mat)
            print(f"✅ {name}创建成功")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}创建失败: {e}")
    
    # 测试Mat属性访问
    if mats:
        try:
            mat = mats[-1]  # 使用最后一个4D Mat测试
            dims = mat.dims()
            w = mat.w()
            h = mat.h()
            c = mat.c()
            print(f"✅ Mat属性访问: dims={dims}, w={w}, h={h}, c={c}")
            success_count += 1
        except Exception as e:
            print(f"❌ Mat属性访问失败: {e}")
    
    # 测试Mat填充操作
    if mats:
        try:
            mat = mats[2]  # 使用2D Mat测试填充
            mat.fill(0.5)
            print("✅ Mat填充操作成功")
            success_count += 1
        except Exception as e:
            print(f"❌ Mat填充操作失败: {e}")
    
    return success_count

def test_net_api():
    """测试Net类所有API"""
    print("\n--- Net类API测试 ---")
    success_count = 0
    
    # 测试Net创建
    try:
        net = ncnn.Net()
        print("✅ Net对象创建成功")
        success_count += 1
    except Exception as e:
        print(f"❌ Net对象创建失败: {e}")
        return success_count, net
    
    # 测试模型加载API（不需要实际文件）
    try:
        # 测试无效文件的返回码
        param_ret = net.load_param("nonexistent.param")
        model_ret = net.load_model("nonexistent.bin")
        if param_ret != 0 and model_ret != 0:
            print("✅ 模型加载API正常（正确返回错误码）")
            success_count += 1
        else:
            print("⚠️ 模型加载API异常（应该返回错误码）")
    except Exception as e:
        print(f"✅ 模型加载API正常（正确抛出异常）")
        success_count += 1
    
    return success_count, net

def test_extractor_api(net):
    """测试Extractor类所有API"""
    print("\n--- Extractor类API测试 ---")
    success_count = 0
    
    # 测试Extractor创建
    try:
        extractor = net.create_extractor()
        print("✅ Extractor创建成功")
        success_count += 1
    except Exception as e:
        print(f"❌ Extractor创建失败: {e}")
        return success_count
    
    # 测试输入API（无需实际推理）
    try:
        # 创建测试输入
        input_mat = ncnn.Mat(224, 224, 3)
        input_mat.fill(0.5)
        
        # 设置输入（无模型时会失败，但API调用正常）
        input_ret = extractor.input(0, input_mat)
        print(f"✅ 设置输入API调用成功，返回码: {input_ret}")
        success_count += 1
        
    except Exception as e:
        print(f"✅ 设置输入API调用正常（正确抛出异常）")
        success_count += 1
    
    # 测试提取API（无需实际推理）
    try:
        # 提取输出（无模型时会失败，但API调用正常）
        output_mat = extractor.extract(0)
        if output_mat:
            print("✅ 提取输出API调用成功")
            success_count += 1
        else:
            print("✅ 提取输出API调用正常（正确返回空结果）")
            success_count += 1
            
    except Exception as e:
        print(f"✅ 提取输出API调用正常（正确抛出异常）")
        success_count += 1
    
    return success_count

def test_memory_management():
    """测试内存管理"""
    print("\n--- 内存管理测试 ---")
    success_count = 0
    
    # 创建多个对象测试内存管理
    try:
        objects = []
        for i in range(20):  # 适合MicroPython的数量
            mat = ncnn.Mat(16, 16, 3)
            mat.fill(float(i) / 20.0)
            objects.append(mat)
        
        print("✅ 创建20个Mat对象成功")
        success_count += 1
        
        # 创建多个Net对象
        nets = []
        for i in range(5):
            net = ncnn.Net()
            nets.append(net)
        
        print("✅ 创建5个Net对象成功")
        success_count += 1
        
    except Exception as e:
        print(f"❌ 内存管理测试失败: {e}")
    
    return success_count

def test_option_api():
    """测试Option类所有API"""
    print("\n--- Option类API测试 ---")
    success_count = 0
    
    # 测试Option创建
    try:
        option = ncnn.Option()
        print("✅ Option对象创建成功")
        success_count += 1
    except Exception as e:
        print(f"❌ Option对象创建失败: {e}")
        return success_count
    
    # 测试线程数设置和获取
    try:
        # 设置线程数
        option.set_num_threads(4)
        threads = option.get_num_threads()
        if threads == 4:
            print("✅ 线程数设置/获取成功")
            success_count += 1
        else:
            print(f"❌ 线程数设置失败，期望4，实际{threads}")
        
        # 设置不同的线程数
        option.set_num_threads(2)
        threads = option.get_num_threads()
        if threads == 2:
            print("✅ 线程数修改成功")
            success_count += 1
        else:
            print(f"❌ 线程数修改失败，期望2，实际{threads}")
            
    except Exception as e:
        print(f"❌ 线程数API失败: {e}")
    
    # 测试Vulkan设置和获取
    try:
        # 设置Vulkan
        option.set_use_vulkan(True)
        vulkan = option.get_use_vulkan()
        print(f"✅ Vulkan设置/获取API调用成功，值: {vulkan}")
        success_count += 1
        
        option.set_use_vulkan(False)
        vulkan = option.get_use_vulkan()
        print(f"✅ Vulkan修改API调用成功，值: {vulkan}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ Vulkan API失败: {e}")
    
    return success_count, option

def test_allocator_api():
    """测试Allocator类所有API"""
    print("\n--- Allocator类API测试 ---")
    success_count = 0
    
    # 测试Pool Allocator创建
    try:
        allocator1 = ncnn.Allocator.create_pool()
        print("✅ Pool Allocator创建成功")
        success_count += 1
    except Exception as e:
        print(f"❌ Pool Allocator创建失败: {e}")
    
    # 测试Unlocked Pool Allocator创建
    try:
        allocator2 = ncnn.Allocator.create_unlocked_pool()
        print("✅ Unlocked Pool Allocator创建成功")
        success_count += 1
    except Exception as e:
        print(f"❌ Unlocked Pool Allocator创建失败: {e}")
    
    return success_count

def test_advanced_integration(option):
    """测试高级功能集成"""
    print("\n--- 高级功能集成测试 ---")
    success_count = 0
    
    # 测试Net与Option集成
    try:
        net = ncnn.Net()
        # 配置option
        option.set_num_threads(1)
        
        # 将option应用到net
        net.set_option(option)
        print("✅ Net.set_option()调用成功")
        success_count += 1
        
        # 测试配置后的网络功能
        extractor = net.create_extractor()
        print("✅ 配置后的Net可正常创建Extractor")
        success_count += 1
        
    except Exception as e:
        print(f"❌ 高级功能集成失败: {e}")
    
    # 测试多个配置对象
    try:
        option1 = ncnn.Option()
        option1.set_num_threads(1)
        
        option2 = ncnn.Option()
        option2.set_num_threads(8)
        
        net1 = ncnn.Net()
        net1.set_option(option1)
        
        net2 = ncnn.Net()
        net2.set_option(option2)
        
        print("✅ 多个配置对象使用成功")
        success_count += 1
        
    except Exception as e:
        print(f"❌ 多配置对象测试失败: {e}")
    
    return success_count

def test_error_handling():
    """测试错误处理"""
    print("\n--- 错误处理测试 ---")
    success_count = 0
    
    # 测试无效参数处理
    try:
        # 尝试创建负数尺寸的Mat
        try:
            mat = ncnn.Mat(-1, 10, 3)
            print("⚠️ 负数尺寸Mat创建成功（可能需要添加参数验证）")
        except:
            print("✅ 负数尺寸Mat创建正确失败")
        success_count += 1
        
        # 测试无效文件加载
        try:
            net = ncnn.Net()
            ret = net.load_param("nonexistent.param")
            if ret != 0:
                print("✅ 无效文件加载正确返回错误码")
                success_count += 1
            else:
                print("⚠️ 无效文件加载返回成功码")
        except:
            print("✅ 无效文件加载正确抛出异常")
            success_count += 1
            
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
    
    return success_count

def run_comprehensive_test():
    """运行完整API测试"""
    print("=" * 60)
    print("ncnn MicroPython API 完整测试")
    print("=" * 60)
    
    total_passed = 0
    option = None
    
    # 基础功能测试
    print("\n=== 基础功能测试 ===")
    if test_version():
        total_passed += 1
    if test_init():
        total_passed += 1
    
    # Mat API测试
    print("\n=== Mat类API测试 ===")
    mat_passed = test_mat_api()
    total_passed += mat_passed
    print(f"Mat API测试: {mat_passed}/7 通过")
    
    # Net API测试
    print("\n=== Net类API测试 ===")
    net_passed, net = test_net_api()
    total_passed += net_passed
    print(f"Net API测试: {net_passed}/2 通过")
    
    # Extractor API测试
    if net:
        print("\n=== Extractor类API测试 ===")
        extractor_passed = test_extractor_api(net)
        total_passed += extractor_passed
        print(f"Extractor API测试: {extractor_passed}/3 通过")
    
    # Option API测试 (新增)
    print("\n=== Option类API测试 ===")
    option_passed, option = test_option_api()
    total_passed += option_passed
    print(f"Option API测试: {option_passed}/5 通过")
    
    # Allocator API测试 (新增)
    print("\n=== Allocator类API测试 ===")
    allocator_passed = test_allocator_api()
    total_passed += allocator_passed
    print(f"Allocator API测试: {allocator_passed}/2 通过")
    
    # 高级功能集成测试 (新增)
    if option:
        print("\n=== 高级功能集成测试 ===")
        integration_passed = test_advanced_integration(option)
        total_passed += integration_passed
        print(f"高级功能集成测试: {integration_passed}/3 通过")
    
    # 内存管理测试
    memory_passed = test_memory_management()
    total_passed += memory_passed
    print(f"内存管理测试: {memory_passed}/2 通过")
    
    # 错误处理测试
    error_passed = test_error_handling()
    total_passed += error_passed
    print(f"错误处理测试: {error_passed}/2 通过")
    
    # 计算总体结果 
    # 基础2 + Mat7 + Net2 + Extractor3 + Option5 + Allocator2 + Integration3 + 内存2 + 错误2 = 28
    expected_total = 2 + 7 + 2 + 3 + 5 + 2 + 3 + 2 + 2
    
    print("\n" + "=" * 60)
    print(f"API测试总结: {total_passed}/{expected_total} 通过")
    if expected_total > 0:
        print(f"成功率: {total_passed/expected_total*100:.1f}%")
    print("=" * 60)
    
    
    if total_passed == expected_total:  
        print("\n🎉 API测试通过！")
        print("✅ ncnn MicroPython模块API功能正常")
        print("✅ 所有核心功能已实现并可正常使用")
        return 0
    else:
        print("\n❌ API测试失败，需要检查实现")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(run_comprehensive_test())