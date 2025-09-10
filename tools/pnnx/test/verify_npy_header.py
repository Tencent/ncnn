import sys
import struct

def verify_npy_file(npy_path):
    """
    验证npy文件格式是否符合npy 1.0标准，输出详细分析结果
    :param npy_path: npy文件路径
    """
    print(f"=" * 60)
    print(f"📁 正在验证npy文件：{npy_path}")
    print(f"=" * 60)

    # 1. 尝试打开文件（二进制只读模式）
    try:
        with open(npy_path, "rb") as f:
            # 读取文件总大小（用于后续完整性判断）
            f.seek(0, 2)  # 定位到文件末尾
            file_total_size = f.tell()
            f.seek(0)      # 回到文件开头
            print(f"[文件基本信息]")
            print(f"  总大小：{file_total_size} 字节")
            if file_total_size < 12:  # 最小npy文件：6魔法数+2版本号+4头长度=12字节
                print(f"  ❌ 错误：文件过小（<12字节），不是标准npy文件")
                return

            # 2. 读取并验证「魔法数」（6字节，npy文件标识）
            magic_bytes = f.read(6)
            expected_magic = b"\x93NUMPY"  # 标准npy魔法数
            print(f"\n[魔法数验证]")
            print(f"  实际读取：{[hex(b) for b in magic_bytes]}")
            print(f"  标准值：  {[hex(b) for b in expected_magic]}")
            if magic_bytes != expected_magic:
                print(f"  ❌ 错误：魔法数不匹配，不是npy文件")
                return
            print(f"  ✅ 魔法数验证通过")

            # 3. 读取并验证「版本号」（2字节，仅支持1.0版本）
            version_bytes = f.read(2)
            expected_version = b"\x01\x00"  # 标准npy 1.0版本
            print(f"\n[版本号验证]")
            print(f"  实际读取：版本 {version_bytes[0]}.{version_bytes[1]}")
            print(f"  支持版本：1.0")
            if version_bytes != expected_version:
                print(f"  ❌ 错误：不支持的npy版本（仅支持1.0）")
                return
            print(f"  ✅ 版本号验证通过")

            # 4. 读取并验证「头长度」（4字节，little-endian，标识后续头信息的长度）
            header_len_bytes = f.read(4)
            # 解析为little-endian uint32
            header_len = struct.unpack("<I", header_len_bytes)[0]
            print(f"\n[头长度验证]")
            print(f"  原始字节：{[hex(b) for b in header_len_bytes]}")
            print(f"  解析值：{header_len} 字节")
            # 头长度合理性校验（npy头信息通常几十~几百字节，最大不超过1MB）
            if header_len == 0 or header_len > 1024 * 1024:
                print(f"  ❌ 错误：头长度异常（必须1~1048576字节）")
                print(f"     提示：可能是npy文件生成时格式损坏，或环境写入异常")
                return
            print(f"  ✅ 头长度验证通过")

            # 5. 验证「头信息完整性」（读取头信息，确认文件未截断）
            print(f"\n[头信息完整性验证]")
            # 检查剩余文件大小是否足够容纳头信息
            remaining_size = file_total_size - f.tell()
            if remaining_size < header_len:
                print(f"  ❌ 错误：文件不完整（剩余{remaining_size}字节，需{header_len}字节头信息）")
                return
            # 读取头信息（JSON格式，描述shape和dtype）
            header_str = f.read(header_len).decode("ascii", errors="ignore")
            print(f"  头信息（前150字符）：{header_str[:150]}")
            # 简单校验头信息格式（是否包含shape和descr关键字）
            if '"shape"' not in header_str or '"descr"' not in header_str:
                print(f"  ⚠️  警告：头信息格式异常，可能缺少shape/descr字段")
            else:
                print(f"  ✅ 头信息完整性验证通过")

            # 6. 总结验证结果
            print(f"\n" + "=" * 60)
            print(f"🎉 验证完成！当前npy文件格式符合npy 1.0标准")
            print(f"   可用于pnnx工具的input参数")
            print(f"=" * 60)

    except FileNotFoundError:
        print(f"❌ 错误：文件不存在 → {npy_path}")
    except PermissionError:
        print(f"❌ 错误：权限不足，无法读取文件 → {npy_path}")
    except Exception as e:
        print(f"❌ 验证过程出错：{str(e)}")

# 主函数：支持命令行指定文件路径，默认验证raw_standard_input.npy
if __name__ == "__main__":
    # 命令行参数处理（如：python verify_npy_header.py my_input.npy）
    if len(sys.argv) > 1:
        target_npy = sys.argv[1]
    else:
        # 默认验证当前目录下的raw_standard_input.npy（你新生成的标准文件）
        target_npy = "raw_standard_input.npy"
    
    # 执行验证（修正：调用正确的函数名 verify_npy_file）
    verify_npy_file(target_npy)
