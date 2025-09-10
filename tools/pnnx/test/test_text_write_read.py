# test_text_write_read.py
# 写入复杂文本（包含特殊字符）
text = "npy_test_123!@#$%^&*()_+-=[]{}|;':\",./<>? 中文测试"
with open("test_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

# 读取并对比
with open("test_text.txt", "r", encoding="utf-8") as f:
    read_text = f.read()

if text == read_text:
    print("✅ 文本文件写入/读取完全一致！")
else:
    print("❌ 文本文件存在篡改！")
    print(f"预期：{text}")
    print(f"实际：{read_text}")
