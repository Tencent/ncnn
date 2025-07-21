#ifdef NCNN_MUTITHREAD
#ifdef _WIN32

#include "TheadInfo.h"
namespace ncnn
{

// 初始化静态成员
ThreadInfo* ThreadInfo::thread_info = nullptr;

ThreadInfo::ThreadInfo(/* args */)
{
    int groupCount = GetActiveProcessorGroupCount();
    for (WORD group = 0; group < groupCount; group++) {
        DWORD processorsInGroup = GetActiveProcessorCount(group);
        for (int i = 0; i < static_cast<int>(processorsInGroup); i++) {
            CoreInfo info;
            info.group = group;
            info.id = i + core_infos.size();
            info.affinity = (static_cast<DWORD_PTR>(1) << i);
            core_infos.push_back(info);
        }
    }
}

ThreadInfo* ThreadInfo::get()
{
    static Mutex lock;
    AutoLock guard(lock);
    
    if (!thread_info)
    {
        thread_info = new ThreadInfo();
    }
    return thread_info;
}

CoreInfo ThreadInfo::getCurrentCore()
{
    // 获取当前线程运行的CPU核心（支持多处理器组）
    DWORD_PTR process_affinity, system_affinity;
    GetProcessAffinityMask(GetCurrentProcess(), &process_affinity, &system_affinity);
    
    // 使用扩展API获取处理器组信息
    PROCESSOR_NUMBER proc_num;
    GetCurrentProcessorNumberEx(&proc_num);
    
    for (const auto& core : core_infos)
    {
        // 匹配组号和组内核心编号
        if (core.group == proc_num.Group && (core.affinity & (1ULL << proc_num.Number)))
        {
            return core;
        }
    }
    
    // 未找到时返回默认值
    return { -1, -1, 0 };
}

void ThreadInfo::getAllCore(std::vector<CoreInfo>& out)
{
    out = core_infos;
}
}

#endif
#endif
