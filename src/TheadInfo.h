#ifndef THREAD_INFO_H
#define THREAD_INFO_H
#ifdef NCNN_MUTITHREAD
#if defined _WIN32
#include "cpu.h"
namespace ncnn {
struct CoreInfo
{
public:
    int id;
    int group;
    DWORD_PTR affinity;
};
class ThreadInfo
{
private:
    static ThreadInfo* thread_info;
    std::vector<CoreInfo> core_infos;
    ThreadInfo(/* args */);

public:
    static ThreadInfo* get();
    CoreInfo getCurrentCore();
    void getAllCore(std::vector<CoreInfo>& out);
};
} // namespace ncnn

#endif
#endif
#endif