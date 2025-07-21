#ifndef THREAD_INFO_H
#define THREAD_INFO_H
#ifdef NCNN_MUTITHREAD
#if defined _WIN32
#include "cpu.h"
namespace ncnn
{
struct CoreInfo{
    public:
    int id;
    int group;
    DWORD_PTR affinity;
};
class TheadInfo
{
private:
    static ThreadInfo* thread_info;
    std::vector<CoreInfo> core_infos;
    TheadInfo(/* args */);
public:
    static ThreadInfo* get();
    CoreInfo getCurrentCore();
    void getAllCore(std::vector<CoreInfo>& out);
};
}

#endif
#endif
#endif