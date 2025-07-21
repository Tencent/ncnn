#ifndef THREAD_H
#define THREAD_H
#include "layer.h"
#include "TheadInfo.h"
#if defined __ANDROID__ || defined __linux__ || defined __APPLE__
#include <pthread.h>
#endif
namespace ncnn
{
    struct ThreadInfoExc{
        int threadid;
        size_t start_index;
        size_t end_index;
        ThreadWorkspace* workspace;
        std::vector<ncnn::Mat>* mats;
        Option* opt;
        MutilThread* manager;
        #if defined _WIN32
        CoreInfo* coreinfo;
        #endif
    };
    struct ThreadWorkspace{
        Layer* layer;
    };
    class MutilThread
    {
    private:
        Option m_opt;
        volatile int helpid;
        ThreadWorkspace workspace;
    public:
        MutilThread(ThreadWorkspace _workspace,const Option& opt);
        void join(std::vector<ncnn::Mat>& mats);
        std::vector<bool> threadsComplete;
        ~MutilThread();
    };
    
} // namespace ncnn
#endif
