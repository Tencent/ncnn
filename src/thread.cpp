#include "thread.h"
#include "cpu.h"
#if defined __ANDROID__ || defined __linux__
#include <sched.h>
#endif

#if defined _WIN32
DWORD WINAPI winWorker(LPVOID lpParam)
{
    ncnn::ThreadInfoExc* info = (ncnn::ThreadInfoExc*)lpParam;
    if (info->coreinfo->group >= 0 && info->coreinfo->affinity != 0)
    {
        GROUP_AFFINITY groupAffinity;
        ZeroMemory(&groupAffinity, sizeof(groupAffinity));
        groupAffinity.Group = static_cast<WORD>(info->coreinfo->group);
        groupAffinity.Mask = info->coreinfo->affinity;

        return SetThreadGroupAffinity(GetCurrentThread(), &groupAffinity, NULL) != 0;
    }
    info->workspace->layer->forward_thread(info);
    info->manager->threadsComplete[info->threadid] = true;
    delete info;
    return 0;
}
#else
void* pthreadWorker(void* lpParam)
{
    ncnn::ThreadInfoExc* info = (ncnn::ThreadInfoExc*)lpParam;
#if defined __ANDROID__ || defined __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(info->threadid, &cpuset);
    // 绑定到指定核心
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
#endif
    info->workspace->layer->forward_thread(info);
    info->manager->threadsComplete[info->threadid] = true;
    delete info;
    return nullptr;
}
#endif
namespace ncnn {
MutilThread::MutilThread(ThreadWorkspace _workspace, const Option& opt)
{
    workspace = _workspace;
    m_opt = opt;
    threadsComplete.resize(opt.num_threads);
    for (int i = 0; i < opt.num_threads; i++)
    {
        threadsComplete[i] = false;
    }
    threadsComplete[helpid] = true;
}

MutilThread::~MutilThread()
{
    threadsComplete.clear();
}

void MutilThread::join(std::vector<Mat>& mats)
{
#if defined _WIN32
    Mat mat = mats[0];
    CoreInfo cur = TheadInfo::get()->getCurrentCore();
    std::vector<CoreInfo> cores;
    TheadInfo::get()->getAllCore(cores);
    std::vector<HANDLE> handles;
    ThreadInfoExc* curinfo = nullptr;
    size_t workersize = ((mat.w * mat.h * mat.d) / m_opt.num_threads + 1) * mat.c * mat.elemsize;
    size_t matlen = mats.size();
    for (int i = 0; i < m_opt.num_threads; i++)
    {
        ThreadInfoExc* info = new ThreadInfoExc();
        info->threadid = i;
        info->start_index = i * workersize;
        info->end_index = (i + 1) * workersize;
        if (info->end_index > matlen)
        {
            info->end_index = matlen;
        }
        info->workspace = &workspace;
        info->mats = &mats;
        info->opt = &m_opt;
        info->coreinfo = &cores[i];
        threadsComplete[i] = false;
        info->manager = this;
        if (cur.id == cores[i].id)
        {
            helpid = i;
            threadsComplete[i] = true;
            handles.push_back(nullptr);
            curinfo = info;
            continue;
        }
        handles.push_back(CreateThread(nullptr, 0, winWorker, info, 0, nullptr));
    }
    workspace.layer->forward_inplace(curinfo);
    delete curinfo;
    bool check = true;
    do
    {
        check = false;
        for (int i = 0; i < m_opt.num_threads; i++)
        {
            if (threadsComplete[i] == false)
            {
                check = true;
                break;
            }
        }
    } while (check);
    for (size_t i = 0; i < handles.size(); i++)
    {
        if (handles[i] != nullptr)
        {
            CloseHandle(handles[i]);
        }
    }
    handles.clear();
#else
    Mat mat = mats[0];
    int curid = -1;
#if defined __ANDROID__ || defined __linux__
    curid = sched_getcpu();
#endif

    std::vector<pthread_t> pthread_handles;
    ThreadInfoExc* curinfo = nullptr;
    size_t workersize = ((mat.w * mat.h * mat.d) / m_opt.num_threads + 1) * mat.c * mat.elemsize;
    size_t matlen = mats.size();
    for (int i = 0; i < m_opt.num_threads; i++)
    {
        ThreadInfoExc* info = new ThreadInfoExc();
        info->threadid = i;
        info->start_index = i * workersize;
        info->end_index = (i + 1) * workersize;
        if (info->end_index > matlen)
        {
            info->end_index = matlen;
        }
        info->workspace = &workspace;
        info->mats = &mats;
        info->opt = &m_opt;
        threadsComplete[i] = false;
        info->manager = this;
        if (curid == cores[i].id && curid > 1)
        {
            helpid = i;
            threadsComplete[i] = true;
            curinfo = info;
            continue;
        }
        pthread_handles.push_back(pthread_create(&pthread_handles[i], nullptr, pthreadWorker, info));
    }
    workspace.layer->forward_inplace(curinfo);
    delete curinfo;
    for (size_t i = 0; i < pthread_handles.size(); i++)
    {
        pthread_join(pthread_handles[i], nullptr);
    }
#endif
}
} // namespace ncnn