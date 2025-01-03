#include <stdlib.h>

#include "platform.h"

static int g_once_count = 0;

static void init()
{
    g_once_count++;
}

int test_call_once()
{
    static ncnn::OnceFlag flag = OnceFlagInit;
    ncnn::CallOnce(flag, &init);
    ncnn::CallOnce(flag, &init);
    ncnn::CallOnce(flag, &init);

    if (g_once_count != 1)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

int main()
{
    int ret;

    ret = test_call_once();
    if (ret)
        return ret;

    return 0;
}
