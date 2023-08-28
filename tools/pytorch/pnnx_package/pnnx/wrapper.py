import subprocess
import inspect
import sys
import torch

# model -> torchscript
def trace_model(model, model_name, inputs):
    print("[+] entering trace_model")

    model_torchscript_path = f"{model_name}.pt"  # naming the torchscript file by model name
    print("model_torchscript_path is: ", model_torchscript_path)

    print("start trace...")
    traced_model = torch.jit.trace(model, inputs)
    traced_model.save(model_torchscript_path)
    print("finish trace")


    print("[-] leaving trace_model")
    return model_torchscript_path


def get_model_name(call_func_code):
    print("[+] entering get_model_name")

    args = call_func_code[call_func_code.find('(') + 1:-1].split(',')
    print(args)

    print("[-] leaving get_model_name")
    return args[0]


def export(model, inputs):
    print("[+] entering export")

    # 这里想做一件事：获取传递给函数的变量的原始变量名，以用于后续文件保存的命名
    # generate the filename from the name of the variable passed to it
    # for example: export(model_2333, ...) -> model_2333.pt
    current_frame = inspect.currentframe()
    previous_frame = inspect.getouterframes(current_frame)[1]
    call_func_code = inspect.getframeinfo(previous_frame[0]).code_context[0].strip()

    model_name = get_model_name(call_func_code)
    
    if len(inputs)==1:
        model_torchscript_path = trace_model(model, model_name, inputs)
        run(model_torchscript_path, inputs)


    print("[-] leaving export")


def run(torchscript_path, inputs):
    print("[+] entering pnnx_run")

    if len(inputs)==1:
        cmd = ""
        if sys.platform.startswith('linux'):
            print('当前系统为 Linux')
            cmd += "./bin/pnnx-20230816-ubuntu/pnnx "
        elif sys.platform.startswith('win'):
            print('当前系统为 Windows')
            cmd += "./bin/pnnx-20230816-windows/pnnx.exe "
        elif sys.platform.startswith('darwin'):
            print('当前系统为 macOS')
            cmd += "./bin/pnnx-20230816-macos/pnnx "
        else:
            print('无法识别当前系统')

        
        cmd += torchscript_path
        
        inputshape_list = [dim for dim in inputs.shape]
        inputshape_str = ','.join(map(str,inputshape_list))

        cmd += f" inputshape=[{inputshape_str}]"

        print("cmd is: ", cmd)

        subprocess.run(cmd)
    else:
        pass

    print("[-] leaving pnnx_run")
