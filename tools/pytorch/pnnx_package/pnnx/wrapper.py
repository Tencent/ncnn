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
    # traced_model = torch.jit.trace(model, torch.rand(1, 10))
    traced_model = torch.jit.trace(model, inputs)
    # traced_model('model.pt')
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


def convert_kwargs_to_cmd(**kwargs):
    ext_cmd = ""
    for key, value in kwargs.items():
        if isinstance(value, tuple):
            value_str = ','.join(map(str, value))
            ext_cmd += f" {key}=[{value_str}]"
        else:
            ext_cmd += f" {key}={value}"
    return ext_cmd


def convert_inputshape_to_str(inputshape=None):
    if inputshape is None:
        print("inputshape is None, which is required")
        exit(1)
    
    # generate inputshape string
    gen_str = ""
    if isinstance(inputshape, tuple):
        for idx, inputshape_ in enumerate(inputshape):
            # 不是第一个，添加逗号
            if idx:
                gen_str += ","
            inputshape_list = [dim for dim in inputshape_.shape]
            inputshape_str = ','.join(map(str,inputshape_list))
            gen_str += f"[{inputshape_str}]"
    else:
        inputshape_list = [dim for dim in inputshape.shape]
        inputshape_str = ','.join(map(str,inputshape_list))
        gen_str += f"[{inputshape_str}]"
    return gen_str


def convert_inputshape_to_cmd(inputshape=None, inputshape2=None):
    if inputshape is None:
        print("inputshape is None, which is required")
        exit(1)

    # generate inputshape to cmd string
    gen_input_cmd = " inputshape="
    gen_input_cmd += convert_inputshape_to_str(inputshape)
    if inputshape2 is not None:
        gen_input_cmd += " inputshape2="
        gen_input_cmd += convert_inputshape_to_str(inputshape2)
    return gen_input_cmd


def run(torchscript_path, inputshape=None, inputshape2=None, **kwargs):
    print("[+] entering pnnx_run")

    cmd = ""
    if sys.platform.startswith('linux'):
        cmd += "./bin/pnnx-20230816-ubuntu/pnnx "
    elif sys.platform.startswith('win'):
        cmd += "./bin/pnnx-20230816-windows/pnnx.exe "
    elif sys.platform.startswith('darwin'):
        cmd += "./bin/pnnx-20230816-macos/pnnx "
    else:
        print('无法识别当前系统')

    
    cmd += torchscript_path
    cmd += convert_inputshape_to_cmd(inputshape, inputshape2)
    cmd += convert_kwargs_to_cmd(**kwargs)

    print("cmd is: ", cmd)

    subprocess.run(cmd)

    print("[-] leaving pnnx_run")


def export(model, inputshape=None, inputshape2=None, **kwargs):
    print("[+] entering export")

    # 这里想做一件事：获取传递给函数的变量的原始变量名，以用于后续文件保存的命名
    # generate the filename from the name of the variable passed to it
    # for example: export(model_2333, ...) -> model_2333.pt
    current_frame = inspect.currentframe()
    previous_frame = inspect.getouterframes(current_frame)[1]
    call_func_code = inspect.getframeinfo(previous_frame[0]).code_context[0].strip()

    model_name = get_model_name(call_func_code)
    
    if inputshape is not None:
        model_torchscript_path = trace_model(model, model_name, inputshape)
        run(model_torchscript_path, inputshape=inputshape, inputshape2=inputshape2, **kwargs)
    else:
        print("inputshape is None, which is required")
        exit(1)


    print("[-] leaving export")