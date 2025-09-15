import ncnn_mp
import struct

def read_pgm(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline()
        if header.strip() != b'P5':
            raise TypeError("Only support PGM:P5")

        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        width, height = [int(i) for i in line.split()]

        f.readline()
        pixels = f.read(width * height)
        return bytearray(pixels), width, height

net = ncnn_mp.Net()

net.load_param("modules/mnist.param")
net.load_model("modules/mnist.bin")

pixels_data, width, height = read_pgm("img/pgm/0.pgm")

print(f"image size: {width}x{height}")

in_mat = ncnn_mp.Mat(width, height, c=1)

# MicroPython don't support numpy
float_pixel_data = bytearray(width * height * 4)
for i in range(width * height):
    packed_float = struct.pack('<f', float(pixels_data[i]))
    float_pixel_data[i*4 : i*4+4] = packed_float

in_mat.from_bytes(float_pixel_data)

mean_vals = struct.pack('<f', 0.0)
norm_vals = struct.pack('<f', 1.0 / 255.0)

in_mat.substract_mean_normalize(mean_vals, norm_vals)

ex = net.create_extractor()

ex.input("data", in_mat)
out_mat = ex.extract("fc")
output_data = out_mat.to_bytes()

scores = struct.unpack('<10f', output_data)

max_score = max(scores)
predicted_digit = scores.index(max_score)

print(f"Result: \033[4m{predicted_digit}\033[0m with a score of {max_score:.4f}")