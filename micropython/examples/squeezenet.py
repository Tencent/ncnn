import ncnn_mp
import struct
import sys

def detect_squeezenet(image_bytes, width, height):
    squeezenet = ncnn_mp.Net()
    squeezenet.option.use_vulkan_compute = True
    squeezenet.load_param("modules/squeezenet_v1.1.param")
    squeezenet.load_model("modules/squeezenet_v1.1.bin")

    in_mat = ncnn_mp.Mat.from_pixels_resize(image_bytes, ncnn_mp.Mat.PIXEL_BGR, width, height, width * 3, 227, 227)

    mean_vals = struct.pack('3f', 104.0, 117.0, 123.0)
    in_mat.substract_mean_normalize(mean_vals, None)

    ex = squeezenet.create_extractor()
    ex.input("data", in_mat)
    out_mat = ex.extract("prob")

    out_data_bytes = out_mat.to_bytes()
    cls_scores = list(struct.unpack(f'{out_mat.w}f', out_data_bytes))

    return cls_scores

def print_topk(cls_scores, topk):
    if not cls_scores:
        print("No scores to process.")
        return

    vec = [(score, i) for i, score in enumerate(cls_scores)]

    vec.sort(key=lambda x: x[0], reverse=True)

    print(f"--- Top {topk} results ---")
    for i in range(topk):
        score, index = vec[i]
        print(f"Index: {index}, Score: {score:.4f}")

def read_ppm(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline().strip()
        if header != b'P6':
            raise TypeError("Error: Only PPM P6 format is supported.")

        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        width, height = [int(i) for i in line.split()]
        
        f.readline()
        pixels = f.read(width * height * 3)
        bgr_pixels = bytearray(pixels)
        for i in range(0, len(bgr_pixels), 3):
            r, g, b = bgr_pixels[i], bgr_pixels[i+1], bgr_pixels[i+2]
            bgr_pixels[i], bgr_pixels[i+1], bgr_pixels[i+2] = b, g, r
            
        return bgr_pixels, width, height

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"\033[1;31mUsage:\033[0m {sys.argv[0]} [imagepath]", file=sys.stderr)
        sys.exit(-1)

    imagepath = sys.argv[1]
    pixels_data, width, height = read_ppm(imagepath)

    if pixels_data:
        print(f"Successfully read image: {imagepath} ({width}x{height})")
        
        cls_scores = detect_squeezenet(pixels_data, width, height)
        if cls_scores:
            print_topk(cls_scores, 3)
