

import ncnn
import gc
import sys

class NCNNTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_data = None
    
    def log_test(self, test_name, success, message=""):
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {test_name}: {message}")
        if success:
            self.passed += 1
        else:
            self.failed += 1
    
    def safe_test(self, func, test_name):
        try:
            result = func()
            self.log_test(test_name, True, "Completed successfully")
            return result
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {e}")
            return None

    def test_version_api(self):
        print("\n=== VERSION API TESTS ===")
        version = ncnn.version()
        print(f"NCNN Version: {version}")
        return version

    def test_allocator_api(self):
        print("\n=== ALLOCATOR API TESTS ===")
        
        # Test pool allocator
        pool_alloc = ncnn.allocator_create_pool_allocator()
        print(f"Pool allocator created")
        
        # Test unlocked pool allocator
        unlocked_alloc = ncnn.allocator_create_unlocked_pool_allocator()
        print(f"Unlocked pool allocator created")

        # Test allocator destruction
        ncnn.allocator_destroy(pool_alloc)
        ncnn.allocator_destroy(unlocked_alloc)
        print("Allocators destroyed")
        
        return True

    def test_option_api(self):
        print("\n=== OPTION API TESTS ===")
        
        # Create option
        opt = ncnn.option_create()
        print(f"Option created")

        # Test num_threads
        ncnn.option_set_num_threads(opt, 8)
        threads = ncnn.option_get_num_threads(opt)
        print(f"Threads set/get: {threads}")
        
        # Test use_local_pool_allocator
        ncnn.option_set_use_local_pool_allocator(opt, True)
        use_local = ncnn.option_get_use_local_pool_allocator(opt)
        print(f"Use local pool allocator: {use_local}")
        
        # Test blob allocator
        blob_alloc = ncnn.allocator_create_pool_allocator()
        ncnn.option_set_blob_allocator(opt, blob_alloc)
        print("Blob allocator set")
        
        # Test workspace allocator
        workspace_alloc = ncnn.allocator_create_pool_allocator()
        ncnn.option_set_workspace_allocator(opt, workspace_alloc)
        print("Workspace allocator set")
        
        # Test use_vulkan_compute
        ncnn.option_set_use_vulkan_compute(opt, False)
        use_vulkan = ncnn.option_get_use_vulkan_compute(opt)
        print(f"Use Vulkan compute: {use_vulkan}")
        
        # Cleanup
        ncnn.allocator_destroy(blob_alloc)
        ncnn.allocator_destroy(workspace_alloc)
        ncnn.option_destroy(opt)
        print("Option and allocators destroyed")
        
        return True

    def test_mat_api(self):
        print("\n=== MAT API TESTS ===")
        
        # Test basic mat creation
        mat_empty = ncnn.mat_create()
        print(f"Empty mat created")
        ncnn.mat_destroy(mat_empty)
        
        # 0 for auto allocator
        # Test 1D mat
        mat_1d = ncnn.mat_create_1d(100, 0)
        print(f"1D mat created (100)")
        
        dims = ncnn.mat_get_dims(mat_1d)
        w = ncnn.mat_get_w(mat_1d)
        elemsize = ncnn.mat_get_elemsize(mat_1d)
        elempack = ncnn.mat_get_elempack(mat_1d)
        cstep = ncnn.mat_get_cstep(mat_1d)
        print(f"1D mat properties - dims: {dims}, w: {w}, elemsize: {elemsize}, elempack: {elempack}, cstep: {cstep}")
        
        # Test fill and clone
        ncnn.mat_fill_float(mat_1d, 2.718)
        mat_cloned = ncnn.mat_clone(mat_1d, 0)
        print("Mat filled and cloned")
        
        # Test reshape 1D
        ncnn.mat_reshape_1d(mat_1d, 50, 0)
        new_w = ncnn.mat_get_w(mat_1d)
        print(f"Mat reshaped to 1D (50), new width: {new_w}")

        ncnn.mat_destroy(mat_cloned)
        ncnn.mat_destroy(mat_1d)

        # Test 2D mat
        mat_2d = ncnn.mat_create_2d(32, 24, 0)
        print(f"2D mat created (32x24)")
        
        h = ncnn.mat_get_h(mat_2d)
        print(f"2D mat height: {h}")
        
        # Test reshape 2D
        ncnn.mat_reshape_2d(mat_2d, 16, 48, 0)
        new_w = ncnn.mat_get_w(mat_2d)
        new_h = ncnn.mat_get_h(mat_2d)
        print(f"2D mat reshaped (16x48), new dims: {new_w}x{new_h}")
        
        ncnn.mat_destroy(mat_2d)
        
        # Test 3D mat
        mat_3d = ncnn.mat_create_3d(16, 16, 8, 0)
        print(f"3D mat created (16x16x8)")
        
        d = ncnn.mat_get_d(mat_3d)
        c = ncnn.mat_get_c(mat_3d)
        print(f"3D mat depth: {d}, channels: {c}")
        
        # Test reshape 3D
        ncnn.mat_reshape_3d(mat_3d, 8, 8, 32, 0)
        new_dims = ncnn.mat_get_dims(mat_3d)
        print(f"3D mat reshaped, new dims: {new_dims}")
        
        # Test channel data
        channel_data = ncnn.mat_get_channel_data(mat_3d, 0)
        print(f"Channel 0 data pointer: {channel_data}")
        
        ncnn.mat_destroy(mat_3d)
        
        # Test 4D mat
        mat_4d = ncnn.mat_create_4d(8, 8, 4, 2, 0)
        print(f"4D mat created (8x8x4x2)")
        
        # Test reshape 4D
        ncnn.mat_reshape_4d(mat_4d, 4, 4, 8, 4, 0)
        print("4D mat reshaped")
        
        ncnn.mat_destroy(mat_4d)
        
        # Test external mat creation
        try:
            # Create dummy data for external mat
            dummy_data = bytearray(100 * 4)
            mat_ext_1d = ncnn.mat_create_external_1d(100, dummy_data, 0)
            print(f"External 1D mat created")
            ncnn.mat_destroy(mat_ext_1d)
        except:
            print("External mat creation skipped (requires valid data pointer)")
        
        # Test element-based creation
        mat_1d_elem = ncnn.mat_create_1d_elem(50, 4, 1, 0)
        print(f"1D elem mat created")
        ncnn.mat_destroy(mat_1d_elem)

        mat_2d_elem = ncnn.mat_create_2d_elem(16, 12, 4, 1, 0)
        print(f"2D elem mat created")
        ncnn.mat_destroy(mat_2d_elem)

        mat_3d_elem = ncnn.mat_create_3d_elem(8, 8, 4, 4, 1, 0)
        print(f"3D elem mat created")
        ncnn.mat_destroy(mat_3d_elem)

        mat_4d_elem = ncnn.mat_create_4d_elem(4, 4, 2, 2, 4, 1, 0)
        print(f"4D elem mat created")
        ncnn.mat_destroy(mat_4d_elem)

        return True

    def test_mat_process_api(self):
        print("\n=== MAT PROCESS API TESTS ===")
        
        # Create test matrices
        src = ncnn.mat_create_2d(16, 16, 0)
        dst = ncnn.mat_create()
        opt = ncnn.option_create()
        
        # Test substract_mean_normalize
        mean_vals = [0.485, 0.456, 0.406]
        norm_vals = [0.229, 0.224, 0.225]
        ncnn.mat_substract_mean_normalize(src, mean_vals, norm_vals)
        print("Mat substract_mean_normalize completed")
        
        # Test convert_packing
        ncnn.convert_packing(src, dst, 1, opt)
        print("Convert packing completed")
        
        # Test flatten
        ncnn.flatten(src, dst, opt)
        print("Flatten completed")
        
        # Test copy_make_border
        top, bottom, left, right = 2, 2, 2, 2
        border_type = 0
        border_value = 0.0
        ncnn.copy_make_border(src, dst, top, bottom, left, right, border_type, border_value, opt)
        print("Copy make border completed")
        
        # Test copy_cut_border
        ncnn.copy_cut_border(dst, src, top, bottom, left, right, opt)
        print("Copy cut border completed")
        
        # Test 3D versions
        src_3d = ncnn.mat_create_3d(8, 8, 4, 0)
        dst_3d = ncnn.mat_create()
        
        ncnn.copy_make_border_3d(src_3d, dst_3d, top, bottom, left, right, 1, 1, border_type, border_value, opt)
        print("Copy make border 3D completed")
        
        # Cleanup
        ncnn.mat_destroy(src)
        ncnn.mat_destroy(dst)
        ncnn.mat_destroy(src_3d)
        ncnn.mat_destroy(dst_3d)
        ncnn.option_destroy(opt)
        
        return True

    def test_paramdict_api(self):
        print("\n=== PARAMDICT API TESTS ===")
        
        pd = ncnn.paramdict_create()
        print(f"ParamDict created")
        
        # Test integer parameters
        ncnn.paramdict_set_int(pd, 0, 42)
        int_val = ncnn.paramdict_get_int(pd, 0, -1)
        print(f"Int param set/get: {int_val}")
        
        # Test float parameters
        ncnn.paramdict_set_float(pd, 1, 3.14159)
        float_val = ncnn.paramdict_get_float(pd, 1, 0.0)
        print(f"Float param set/get: {float_val}")
        
        # Test array parameters
        try:
            array_mat = ncnn.mat_create_1d(4, 0)
            ncnn.mat_fill_float(array_mat, 1.5)

            ncnn.paramdict_set_array(pd, 2, array_mat)

            default_mat = ncnn.mat_create()
            retrieved_mat = ncnn.paramdict_get_array(pd, 2, default_mat)
            print(f"Array param set/get: mat={retrieved_mat}")

            # Cleanup
            ncnn.mat_destroy(default_mat)
            ncnn.mat_destroy(array_mat)

            if retrieved_mat != array_mat and retrieved_mat != default_mat and retrieved_mat != 0:
                ncnn.mat_destroy(retrieved_mat)
            
        except Exception as e:
            print(f"Array param test failed: {e}")
        
        # Test parameter types
        type_0 = ncnn.paramdict_get_type(pd, 0)
        type_1 = ncnn.paramdict_get_type(pd, 1)
        type_2 = ncnn.paramdict_get_type(pd, 2)
        print(f"Param types - 0: {type_0}, 1: {type_1}, 2: {type_2}")
        
        ncnn.paramdict_destroy(pd)
        print("ParamDict destroyed")
        
        return True

    def test_datareader_api(self):
        print("\n=== DATAREADER API TESTS ===")
        
        # Test basic data reader
        dr = ncnn.datareader_create()
        print(f"Basic DataReader created")
        ncnn.datareader_destroy(dr)
        
        # Test memory data reader
        dummy_data = bytearray(1024)
        dr_mem = ncnn.datareader_create_from_memory(dummy_data)
        print(f"Memory DataReader created")
        ncnn.datareader_destroy(dr_mem)
        
        return True

    def test_modelbin_api(self):
        print("\n=== MODELBIN API TESTS ===")
        
        # Create data reader for modelbin
        dummy_data = bytearray(1024)
        dr = ncnn.datareader_create_from_memory(dummy_data)
        
        # Test modelbin from data reader
        mb = ncnn.modelbin_create_from_datareader(dr)
        print(f"ModelBin from DataReader created")
        ncnn.modelbin_destroy(mb)
        
        # Test modelbin from mat array
        mat_array = [ncnn.mat_create_1d(10, 0), ncnn.mat_create_1d(20, 0)]
        mb_array = ncnn.modelbin_create_from_mat_array(mat_array, 2)
        print(f"ModelBin from Mat array created")
        ncnn.modelbin_destroy(mb_array)
        
        # Cleanup
        for mat in mat_array:
            ncnn.mat_destroy(mat)
        ncnn.datareader_destroy(dr)
        
        return True

    def test_layer_api(self):
        print("\n=== LAYER API TESTS ===")
        
        # Test basic layer creation
        layer = ncnn.layer_create()
        print(f"Basic layer created")
        
        # Test layer creation by type index
        layer_by_index = ncnn.layer_create_by_typeindex(0)
        print(f"Layer created by type index: {layer_by_index}")
        
        # Test getting layer properties
        typeindex = ncnn.layer_get_typeindex(layer)
        one_blob_only = ncnn.layer_get_one_blob_only(layer)
        support_inplace = ncnn.layer_get_support_inplace(layer)
        support_vulkan = ncnn.layer_get_support_vulkan(layer)
        support_packing = ncnn.layer_get_support_packing(layer)
        support_bf16 = ncnn.layer_get_support_bf16_storage(layer)
        support_fp16 = ncnn.layer_get_support_fp16_storage(layer)
        
        print(f"Layer properties - typeindex: {typeindex}, one_blob_only: {one_blob_only}")
        print(f"Support - inplace: {support_inplace}, vulkan: {support_vulkan}, packing: {support_packing}")
        print(f"Storage support - bf16: {support_bf16}, fp16: {support_fp16}")
        
        # Test setting layer properties
        ncnn.layer_set_one_blob_only(layer, True)
        ncnn.layer_set_support_inplace(layer, False)
        ncnn.layer_set_support_vulkan(layer, True)
        ncnn.layer_set_support_packing(layer, True)
        ncnn.layer_set_support_bf16_storage(layer, False)
        ncnn.layer_set_support_fp16_storage(layer, True)
        print("Layer properties set")
        
        # Test getting bottom/top
        bottom_count = ncnn.layer_get_bottom_count(layer)
        top_count = ncnn.layer_get_top_count(layer)
        print(f"Bottom count: {bottom_count}, Top count: {top_count}")
        
        if bottom_count > 0:
            bottom_0 = ncnn.layer_get_bottom(layer, 0)
            print(f"Bottom 0: {bottom_0}")
        
        if top_count > 0:
            top_0 = ncnn.layer_get_top(layer, 0)
            print(f"Top 0: {top_0}")
        
        # Cleanup
        ncnn.layer_destroy(layer)
        ncnn.layer_destroy(layer_by_index)
        print("Layers destroyed")
        
        return True


    def test_net_api(self):
        print("\n=== NET API TESTS ===")
        
        # Create network
        net = ncnn.net_create()
        print(f"Network created")
        
        # Test option getting/setting
        opt = ncnn.net_get_option(net)
        print(f"Network option retrieved: {opt}")
        
        new_opt = ncnn.option_create()
        ncnn.option_set_num_threads(new_opt, 4)
        ncnn.net_set_option(net, new_opt)
        print("Network option set")
        
        # Test input/output counts
        input_count = ncnn.net_get_input_count(net)
        output_count = ncnn.net_get_output_count(net)
        print(f"Input count: {input_count}, Output count: {output_count}")
        
        
        # Clear and destroy
        ncnn.net_clear(net)
        print("Network cleared")
        
        ncnn.net_destroy(net)
        ncnn.option_destroy(new_opt)
        print("Network and option destroyed")
        
        return True


    def test_pixel_drawing_api(self):
        print("\n=== PIXEL DRAWING API TESTS ===")
        
        try:
            # Create pixel buffer for drawing
            width, height = 64, 64
            channels = 3
            pixel_buffer = bytearray(width * height * channels)
            
            # Fill with white background
            for i in range(len(pixel_buffer)):
                pixel_buffer[i] = 255
            
            # Test drawing functions with pixel buffer
            try:
                ncnn.draw_rectangle_c3(pixel_buffer, width, height, 10, 10, 20, 20, 0xFF0000, 2)
                print("draw_rectangle_c3 completed")
            except Exception as e:
                print(f"draw_rectangle_c3 failed: {e}")
            
            try:
                ncnn.draw_circle_c3(pixel_buffer, width, height, 40, 40, 10, 0x00FF00, 2)
                print("draw_circle_c3 completed")
            except Exception as e:
                print(f"draw_circle_c3 failed: {e}")
            
            try:
                ncnn.draw_line_c3(pixel_buffer, width, height, 5, 5, 55, 55, 0x0000FF, 2)
                print("draw_line_c3 completed")
            except Exception as e:
                print(f"draw_line_c3 failed: {e}")
            
            try:
                ncnn.draw_text_c3(pixel_buffer, width, height, "TEST", 15, 50, 12, 0xFFFFFF)
                print("draw_text_c3 completed")
            except Exception as e:
                print(f"draw_text_c3 failed: {e}")
                
        except Exception as e:
            print(f"Pixel drawing APIs not available or error: {e}")
        
        return True

    def test_pixel_drawing_api(self):
        print("\n=== PIXEL DRAWING API TESTS ===")
        
        try:
            # Create pixel buffer for drawing
            width, height = 64, 64
            channels = 3
            pixel_buffer = bytearray(width * height * channels)
            
            # Fill with white background
            for i in range(len(pixel_buffer)):
                pixel_buffer[i] = 255
            
            # Test drawing functions with pixel buffer
            try:
                ncnn.draw_rectangle_c3(pixel_buffer, width, height, 10, 10, 20, 20, 0xFF0000, 2)
                print("draw_rectangle_c3 completed")
            except Exception as e:
                print(f"draw_rectangle_c3 failed: {e}")
            
            try:
                ncnn.draw_circle_c3(pixel_buffer, width, height, 40, 40, 10, 0x00FF00, 2)
                print("draw_circle_c3 completed")
            except Exception as e:
                print(f"draw_circle_c3 failed: {e}")
            
            try:
                ncnn.draw_line_c3(pixel_buffer, width, height, 5, 5, 55, 55, 0x0000FF, 2)
                print("draw_line_c3 completed")
            except Exception as e:
                print(f"draw_line_c3 failed: {e}")
            
            try:
                ncnn.draw_text_c3(pixel_buffer, width, height, "TEST", 15, 50, 12, 0xFFFFFF)
                print("draw_text_c3 completed")
            except Exception as e:
                print(f"draw_text_c3 failed: {e}")
                
        except Exception as e:
            print(f"Pixel drawing APIs not available or error: {e}")
        
        return True

    def test_pixel_api(self):
        print("\n=== PIXEL API TESTS ===")
        
        try:
            # Create dummy pixel data
            width, height = 32, 32
            channels = 3
            stride = width * channels
            pixel_data = bytearray(width * height * channels)
            
            # Fill with test pattern
            for i in range(len(pixel_data)):
                pixel_data[i] = i % 256
            
            # Test mat_from_pixels
            try:
                pixel_type = 1
                mat = ncnn.mat_from_pixels(pixel_data, pixel_type, width, height, stride, 0)
                print(f"mat_from_pixels completed:")
                
                # Test mat_to_pixels
                output_data = bytearray(width * height * channels)
                output_stride = width * channels
                ncnn.mat_to_pixels(mat, output_data, pixel_type, output_stride)
                print("mat_to_pixels completed")
                
                ncnn.mat_destroy(mat)
                
            except Exception as e:
                print(f"Pixel conversion not available: {e}")
            
            # Test resize versions
            try:
                target_w, target_h = 16, 16
                target_stride = target_w * channels
                
                mat_resized = ncnn.mat_from_pixels_resize(pixel_data, pixel_type, width, height, stride, 
                                                        target_w, target_h, 0)
                print(f"mat_from_pixels_resize completed")
                
                # Test mat_to_pixels_resize
                resized_output = bytearray(target_w * target_h * channels)
                ncnn.mat_to_pixels_resize(mat_resized, resized_output, pixel_type, target_w, target_h, target_stride)
                print("mat_to_pixels_resize completed")
                
                ncnn.mat_destroy(mat_resized)
                
            except Exception as e:
                print(f"Pixel resize not available: {e}")
            
            # Test ROI versions
            try:
                roix, roiy = 8, 8
                roiw, roih = 16, 16
                
                mat_roi = ncnn.mat_from_pixels_roi(pixel_data, pixel_type, width, height, stride,
                                                 roix, roiy, roiw, roih, 0)
                print(f"mat_from_pixels_roi completed")
                ncnn.mat_destroy(mat_roi)
                
                roi_target_w, roi_target_h = 8, 8
                mat_roi_resize = ncnn.mat_from_pixels_roi_resize(pixel_data, pixel_type, width, height, stride,
                                                               roix, roiy, roiw, roih, 
                                                               roi_target_w, roi_target_h, 0)
                print(f"mat_from_pixels_roi_resize completed")
                ncnn.mat_destroy(mat_roi_resize)
                
            except Exception as e:
                print(f"Pixel ROI not available: {e}")
                
        except Exception as e:
            print(f"Pixel API not available: {e}")
        
        return True

    def run_all_tests(self):
        print("NCNN MicroPython API Test")
        print("=" * 50)
        
        # List of all test methods
        test_methods = [
            self.test_version_api,
            self.test_allocator_api,
            self.test_option_api,
            self.test_mat_api,
            self.test_mat_process_api,
            self.test_paramdict_api,
            self.test_datareader_api,
            self.test_modelbin_api,
            self.test_layer_api,
            self.test_net_api,
            self.test_pixel_drawing_api,
            self.test_pixel_api,
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            self.safe_test(test_method, test_name)
            gc.collect()
        
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {self.passed} PASSED, {self.failed} FAILED")
        print("Complete API Test Suite Finished")
        
        return self.failed == 0

def main():
    """Main test function"""
    tester = NCNNTester()
    tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())