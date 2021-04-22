include!("ncnn_c_api.rs");
 


use std::ffi::CStr;
use std::time::Instant; // timer
use std::env;
use std::path::PathBuf;
use std::str::FromStr;
use std::os::raw::c_char;
use std::time::{Duration, SystemTime};
use std::ffi::CString;
impl __ncnn_datareader_t{
    fn read(        dr: ncnn_datareader_t,        buf: *mut ::std::os::raw::c_void,        size: size_t,    ) -> size_t{
        // memset(buf, 0, size);
        unsafe{ 
        std::ptr::write_bytes(buf, 0, size as usize);
        println!("write_bytes {}", size);
        }
        return size;
    }
}
fn main() {
    unsafe {
        let opt = ncnn_option_create();
        ncnn_option_set_use_vulkan_compute(opt, 1);
        ncnn_option_set_num_threads(opt, 4);

        let   model  =vec!["squeezenet", "227","squeezenet_int8", "227", "mobilenet", "224", "mobilenet_int8", "224", "mobilenet_v2", "224", "mobilenet_v3", "224",
        "shufflenet", "224", "shufflenet_v2", "224", "mnasnet", "224", "proxylessnasnet", "224",
        "efficientnet_b0", "224", "regnety_400m", "224", "blazeface", "128", "googlenet", "224", "googlenet_int8", "224",
        "resnet18", "224", "resnet18_int8", "224", "alexnet", "227", "vgg16", "224", "vgg16_int8", "224", "resnet50", "224", "resnet50_int8", "224",
        "squeezenet_ssd", "300", "squeezenet_ssd_int8", "300", "mobilenet_ssd", "300", "mobilenet_ssd_int8", "300",
        "mobilenet_yolo", "416", "mobilenetv2_yolov3", "352", "yolov4-tiny", "416"];

        let mut i = 0;
        let g_enable_cooling_down=true;
        while i <= model.len()-1 
        {
            let modelname=model[i];            

   
            i += 1;
            let wh=model[i].parse::<i32>().unwrap();            
 
            i += 1;

            {
                let ncnn_allocator_t0=ncnn_allocator_create_unlocked_pool_allocator();
  
                let matin = ncnn_mat_create(wh, wh, 3);
              
                ncnn_mat_fill_float(matin, 0.01);
          
                let net = ncnn_net_create();
                ncnn_net_set_option(net, opt);
                let mut parampath=modelname.to_string() +".param";
            
                let ip = CString::new(parampath).unwrap();
                let mut rtn=ncnn_net_load_param(net, ip.as_ptr());
             
                let mut dr= ncnn_DataReader_read_empty() ;
           
                rtn=ncnn_net_load_model_datareader(net, dr);
           
                if (g_enable_cooling_down)
                {
                    // sleep 10 seconds for cooling down SOC  
                        std::thread::sleep(std::time::Duration::from_millis(10));
                }
            
                let mut mutout:ncnn_mat_t =ncnn_mat_create(wh, wh, 3);
                // println!("ncnn_mat_create {}",0);
                // warm up
                // for (int i = 0; i < g_warmup_loop_count; i++)
                let mut k=0;  

                while k<4
                {
                    let ex = ncnn_extractor_create(net);
                    // ncnn_extractor_input(ex, "data".as_ptr() as *const c_char, matin);
                    let data = CString::new( "data").unwrap();
                    ncnn_extractor_input(ex, data.as_ptr(), matin);
                    let output = CString::new( "output").unwrap();
                    ncnn_extractor_extract(ex, output.as_ptr(),&mut mutout  as *mut _);
                    k=k+1;
                }
            
                {
                    let mut tests=0;
                    let mut time_min = std::u128::MAX;
                    let mut time_max = std::u128::MIN;
                    let mut time_avg = 0;
                    while tests < 10 
                    {
                        let   start= Instant::now();
                
                        {
                            let ex = ncnn_extractor_create(net);
                        //     ncnn_extractor_input(ex, "data".as_ptr() as *const c_char, matin);
                        // ncnn_extractor_extract(ex, "output".as_ptr() as *const c_char,&mut mutout  as *mut _);
                            let data = CString::new( "data").unwrap();
                            ncnn_extractor_input(ex, data.as_ptr(), matin);
                            let output = CString::new( "output").unwrap();
                            ncnn_extractor_extract(ex, output.as_ptr(),&mut mutout  as *mut _);
                        }
                
                        // let end = Instant::now();
                        
                        // let time = end.duration_since(start).expect("Clock may have gone backwards");
                        // let time = end.duration_since(start).expect("Clock may have gone backwards");
                        // time_min = time.as_millis().min(std::u128::MIN);
                        // time_max = time.as_millis().max(std::u128::MAX);
                        let end=start.elapsed().as_millis();
                        time_min = end.min(time_min);
                        time_max = end.max(time_max);
                        time_avg +=start.elapsed().as_millis();
                        tests=tests+1;
                    }
                
                    time_avg /= 10;
                    println!(" {} min {}  max {} avg {} ",modelname,time_min, time_max, time_avg);
                    // fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
                }
                
            }
        }


        // while(model.is_empty()==false)
        // {
        //     let mmdel=model.pop();
        //     let whstr=model.pop();
        //     let wh = FromStr::from_str(whstr).unwrap();
        // }

        // int modellen = sizeof(model) / sizeof(model[0]);
        // for (int i = 0; i < modellen; i++)
        // {
        //     char* mmodel = model[i];
        //     int wh = atoi(model[++i]);
        //     benchmark(mmodel, wh, wh, 3, opt);
        // }
    
    }
    // println!("Hello, world!");
}
