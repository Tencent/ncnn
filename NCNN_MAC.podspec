#
#  Be sure to run `pod spec lint NCNN_MAC.podspec' to ensure this is a
#  valid spec and to remove all comments including this before submitting the spec.
#
#  To learn more about Podspec attributes see https://guides.cocoapods.org/syntax/podspec.html
#  To see working Podspecs in the CocoaPods repo see https://github.com/CocoaPods/Specs/
#

Pod::Spec.new do |spec|

  # ―――  Spec Metadata  ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  These will help people to find your library, and whilst it
  #  can feel like a chore to fill in it's definitely to your advantage. The
  #  summary should be tweet-length, and the description more in depth.
  #

  spec.name         = "NCNN_MAC"
  spec.version      = "20210124"
  spec.summary      = "ncnn powerby Tencent."

  # This description is used to generate tags and improve search results.
  #   * Think: What does it do? Why did you write it? What is the focus?
  #   * Try to keep it short, snappy and to the point.
  #   * Write the description between the DESC delimiters below.
  #   * Finally, don't worry about the indent, CocoaPods strips it!
  spec.description  = <<-DESC
  ncnn is a high-performance neural network inference framework optimized for the mobile platform.
    DESC

   spec.homepage     = "https://github.com/Tencent/ncnn"

   spec.license      = "BSD 3-Clause"
   spec.license      = { :type => "BSD 3-Clause", :file => "https://github.com/Tencent/ncnn/LICENSE.txt" }
0
  spec.author             = { "DCTech" => "412200533@qq.com" }

  spec.platform     = :osx
  spec.platform     = :osx, "10.12"

  spec.source       = { :http => "https://github.com/Tencent/ncnn/releases/download/20210124/ncnn-20210124-macos-vulkan.zip"}
  spec.public_header_files = "${PODS_ROOT}/NCNN_IOS/openmp.framework/Headers/*.h", "${PODS_ROOT}/NCNN_IOS/ncnn.framework/Header/SPIRV/*.h","${PODS_ROOT}/NCNN_IOS/ncnn.framework/Header/ncnn/*.h","${PODS_ROOT}/NCNN_IOS/ncnn.framework/Header/glslang/Public/*.h"

  spec.preserve_paths = "**/openmp.framework", "**/ncnn.framework","**/glslang.framework"

  spec.xcconfig = { 'HEADER_SEARCH_PATHS' => '${PODS_ROOT}/NCNN_IOS/openmp.framework/Headers/;${PODS_ROOT}/NCNN_IOS/ncnn.framework/Headers/;${PODS_ROOT}/NCNN_IOS/glslang.framework/Headers/'
  }

end
