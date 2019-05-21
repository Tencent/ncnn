
## NCNN Image Watch Plugin for Visual Studio
Image Watch plugin is a good tool for better understanding insight of images. This tiny work offer a ".natvis" file which could add ncnn::Mat class support for Image Watch, and users could debug ncnn::Mat image just like debuging cv::Mat via Image Watch.

To use this plugin, please move this "ImageWatchNCNN.natvis" file to "C:/user/${your user name}/Documents/Visual Studio ${VS_Version}/Visualizers" folder. If not exist this folder, create it(such as: "C:\Users\nihui\Documents\Visual Studio 2017\Visualizers"). 

![](https://github.com/Tencent/ncnn/blob/master/tools/plugin/snapshot.png)

See [Image Watch Help](https://imagewatch.azurewebsites.net/ImageWatchHelp/ImageWatchHelp.htm) page for more advanced using tips of Image Watch(For example, get single channel from channels, such as getting confidence heatmap from forward result list {confidence, x1, y1, x2, y2}).