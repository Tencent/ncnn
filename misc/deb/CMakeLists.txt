#准备的版本设置
message(STATUS "CMAKE_TOOLCHAIN_FILE =DEB")
 

#说明要生成的是deb包

set(CPACK_GENERATOR "DEB")

############下面是设置debian/control文件中的内容
set(NCNN_VERSION_MAJOR 1)
set(NCNN_VERSION_MINOR 0)
set(NCNN_VERSION_PATCH ${NCNN_VERSION})
#设置版本信息
set(CPACK_PACKAGE_VERSION_MAJOR "${NCNN_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${NCNN_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${NCNN_VERSION_PATCH}")

#设置安装包的包名，打好的包将会是packagename-version-linux.deb，如果不设置，默认是工程名
set(CPACK_PACKAGE_NAME "ncnn")

#设置程序名，就是程序安装后的名字
set(CPACK_DEBIAN_PACKAGE_NAME "ncnn")

#设置架构
#set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")

#设置依赖
#set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libdbus-1-3, libqt5dbus5, libqt5core5a, libqt5gui5, libqt5webkit5, libqt5network5, ssh, sshfs")

#设置section
set(CPACK_DEBIAN_PACKAGE_SECTION "AI")

#设置priority
set(CPACK_DEBIAN_PACKAGE_PRIORITY "Optional")

#设置description
set(CPACK_PACKAGE_DESCRIPTION "ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/Tencent/ncnn") 
set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/images/128-ncnn.png")
#设置联系方式
set(CPACK_PACKAGE_CONTACT "QQ群：637093648")

#设置维护人
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "members of QQ群：637093648")

##############debian/control设置完成了，下面设置控制脚本，动态库，changelog等

#lib目录下保存这一些动态库和安装控制脚本（postinst、postrm、preinst、prerm）  doc目录下是changelog和copyright等文件 这些东西是打包时要依赖的一些额外的东西
set(LIBS_DIR ${CMAKE_SOURCE_DIR}/lib)
set(DOCS_DIR ${CMAKE_SOURCE_DIR}/doc)

#设置控制脚本，这些都是大包者本人自己准备的，没有就算了
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
 "${LIBS_DIR}/postinst;${LIBS_DIR}/postrm;${LIBS_DIR}/prerm")

#把动态库文件安装到指定的位置，这些动态库是程序必须依赖的且无法通过其他方法安装，所以我只能想到用这种笨办法来设置了。
#install(FILES ${LIBS_DIR}/libqca-gnupg.so ${LIBS_DIR}/libqca-logger.so DESTINATION /usr/lib/x86_64-linux-gnu/)

#最后把我们的changelog等东西也加进来，project是你的工程名
#install(FILES ${DOCS_DIR}/changelog.Debian.gz ${DOCS_DIR}/changelog.gz ${DOCS_DIR}/copyright ${DOCS_DIR}/README.Debian DESTINATION /usr/share/doc/project)
set(CPACK_DEB_COMPONENT_INSTALL ON)
#最后 别忘记最重要的  我们使用cpack工具来实现打包工作的
include(CPack)