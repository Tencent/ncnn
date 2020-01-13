#!/usr/bin/bash

NAME=ncnn

##### package android lib
ANDROIDPKGNAME=${NAME}-android-lib
rm -rf $ANDROIDPKGNAME
mkdir -p $ANDROIDPKGNAME
mkdir -p $ANDROIDPKGNAME/armeabi-v7a
mkdir -p $ANDROIDPKGNAME/arm64-v8a
mkdir -p $ANDROIDPKGNAME/x86
mkdir -p $ANDROIDPKGNAME/x86_64
mkdir -p $ANDROIDPKGNAME/include
cp build-android-armv7/install/lib/lib${NAME}.a $ANDROIDPKGNAME/armeabi-v7a/
cp build-android-aarch64/install/lib/lib${NAME}.a $ANDROIDPKGNAME/arm64-v8a/
cp build-android-x86/install/lib/lib${NAME}.a $ANDROIDPKGNAME/x86/
cp build-android-x86_64/install/lib/lib${NAME}.a $ANDROIDPKGNAME/x86_64/
cp -r build-android-aarch64/install/include/* $ANDROIDPKGNAME/include/
rm -f $ANDROIDPKGNAME.zip
zip -9 -r $ANDROIDPKGNAME.zip $ANDROIDPKGNAME

##### package ios framework
IOSPKGNAME=${NAME}.framework
rm -rf $IOSPKGNAME
mkdir -p $IOSPKGNAME/Versions/A/Headers
mkdir -p $IOSPKGNAME/Versions/A/Resources
ln -s A $IOSPKGNAME/Versions/Current
ln -s Versions/Current/Headers $IOSPKGNAME/Headers
ln -s Versions/Current/Resources $IOSPKGNAME/Resources
ln -s Versions/Current/${NAME} $IOSPKGNAME/${NAME}
lipo -create \
    build-ios/install/lib/lib${NAME}.a \
    build-ios-sim/install/lib/lib${NAME}.a \
    -o $IOSPKGNAME/Versions/A/${NAME}
cp -r build-ios/install/include/* $IOSPKGNAME/Versions/A/Headers/
cp Info.plist ${IOSPKGNAME}/Versions/A/Resources/
rm -f $IOSPKGNAME.zip
zip -9 -y -r $IOSPKGNAME.zip $IOSPKGNAME


##### package android lib vulkan
ANDROIDPKGNAME=${NAME}-android-vulkan-lib
rm -rf $ANDROIDPKGNAME
mkdir -p $ANDROIDPKGNAME
mkdir -p $ANDROIDPKGNAME/armeabi-v7a
mkdir -p $ANDROIDPKGNAME/arm64-v8a
mkdir -p $ANDROIDPKGNAME/x86
mkdir -p $ANDROIDPKGNAME/x86_64
mkdir -p $ANDROIDPKGNAME/include
cp build-android-armv7-vulkan/install/lib/lib${NAME}.a $ANDROIDPKGNAME/armeabi-v7a/
cp build-android-aarch64-vulkan/install/lib/lib${NAME}.a $ANDROIDPKGNAME/arm64-v8a/
cp build-android-x86-vulkan/install/lib/lib${NAME}.a $ANDROIDPKGNAME/x86/
cp build-android-x86_64-vulkan/install/lib/lib${NAME}.a $ANDROIDPKGNAME/x86_64/
cp -r build-android-aarch64-vulkan/install/include/* $ANDROIDPKGNAME/include/
rm -f $ANDROIDPKGNAME.zip
zip -9 -r $ANDROIDPKGNAME.zip $ANDROIDPKGNAME

##### package ios framework vulkan
IOSPKGNAME=${NAME}-vulkan.framework
rm -rf $IOSPKGNAME
mkdir -p $IOSPKGNAME/Versions/A/Headers
mkdir -p $IOSPKGNAME/Versions/A/Resources
ln -s A $IOSPKGNAME/Versions/Current
ln -s Versions/Current/Headers $IOSPKGNAME/Headers
ln -s Versions/Current/Resources $IOSPKGNAME/Resources
ln -s Versions/Current/${NAME} $IOSPKGNAME/${NAME}
lipo -create \
    build-ios-vulkan/install/lib/lib${NAME}.a \
    build-ios-sim-vulkan/install/lib/lib${NAME}.a \
    -o $IOSPKGNAME/Versions/A/${NAME}
cp -r build-ios-vulkan/install/include/* $IOSPKGNAME/Versions/A/Headers/
cp Info.plist ${IOSPKGNAME}/Versions/A/Resources/
rm -f $IOSPKGNAME.zip
zip -9 -y -r $IOSPKGNAME.zip $IOSPKGNAME
