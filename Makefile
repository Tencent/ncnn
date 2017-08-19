# ncnn@tencent instal for raspbery pi
# by cgoxopx
all:
	-mkdir build-raspberry-armv7
	cd build-raspberry-armv7 && \
	cmake ../src &&\
	make