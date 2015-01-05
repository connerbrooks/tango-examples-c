#environment setup
GCC=$(NDK_ROOT)/toolchains/arm-linux-androideabi-4.6/gen_standalone/linux-x86_64/bin/arm-linux-androideabi-g++
NVCC=$(CUDA_TOOLKIT_ROOT)/bin/nvcc -ccbin $(GCC) -target-cpu-arch=ARM -m32 -arch=sm_30 -O3 -Xptxas '-dlcm=ca' -target-os-variant=Android

#INCLUDES+= /helper/files/include

CFLAGS += $(addprefix -I, $(INCLUDES))

-include subdir.mk

lib_normals.a: $(OBJS)
	$(NVCC) -lib -o "$@" $(OBJS)

clean:
	rm -rf *.a $(OBJS)
