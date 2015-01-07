#
# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
LOCAL_PATH:= $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE := lib_normals 
LOCAL_SRC_FILES := cuda/lib_normals.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libcudart_static
LOCAL_LIB_PATH += $(CUDA_TOOLKIT_ROOT)/targets/armv7-linux-androideabi/lib/
LOCAL_SRC_FILES := $(LOCAL_LIB_PATH)/libcudart_static.a
include $(PREBUILT_STATIC_LIBRARY)


include $(CLEAR_VARS)
LOCAL_MODULE := libtango-prebuilt
LOCAL_SRC_FILES := ../tango-service-sdk/libtango_client_api.so
LOCAL_EXPORT_C_INCLUDES := ../tango-service-sdk/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

NVPACK := $(NDK_ROOT)/..
include $(NVPACK_ROOT)/OpenCV-2.4.8.2-Tegra-sdk/sdk/native/jni/OpenCV-tegra5-static-cuda.mk

LOCAL_MODULE    := libpoint_cloud_jni_example
LOCAL_STATIC_LIBRARIES := lib_normals libcudart_static
LOCAL_STATIC_LIBRARIES += nv_and_util nv_egl_util nv_glesutil nv_shader nv_file android_native_app_glue
LOCAL_SHARED_LIBRARIES := libtango-prebuilt
LOCAL_CFLAGS    := -std=c++11
LOCAL_SRC_FILES := jni/tango_data.cpp \
                   jni/tango_pointcloud.cpp \
                   jni/pointcloud.cpp \
                   ../tango-gl-renderer/camera.cpp \
                   ../tango-gl-renderer/gl_util.cpp \
                   ../tango-gl-renderer/grid.cpp \
                   ../tango-gl-renderer/axis.cpp \
                   ../tango-gl-renderer/transform.cpp \
                   ../tango-gl-renderer/frustum.cpp

LOCAL_C_INCLUDES := ../tango-gl-renderer/include \
                    ../third-party/glm/ \
										$(CUDA_TOOLKIT_ROOT)/targets/armv7-linux-androideabi/include \
										cuda

LOCAL_LDLIBS    := -llog -lGLESv2 -L$(SYSROOT)/usr/lib -landroid -lEGL
include $(BUILD_SHARED_LIBRARY)

$(call import-module,android/native_app_glue)
$(call import-add-path, $(NVPACK)/Samples/TDK_Samples/libs/jni)

$(call import-module,nv_and_util)
$(call import-module,nv_egl_util)
$(call import-module,nv_shader)
$(call import-module,nv_file)
$(call import-module,nv_glesutil)

