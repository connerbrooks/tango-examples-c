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

#PCL libraries - full support
PCL_STATIC_LIB_DIR := ../third-party/pcl/pcl-android/lib
BOOST_STATIC_LIB_DIR := ../third-party/pcl/boost-android/lib
FLANN_STATIC_LIB_DIR := ../third-party/pcl/flann-android/lib
					
PCL_STATIC_LIBRARIES := pcl_common pcl_geometry pcl_kdtree pcl_octree pcl_sample_consensus \
							pcl_surface pcl_features pcl_io pcl_keypoints pcl_recognition \
							pcl_search pcl_tracking pcl_filters pcl_io_ply pcl_ml \
							pcl_registration pcl_segmentation 
BOOST_STATIC_LIBRARIES := boost_date_time boost_iostreams boost_regex boost_system \
							boost_filesystem boost_program_options boost_signals \
							boost_thread
FLANN_STATIC_LIBRARIES := flann_cpp_s flann_s

										
define build_pcl_static
	include $(CLEAR_VARS)
	LOCAL_MODULE:=$1
	LOCAL_SRC_FILES:=$(PCL_STATIC_LIB_DIR)/lib$1.a
	include $(PREBUILT_STATIC_LIBRARY)
endef

define build_boost_static
	include $(CLEAR_VARS)
	LOCAL_MODULE:=$1
	LOCAL_SRC_FILES:=$(BOOST_STATIC_LIB_DIR)/lib$1.a
	include $(PREBUILT_STATIC_LIBRARY)
endef

define build_flann_static
	include $(CLEAR_VARS)
	LOCAL_MODULE:=$1
	LOCAL_SRC_FILES:=$(FLANN_STATIC_LIB_DIR)/lib$1.a
	include $(PREBUILT_STATIC_LIBRARY)
endef

$(foreach module,$(PCL_STATIC_LIBRARIES),$(eval $(call build_pcl_static,$(module))))
$(foreach module,$(BOOST_STATIC_LIBRARIES),$(eval $(call build_boost_static,$(module))))
$(foreach module,$(FLANN_STATIC_LIBRARIES),$(eval $(call build_flann_static,$(module))))					



include $(CLEAR_VARS)
LOCAL_MODULE := libtango-prebuilt
LOCAL_SRC_FILES := ../tango-service-sdk/libtango_client_api.so
LOCAL_EXPORT_C_INCLUDES := ../tango-service-sdk/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE    := libpoint_cloud_jni_example
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
                    ../third-party/glm/

#pcl library
LOCAL_LDFLAGS += -L../third-party/pcl/pcl-android/lib \
				 -L../third-party/pcl/boost-android/lib \
				 -L../third-party/pcl/flann-android/lib
				  				
LOCAL_C_INCLUDES += ../third-party/pcl/pcl-android/include/pcl-1.6 \
					../third-party/pcl/boost-android/include \
					../third-party/pcl/eigen \
					../third-party/pcl/flann-android/include  		
			
LOCAL_STATIC_LIBRARIES   += pcl_common pcl_geometry pcl_kdtree pcl_octree pcl_sample_consensus \
							pcl_surface pcl_features pcl_io pcl_keypoints pcl_recognition \
							pcl_search pcl_tracking pcl_filters pcl_io_ply pcl_ml \
							pcl_registration pcl_segmentation 
							
LOCAL_STATIC_LIBRARIES   += boost_date_time boost_iostreams boost_regex boost_system \
							boost_filesystem boost_program_options boost_signals \
							boost_thread


LOCAL_LDLIBS    := -llog -lGLESv2 -L$(SYSROOT)/usr/lib
include $(BUILD_SHARED_LIBRARY)
