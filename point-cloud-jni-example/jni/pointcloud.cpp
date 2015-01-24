/*
 * Copyright 2014 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pointcloud.h"

static const char kVertexShader[] =
    "attribute vec4 vertex;\n"
    "attribute float color;\n"
    "uniform mat4 mvp;\n"
    "varying vec4 v_color;\n"
    "void main() {\n"
    "  gl_PointSize = 5.0;\n"
    "  gl_Position = mvp*vertex;\n"
    "  v_color = vec4(color, color, color, 1);\n"
    "}\n";

static const char kFragmentShader[] = 
	"varying vec4 v_color;\n"
    "void main() {\n"
    "  gl_FragColor = v_color;\n"
    "}\n";

static const glm::mat4 inverse_z_mat = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
                                                 0.0f, -1.0f, 0.0f, 0.0f,
                                                 0.0f, 0.0f, -1.0f, 0.0f,
                                                 0.0f, 0.0f, 0.0f, 1.0f);

Pointcloud::Pointcloud() {
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  shader_program_ = GlUtil::CreateProgram(kVertexShader, kFragmentShader);
  if (!shader_program_) {
    LOGE("Could not create program.");
  }
  uniform_mvp_mat_ = glGetUniformLocation(shader_program_, "mvp");
  attrib_vertices_ = glGetAttribLocation(shader_program_, "vertex");
  attrib_colors_ = glGetAttribLocation(shader_program_, "color");
  glGenBuffers(1, &vertex_buffers_);
  glGenBuffers(1, &color_buffers_);
}

void Pointcloud::Render(glm::mat4 projection_mat, glm::mat4 view_mat,
                        glm::mat4 model_mat, int depth_buffer_size,
                        float *depth_data_buffer, float *color_data_buffer) {
  glUseProgram(shader_program_);

  // Lock xyz_ij mutex.
  pthread_mutex_lock(&TangoData::GetInstance().xyzij_mutex);

  // Calculate model view projection matrix.
  glm::mat4 mvp_mat = projection_mat * view_mat * model_mat * inverse_z_mat;
  glUniformMatrix4fv(uniform_mvp_mat_, 1, GL_FALSE, glm::value_ptr(mvp_mat));

  // Bind vertex buffer.
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffers_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * depth_buffer_size,
               depth_data_buffer, GL_STATIC_DRAW);
  
  glEnableVertexAttribArray(attrib_vertices_);
  glVertexAttribPointer(attrib_vertices_, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  // Bind color buffer. 
  glBindBuffer(GL_ARRAY_BUFFER, color_buffers_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * depth_buffer_size / 3,
               color_data_buffer, GL_STATIC_DRAW);

  glEnableVertexAttribArray(attrib_colors_);
  glVertexAttribPointer(attrib_colors_, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  // Render.
  glDrawArrays(GL_POINTS, 0, 3 * depth_buffer_size);

  // Unlock xyz_ij mutex.
  pthread_mutex_unlock(&TangoData::GetInstance().xyzij_mutex);

  GlUtil::CheckGlError("glDrawArray()");
  glUseProgram(0);
  GlUtil::CheckGlError("glUseProgram()");
}
