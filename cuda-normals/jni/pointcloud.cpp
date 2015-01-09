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
	"uniform mat4 mvp;\n"
	"varying vec4 v_color;\n"
	"void main() {\n"
	"  gl_Position = mvp*vertex;\n"
	"  v_color = vertex;\n"
	"}\n";


static const char kFragmentShader[] = 
	"varying vec4 v_color;\n"
    "void main() {\n"
    "  gl_FragColor = vec4(v_color);\n"
    "}\n";

static const glm::mat4 inverse_z_mat = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
                                                 0.0f, -1.0f, 0.0f, 0.0f,
                                                 0.0f, 0.0f, -1.0f, 0.0f,
                                                 0.0f, 0.0f, 0.0f, 1.0f);

Pointcloud::Pointcloud() {
  //glLineWidth(2.0f);

  shader_program_ = GlUtil::CreateProgram(kVertexShader, kFragmentShader);
  if (!shader_program_) {
    LOGE("Could not create program.");
  }
  uniform_mvp_mat_ = glGetUniformLocation(shader_program_, "mvp");
  attrib_vertices_ = glGetAttribLocation(shader_program_, "vertex");
  glGenBuffers(1, &vertex_buffers_);

  uint32_t max_vertices = 70000; // magic number for test

  // hold twice the vertices of the depth buffer for lines.
  normal_data_buffer = new float[6 * max_vertices];
}

Pointcloud::~Pointcloud() {
	delete[] normal_data_buffer;
}

void Pointcloud::Render(glm::mat4 projection_mat, glm::mat4 view_mat,
                        glm::mat4 model_mat, int depth_buffer_size,
                        float *depth_data_buffer, float *normal_buffer) {
  glUseProgram(shader_program_);

  // Lock xyz_ij mutex.
  pthread_mutex_lock(&TangoData::GetInstance().xyzij_mutex);

  // Calculate model view projection matrix.
  glm::mat4 mvp_mat = projection_mat * view_mat * model_mat * inverse_z_mat;
  glUniformMatrix4fv(uniform_mvp_mat_, 1, GL_FALSE, glm::value_ptr(mvp_mat));

  int iter, counter = 0;
  for (iter = 0; iter < depth_buffer_size; iter += 3) {
	memcpy(normal_data_buffer + counter, depth_data_buffer + iter, 3 * sizeof(float));
	// add normals to value of line to draw increase order of magnitude of vectors
	// so that they are visible
	depth_data_buffer[iter+0] += normal_buffer[iter+0] * 100;
	depth_data_buffer[iter+1] += normal_buffer[iter+1] * 100;
	depth_data_buffer[iter+2] += normal_buffer[iter+2] * 100;
	memcpy(normal_data_buffer + 3 + counter, depth_data_buffer + iter, 3 * sizeof(float));
	counter += 6;
  }

  // Bind vertex buffer.
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffers_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * depth_buffer_size * 2,
               normal_data_buffer, GL_STATIC_DRAW);

  glEnableVertexAttribArray(attrib_vertices_);
  glVertexAttribPointer(attrib_vertices_, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glDrawArrays(GL_LINES, 0, depth_buffer_size);

  // Unlock xyz_ij mutex.
  pthread_mutex_unlock(&TangoData::GetInstance().xyzij_mutex);

  GlUtil::CheckGlError("glDrawArray()");
  glUseProgram(0);
  GlUtil::CheckGlError("glUseProgram()");
}
