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

#include "tango-gl-renderer/plane.h"

static const char kVertexShader[] = "attribute vec4 vertex;\n"
    "uniform mat4 mvp;\n"
    "void main() {\n"
    "  gl_Position = mvp*vertex;\n"
    "}\n";

static const char kFragmentShader[] =
    "void main() {\n"
    "  gl_FragColor = vec4(0.85f,0.85f,0.85f,1);\n"
    "}\n";

static const float vertices[] = {
  	  -5.0f, 0.0f, -5.0f,
  	  -5.0f, 0.0f, 5.0f,
  	  5.0f, 0.0f, 5.0f,
  	  -5.0f, 0.0f, -5.0f,
  	  -5.0f, 0.0f, 5.0f,
  	  5.0f, 0.0f, 5.0f
};

Plane::Plane() {
  shader_program_ = GlUtil::CreateProgram(kVertexShader, kFragmentShader);
  if (!shader_program_) {
    LOGE("Could not create program.");
  }
  
  uniform_mvp_mat_ = glGetUniformLocation(shader_program_, "mvp");
  attrib_vertices_ = glGetAttribLocation(shader_program_, "vertex");
  traverse_len_ = 15;

  // Binding vertex buffer object.
  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * traverse_len_, vertices,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Plane::~Plane() {
  glDeleteShader(shader_program_);
  delete[] vertices_;
}

void Plane::Render(const glm::mat4& projection_mat,
                  const glm::mat4& view_mat) const {
  glUseProgram(shader_program_);

  // Calculate model view projection matrix.
  glm::mat4 model_mat = GetTransformationMatrix();
  glm::mat4 mvp_mat = projection_mat * view_mat * model_mat;
  glUniformMatrix4fv(uniform_mvp_mat_, 1, GL_FALSE, glm::value_ptr(mvp_mat));

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  int buffer_size = 0;
  glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &buffer_size);
  buffer_size = buffer_size / sizeof(float);

  glEnableVertexAttribArray(attrib_vertices_);
  glVertexAttribPointer(attrib_vertices_, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glDrawArrays(GL_TRIANGLES, 0, buffer_size);
  GlUtil::CheckGlError("grid glDrawArray()");
  glDisableVertexAttribArray(attrib_vertices_);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glUseProgram(0);
  GlUtil::CheckGlError("glUseProgram()");
}


