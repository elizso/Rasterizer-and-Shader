//
// Created by Eliz So on 24/3/2026.
//


#ifndef RASTERIZER_TASK2_H
#define RASTERIZER_TASK2_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Eigen/Dense"
// #include "Eigen/src/Core/Matrix.h"
#include "OBJ_Loader.h"
#include "Shader.hpp"
#include "Texture.hpp"
#include "Triangle.hpp"
#include "global.hpp"
#include "rasterizer.hpp"

using namespace Eigen;
using namespace std;
Eigen::Matrix4f look_at(Eigen::Vector3f eye_pos, Eigen::Vector3f target,
                        Eigen::Vector3f up = Eigen::Vector3f(0, 1, 0));
Vector3f normal_fragment_shader(const fragment_shader_payload &payload);
Vector3f blinn_phong_fragment_shader(const fragment_shader_payload &payload);
Vector3f texture_fragment_shader(const fragment_shader_payload &payload);
Vector3f bump_fragment_shader(const fragment_shader_payload &payload);
Vector3f vertex_shader(const vertex_shader_payload &payload);
Matrix4f get_model_matrix(float rotation_angle, const Vector3f &axis, const Vector3f &translation);
Matrix4f get_view_matrix(Vector3f eye_pos);
Matrix4f get_projection_matrix(float eye_fovy, float aspect_ratio, float zNear, float zFar);


#endif //RASTERIZER_TASK2_H

