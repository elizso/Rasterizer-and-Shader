//
// Created by Eliz So on 27/3/2026.
//

#ifndef RASTERIZER_TASK3_UTILS_H
#define RASTERIZER_TASK3_UTILS_H
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Eigen/Dense"
// #include "Eigen/src/Core/Matrix.h"
#include "Shader.hpp"
#include "Texture.hpp"
#include "Triangle.hpp"
#include "global.hpp"
#include "rasterizer.hpp"


Eigen::Matrix4f get_rotation(float rotation_angle, const Eigen::Vector3f &axis);
Eigen::Matrix4f get_translation(const Eigen::Vector3f &translation);
Eigen::Matrix4f look_at(Eigen::Vector3f eye_pos, Eigen::Vector3f target,
                        Eigen::Vector3f up = Vector3f(0, 1, 0));
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos);
Eigen::Matrix4f get_model_matrix(float rotation_angle, const Eigen::Vector3f &axis,
                                 const Eigen::Vector3f &translation);
Eigen::Matrix4f get_model_matrix(float rotation_angle, const Eigen::Vector3f &axis,const Vector3f &translation, float scale);
Eigen::Matrix4f get_projection_matrix(float eye_fovy, float aspect_ratio, float zNear, float zFar);
Eigen::Vector3f vertex_shader(const vertex_shader_payload &payload);
Eigen::Vector3f reflect(const Eigen::Vector3f &vec, const Eigen::Vector3f &axis);
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload &payload);
Eigen::Vector3f blinn_phong_fragment_shader(const fragment_shader_payload &payload);
Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload &payload);
Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload &payload) ;
Eigen::Vector3f triangle_normal(Vector3f v0, Vector3f v1, Vector3f v2);



#endif //RASTERIZER_TASK3_UTILS_H