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

Eigen::Vector3f triangle_normal(Eigen::Vector3f v0, Eigen::Vector3f v1, Eigen::Vector3f v2)
{
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    Vector3f normal = e1.cross(e2).normalized();
    return normal;
}

Eigen::Matrix4f get_rotation(float rotation_angle, const Eigen::Vector3f &axis) {
    // Calculate a rotation matrix from rotation axis and angle.
    // Note: rotation_angle is in degree.
    Eigen::Matrix4f rotation_matrix = Eigen::Matrix4f::Identity();

    float rotation_angle_rad = rotation_angle * MY_PI / 180.0;
    float cos_theta = cos(rotation_angle_rad);
    float sin_theta = sin(rotation_angle_rad);

    Eigen::Vector3f axis_ = axis.normalized();
    Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f ux;
    ux << 0, -axis_.z(), axis_.y(), axis_.z(), 0, -axis_.x(), -axis_.y(), axis_.x(), 0;

    Eigen::Matrix3f rotation_matrix_3x3 =
        cos_theta * identity + (1 - cos_theta) * (axis_ * axis_.transpose()) + sin_theta * ux;
    rotation_matrix.block<3, 3>(0, 0) = rotation_matrix_3x3;

    return rotation_matrix;
}

Eigen::Matrix4f get_translation(const Eigen::Vector3f &translation) {
    // Calculate a transformation matrix of given translation vector.
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans(0, 3) = translation.x();
    trans(1, 3) = translation.y();
    trans(2, 3) = translation.z();
    return trans;
}

Eigen::Matrix4f look_at(Eigen::Vector3f eye_pos, Eigen::Vector3f target,
                        Eigen::Vector3f up = Eigen::Vector3f(0, 1, 0)) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Vector3f z = (eye_pos - target).normalized();
    Eigen::Vector3f x = up.cross(z).normalized();
    Eigen::Vector3f y = z.cross(x).normalized();

    Eigen::Matrix4f rotate;
    rotate << x.x(), x.y(), x.z(), 0, y.x(), y.y(), y.z(), 0, z.x(), z.y(), z.z(), 0, 0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1, -eye_pos[2], 0, 0, 0, 1;

    view = rotate * translate * view;
    return view;
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    view = look_at(eye_pos, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 1, 0));

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle, const Eigen::Vector3f &axis,
                                 const Eigen::Vector3f &translation) {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f rotation = get_rotation(rotation_angle, axis);

    Eigen::Matrix4f trans = get_translation(translation);

    model = trans * rotation * model;
    return model;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle, const Eigen::Vector3f &axis,const Vector3f &translation, float scale)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f scale_mat = Eigen::Matrix4f::Identity();
    scale_mat(0,0) = scale;
    scale_mat(1,1) = scale;
    scale_mat(2,2) = scale;

    Eigen::Matrix4f rotation = get_rotation(rotation_angle, axis);
    Eigen::Matrix4f trans = get_translation(translation);

    model = trans * rotation * scale_mat;
    return model;

}

Eigen::Matrix4f get_projection_matrix(float eye_fovy, float aspect_ratio, float zNear, float zFar) {
    // Create the projection matrix for the given parameters.
    // Then return it.
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    float eye_fovy_rad = eye_fovy * MY_PI / 180.0;
    float top = zNear * tan(eye_fovy_rad / 2.0);
    float bottom = -top;
    float right = top * aspect_ratio;
    float left = -right;

    projection << zNear / right, 0, 0, 0, 0, zNear / top, 0, 0, 0, 0, (zNear + zFar) / (zNear - zFar),
        2 * zNear * zFar / (zNear - zFar), 0, 0, -1, 0;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload &payload) {
    return payload.position;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f &vec, const Eigen::Vector3f &axis) {
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

// TODO: Task2 Implement the following fragment shaders
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload &payload) {

    // convert nomral vector from [-1, 1] to [0, 1] and then to [0, 255]
    
    Eigen::Vector3f normal = payload.normal.normalized();
    if (normal.norm() < 1e-6f) {
        return Eigen::Vector3f(255, 0, 0); // red = broken normals
    }

    normal.normalize();
    return (normal + Eigen::Vector3f(1,1,1)) * 0.5f * 255.0f;

}

// TODO: Task2 Implement the following fragment shaders
Eigen::Vector3f blinn_phong_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{10, 10, 10};

    float p = 150; // sharpness 
    

    auto lights = payload.view_lights;
    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal.normalized();
    Eigen::Vector3f result_color = {0, 0, 0};

    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);  // cwiseProduct--dot product
    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *diffuse*, and
        // *specular* components are. Then, accumulate that result on the *result_color*
        // object.
        auto light_dir = light.position - point;
        auto l = light_dir.normalized();
        auto v = -point.normalized();
        auto h = (v + l).normalized();
        float r2 = light_dir.dot(light_dir);

        auto Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.dot(l));
        auto Ls = ks.cwiseProduct(light.intensity / r2) * pow(std::max(0.0f, normal.dot(h)), p);
        result_color += Ld + Ls;
    }
    result_color += La;
    result_color = result_color.cwiseMin(1.0f).cwiseMax(0.0f);

    return result_color * 255.f;
}

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture) {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{10, 10, 10};

    float p = 150;
    float radius = 10;

    std::vector<light> lights = payload.view_lights;
    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal.normalized();

    Eigen::Vector3f result_color = {0, 0, 0};
    Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity);  // cwiseProduct--dot product

    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *ambient*,
        // *diffuse*, and *specular* components are. Then, accumulate that result on the
        // *result_color* object.
        auto light_dir = light.position - point;
        auto l = light_dir.normalized();
        auto v = -point.normalized();
        auto h = (v + l).normalized();
        float r2 = light_dir.dot(light_dir);

        auto Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.dot(l));
        auto Ls = ks.cwiseProduct(light.intensity / r2) * pow(std::max(0.0f, normal.dot(h)), p);
        result_color += Ld + Ls;
    }
    result_color += La;
    result_color = result_color.cwiseMin(1.0f).cwiseMax(0.0f);

    return result_color * 255.f;
}

// TODO: Task2 Implement the following fragment shaders
Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{10, 10, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal.normalized();
    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (f(u+1/w,v)-f(u,v))
    // dV = kh * kn * (f(u,v+1/h)-f(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
    // You can implement the function f as f(u,v) = norm of payload.texture->getColor(u, v) 
    Eigen::Vector3f n = normal.normalized();
    Eigen::Vector3f t = Eigen::Vector3f(n.x() * n.y() / (sqrt(n.x() * n.x() + n.z() * n.z())), sqrt(n.x() * n.x() + n.z() * n.z()),
                         n.z() * n.y() / (sqrt(n.x() * n.x() + n.z() * n.z())));
    Eigen::Vector3f b = n.cross(t);
    Eigen::Matrix3f TBN;
    TBN << t.x(), b.x(), n.x(), t.y(), b.y(), n.y(), t.z(), b.z(), n.z();
    float fuv = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y()).norm();
    float fuv_u = payload.texture->getColor(payload.tex_coords.x() + 1.0f / payload.texture->width, payload.tex_coords.y()).norm();
    float fuv_v = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y() + 1.0f / payload.texture->height).norm();
    float dU = kh * kn * (fuv_u - fuv);
    float dV = kh * kn * (fuv_v - fuv);
    Eigen::Vector3f ln = Eigen::Vector3f(-dU, -dV, 1);
    n = (TBN * ln).normalized();

    Eigen::Vector3f result_color = n;
    // result_color = n;
    result_color = result_color.cwiseMin(1.0f).cwiseMax(0.0f);

    return result_color * 255.f;
}