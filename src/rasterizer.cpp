
#include "rasterizer.hpp"

#include <math.h>

#include <algorithm>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

#include "Shader.hpp"
#include "Triangle.hpp"

using namespace Eigen;
using namespace std;

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f> &normals) {
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}

void rst::rasterizer::post_process_buffer() {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = get_index(x, y);
            for (int i = 0; i < 4; i++) {
                frame_buf[index] += ssaa_frame_buf[4 * index + i];
            }
            frame_buf[index] /= 4;
        }
    }
}

// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end) {
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

    dx = x2 - x1;
    dy = y2 - y1;
    dx1 = fabs(dx);
    dy1 = fabs(dy);
    px = 2 * dy1 - dx1;
    py = 2 * dx1 - dy1;

    if (dy1 <= dx1) {
        if (dx >= 0) {
            x = x1;
            y = y1;
            xe = x2;
        } else {
            x = x2;
            y = y2;
            xe = x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; x < xe; i++) {
            x = x + 1;
            if (px < 0) {
                px = px + 2 * dy1;
            } else {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
                    y = y + 1;
                } else {
                    y = y - 1;
                }
                px = px + 2 * (dy1 - dx1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    } else {
        if (dy >= 0) {
            x = x1;
            y = y1;
            ye = y2;
        } else {
            x = x2;
            y = y2;
            ye = y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; y < ye; i++) {
            y = y + 1;
            if (py <= 0) {
                py = py + 2 * dx1;
            } else {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
                    x = x + 1;
                } else {
                    x = x - 1;
                }
                py = py + 2 * (dx1 - dy1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector4f *_v) {
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f p(x, y, 1.);
    Vector3f f0, f1, f2;
    f0 = (p - v[0]).cross(v[1] - v[0]);
    f1 = (p - v[1]).cross(v[2] - v[1]);
    f2 = (p - v[2]).cross(v[0] - v[2]);
    if (f0.dot(f1) > 0 && f1.dot(f2) > 0)
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
               (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() -
                v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
               (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() -
                v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
               (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() -
                v[1].x() * v[0].y());
    return {c1, c2, c3};
}

// Task1 Implement this function
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type, bool culling, bool anti_aliasing) {
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;



    Eigen::Matrix4f mvp = projection * view * model;
    for (auto &i : ind) {
        Triangle t;

        std::array<Eigen::Vector4f, 3> mm{view * model * to_vec4(buf[i[0]], 1.0f),
                                          view * model * to_vec4(buf[i[1]], 1.0f),
                                          view * model * to_vec4(buf[i[2]], 1.0f)};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v) { return v.template head<3>(); });


        // Task1 Enable back face culling
        if (culling) {

            Vector3f v_0 = viewspace_pos[0];
            Vector3f v_1 = viewspace_pos[1];
            Vector3f v_2 = viewspace_pos[2];

            Vector3f e1 = v_1 - v_0;
            Vector3f e2 = v_2 - v_0;
            
            Vector3f normal = e1.cross(e2);
            Vector3f view_dir = -v_0;
            if (normal.dot(view_dir) <= 0) {
                continue;
            }

        }

        Eigen::Vector4f v[] = {mvp * to_vec4(buf[i[0]], 1.0f), mvp * to_vec4(buf[i[1]], 1.0f),
                               mvp * to_vec4(buf[i[2]], 1.0f)};

        // Homogeneous division
        for (auto &vec : v) {
            vec /= vec.w();
        }
        // Viewport transformation
        for (auto &vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i]);
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);
/*************** update below *****************/
        rasterize_triangle(t, anti_aliasing);
    }
    if (anti_aliasing){
        post_process_buffer();
    }
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList, bool culling, rst::Shading shading, bool shadow) {
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;

    std::vector<light> viewspace_lights;
    for (const auto &l : lights) {
        light view_space_light;
        view_space_light.position = (view * to_vec4(l.position, 1.0f)).head<3>();
        view_space_light.intensity = l.intensity;
        viewspace_lights.push_back(view_space_light);
    }

    for (const auto &t : TriangleList) {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{(view * model * t->v[0]), (view * model * t->v[1]), (view * model * t->v[2])};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v) { return v.template head<3>(); });

        std::array<Eigen::Vector4f, 3> wm{
            (model * t->v[0]),
            (model * t->v[1]),
            (model * t->v[2])
        };

        std::transform(wm.begin(), wm.end(), worldspace_pos.begin(),
                       [](auto &v) { return v.template head<3>(); });

        // Task1 Enable back face culling
        if (culling) {
            Vector3f v_0 = viewspace_pos[0];
            Vector3f v_1 = viewspace_pos[1];
            Vector3f v_2 = viewspace_pos[2];

            Vector3f e1 = v_1 - v_0;
            Vector3f e2 = v_2 - v_0;
            
            Vector3f normal = e1.cross(e2);
            Vector3f view_dir = -v_0;
            if (normal.dot(view_dir) <= 0) {
                continue;
            }
        }

        Eigen::Vector4f v[] = {mvp * t->v[0], mvp * t->v[1], mvp * t->v[2]};
        // Homogeneous division
        for (auto &vec : v) {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {inv_trans * to_vec4(t->normal[0], 0.0f), inv_trans * to_vec4(t->normal[1], 0.0f),
                               inv_trans * to_vec4(t->normal[2], 0.0f)};

        // Viewport transformation
        for (auto &vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            // screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i) {
            // view space normal
            newtri.setNormal(i, n[i].head<3>());
        }



        // Also pass view space vertice position
        if (!shadow)
            rasterize_triangle(newtri, viewspace_pos, viewspace_lights, shading);
        else
            rasterize_triangle_shadow(newtri, viewspace_pos, {worldspace_pos[0], worldspace_pos[1], worldspace_pos[2]}, viewspace_lights);

    }
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f &vert1,
                                   const Eigen::Vector3f &vert2, const Eigen::Vector3f &vert3, float weight) {
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f &vert1,
                                   const Eigen::Vector2f &vert2, const Eigen::Vector2f &vert3, float weight) {
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

// Task1 Implement this function
void rst::rasterizer::rasterize_triangle(const Triangle &t, bool anti_aliasing) {
    auto v = t.toVector4();

    // 1. Find out the bounding box of current triangle.
    int min_x = (int)floor(v[0].x()), max_x = (int)ceil(v[0].x());
    int min_y = (int)floor(v[0].y()), max_y = (int)ceil(v[0].y());

    for (int i = 1; i < 3; ++i) {
        min_x = min(min_x, (int)floor(v[i].x())), min_y = min(min_y, (int)floor(v[i].y()));
        max_x = max(max_x, (int)ceil(v[i].x())), max_y = max(max_y, (int)ceil(v[i].y()));
    }

    min_x = max(0, min_x), min_y = max(0, min_y);
    max_x = min(width - 1, max_x), max_y = min(height - 1, max_y);

    // 2. Iterate through the pixel and find if the current pixel is inside the triangle
    // Subpixel sampling if do anti-aliasing
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {

            if (anti_aliasing){
                std::array<Vector2f, 4> offsets = {
                    Vector2f(0.25f, 0.25f),Vector2f(0.75f, 0.25f),
                    Vector2f(0.25f, 0.75f),Vector2f(0.75f, 0.75f)
                };

                int pixel_index = get_index(x, y);
                for (int i = 0; i < 4; ++i){
                    float ix = x + offsets[i].x(), iy = y + offsets[i].y();
                    if (!insideTriangle(ix, iy, t.v)) continue;

                    auto[alpha, beta, gamma] = computeBarycentric2D(ix, iy, t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    int ssaa_index = 4 * pixel_index + i;
                    if (z_interpolated < ssaa_depth_buf[ssaa_index]) {
                        ssaa_depth_buf[ssaa_index] = z_interpolated;
                        Vector3f interpolated_color = 255*(alpha * t.color[0] + beta * t.color[1] + gamma * t.color[2]);
                        ssaa_frame_buf[ssaa_index] = interpolated_color;
                    }
                }
            }

            else if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                int ind = get_index(x, y);

                if (z_interpolated < depth_buf[ind]) {
                    depth_buf[ind] = z_interpolated;
                    Vector3f interpolated_color = 255*(alpha * t.color[0] + beta * t.color[1] + gamma * t.color[2]);
                    set_pixel(Vector2i(x, y), interpolated_color);
                }
            }
            
        }
    }
 
    


    // 3. If so, use the following code to get the interpolated z value.
    // auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
    // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    // z_interpolated *= w_reciprocal;

    // 4. Check if the interpolated depth is closer than the value in the depth buffer.
    
    // 5. If so, update the depth buffer and set the current pixel to the color of the interpolated color of vertices
    // set_pixel(Vector2i(x, y), interpolated_color);
}

// Task2 Implement this function
void rst::rasterizer::rasterize_triangle(const Triangle &t, const std::array<Eigen::Vector3f, 3> &view_pos,
                                         const std::vector<light> &view_lights, rst::Shading shading, bool shadow) {
    auto v = t.toVector4();

    // Find the bounding box of the triangle.
        int min_x = (int)floor(v[0].x());
        int max_x = (int)ceil(v[0].x());
        int min_y = (int)floor(v[0].y());
        int max_y = (int)ceil(v[0].y());
        for (int i = 1; i < 3; ++i) {
            min_x = min(min_x, (int)floor(v[i].x()));
            min_y = min(min_y, (int)floor(v[i].y()));
            max_x = max(max_x, (int)ceil(v[i].x()));
            max_y = max(max_y, (int)ceil(v[i].y()));
        }
        min_x = max(0, min_x);
        min_y = max(0, min_y);
        max_x = min(width - 1, max_x);
        max_y = min(height - 1, max_y);


    if (shading == rst::Shading::Flat) {

        // Find the normal of the triangle 
        Vector3f e1 = view_pos[1] - view_pos[0];
        Vector3f e2 = view_pos[2] - view_pos[0];
        Vector3f normal = e1.cross(e2).normalized();

        Vector3f color = 255*(t.color[0] + t.color[1] + t.color[2]) / 3.0f;
        Vector2f tex_coords = (t.tex_coords[0] + t.tex_coords[1] + t.tex_coords[2]) / 3.0f;
        Vector3f shading_coords = (view_pos[0] + view_pos[1] + view_pos[2]) / 3.0f;

        fragment_shader_payload payload(color, normal, tex_coords, view_lights, texture ? &*texture : nullptr);
        payload.view_pos = shading_coords;
        auto pixel_color = fragment_shader(payload);

        for (int x = min_x; x <= max_x; x++) {
            for (int y = min_y; y <= max_y; y++) {
                if (!insideTriangle(x + 0.5, y + 0.5, t.v)) continue;

                auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5f, y + 0.5f, t.v);

                float w_reciprocal =
                    1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());

                float z_interpolated =
                    alpha * v[0].z() / v[0].w() +
                    beta  * v[1].z() / v[1].w() +
                    gamma * v[2].z() / v[2].w();

                z_interpolated *= w_reciprocal;

                int ind = get_index(x, y);

                if (depth_buf[ind] > z_interpolated) {
                    depth_buf[ind] = z_interpolated;

                    set_pixel(Vector2i(x, y), pixel_color);
                }

            }
        }

    } else if (shading == rst::Shading::Gouraud) {

        std::vector<Vector3f> vector_colours = {};
        for (int i = 0; i < 3; i++){
            Vector3f normal = t.normal[i].normalized();
            Vector3f color = 255*t.color[i];
            Vector2f tex_coords = t.tex_coords[i];
            Vector3f shading_coords = view_pos[i];

            fragment_shader_payload payload(color, normal, tex_coords, view_lights, texture ? &*texture : nullptr);
            payload.view_pos = shading_coords;
            auto vertex_color = fragment_shader(payload);
            vector_colours.push_back(vertex_color);
        }
        for (int x = min_x; x <= max_x; x++) {
            for (int y = min_y; y <= max_y; y++) {
                if (!insideTriangle(x + 0.5, y + 0.5, t.v)) continue;

                auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma /v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() /
                v[2].w(); 
                z_interpolated *= w_reciprocal;

                int ind = get_index(x, y);

                if (z_interpolated < depth_buf[ind]) {
                    depth_buf[ind] = z_interpolated;
                    float weight = alpha + beta + gamma;
                    Vector3f interpolated_color = (alpha * vector_colours[0] + beta * vector_colours[1] + gamma * vector_colours[2]) / weight;

                    set_pixel(Vector2i(x, y), interpolated_color);
                }
            }
        }
        
    } else if (shading == rst::Shading::Phong) {
        
        // iterate through the pixel and find if the current pixel is inside the
        // triangle If so, use the following code to get the interpolated z value.
        // auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
        // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma /v[2].w());
        // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() /
        // v[2].w(); z_interpolated *= w_reciprocal;
        for (int x = min_x; x <= max_x; x++) {
            for (int y = min_y; y <= max_y; y++) {
                if (!insideTriangle(x + 0.5, y + 0.5, t.v)) continue;

                auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma /v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() /
                v[2].w(); 
                z_interpolated *= w_reciprocal;

                int ind = get_index(x, y);

                if (z_interpolated < depth_buf[ind]) {
                    depth_buf[ind] = z_interpolated;
                    float weight = alpha + beta + gamma;
                    Vector3f interpolated_color = 255*(alpha * t.color[0] + beta * t.color[1] + gamma * t.color[2]) / weight;
                    Vector3f interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], weight).normalized();
                    Vector2f interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], weight);
                    Vector3f interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], weight);


                    fragment_shader_payload payload(interpolated_color, interpolated_normal, interpolated_texcoords,
                                                        view_lights, texture ? &*texture : nullptr);
                    payload.view_pos = interpolated_shadingcoords;

                    auto pixel_color = fragment_shader(payload);

                    set_pixel(Vector2i(x, y), pixel_color);
                }
            }
        }


    }
}

//TODO : Task 3
void rst::rasterizer::rasterize_triangle_shadow(const Triangle& t,
                               const std::array<Eigen::Vector3f, 3>& view_pos,
                               const std::array<Eigen::Vector3f, 3>& world_pos,
                               const std::vector<light>& view_lights)
{
    auto v = t.toVector4();
    const float bias = 0.001f;

    int min_x = max(0,         (int)floor(min({v[0].x(), v[1].x(), v[2].x()})));
    int max_x = min(width - 1, (int)ceil (max({v[0].x(), v[1].x(), v[2].x()})));
    int min_y = max(0,         (int)floor(min({v[0].y(), v[1].y(), v[2].y()})));
    int max_y = min(height - 1,(int)ceil (max({v[0].y(), v[1].y(), v[2].y()})));

    min_x = max(0, min_x);
    min_y = max(0, min_y);
    max_x = min(width - 1,  max_x);
    max_y = min(height - 1, max_y);

    if (min_x > max_x || min_y > max_y) return;

    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            if (!insideTriangle(x + 0.5f, y + 0.5f, t.v)) continue;

            auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5f, y + 0.5f, t.v);
            float w_reciprocal = 1.0f / (alpha/v[0].w() + beta/v[1].w() + gamma/v[2].w());
            float z_interpolated = (alpha*v[0].z()/v[0].w()
                                  + beta *v[1].z()/v[1].w()
                                  + gamma*v[2].z()/v[2].w()) * w_reciprocal;

            int ind = get_index(x, y);
            if (z_interpolated >= depth_buf[ind]) continue;
            depth_buf[ind] = z_interpolated;

            float weight = alpha + beta + gamma;

            // shading
            Vector3f interpolated_color  = 255*(alpha*t.color[0]   + beta*t.color[1]   + gamma*t.color[2])  / weight;
            Vector3f interpolated_normal = (alpha*t.normal[0]  + beta*t.normal[1]  + gamma*t.normal[2]).normalized();
            Vector2f interpolated_uv     = (alpha*t.tex_coords[0] + beta*t.tex_coords[1] + gamma*t.tex_coords[2]) / weight;
            Vector3f interpolated_vpos   = (alpha*view_pos[0]  + beta*view_pos[1]  + gamma*view_pos[2]) / weight;

            fragment_shader_payload payload(interpolated_color, interpolated_normal,
                                            interpolated_uv, view_lights,
                                            texture ? &*texture : nullptr);
            payload.view_pos = interpolated_vpos;
            auto pixel_color = fragment_shader(payload);

            // known to be flat
            float table_y = world_pos[0].y();

            // find NDC
            float ndc_x = (x + 0.5f) / (0.5f * width)  - 1.0f;
            float ndc_y = (y + 0.5f) / (0.5f * height) - 1.0f;

            Matrix4f inv_pv = (projection * view).inverse();
            Vector4f near_pt = inv_pv * Vector4f(ndc_x, ndc_y, -1.0f, 1.0f);
            Vector4f far_pt  = inv_pv * Vector4f(ndc_x, ndc_y,  1.0f, 1.0f);
            near_pt /= near_pt.w();
            far_pt  /= far_pt.w();

            // Ray from near to far
            Vector3f ray_origin = near_pt.head<3>();
            Vector3f ray_dir    = (far_pt.head<3>() - ray_origin).normalized();

            // Intersect ray with horizontal plane y = table_y
            float t_hit = (table_y - ray_origin.y()) / ray_dir.y();

            if (ray_dir.y() == 0.0f || t_hit < 0.0f) {
                set_pixel(Vector2i(x, y), pixel_color);  // no shadow, just write color
                continue;
            }
            Vector3f frag_world = ray_origin + t_hit * ray_dir;

            // shadow lookup
            for (int li = 0; li < (int)light_projections.size(); li++) {
                Vector4f fl = light_projections[li] * shadow_views[li] * to_vec4(frag_world, 1.0f);
                fl /= fl.w();

                int sx = (int)(0.5f * width  * (fl.x() + 1.0f));
                int sy = (int)(0.5f * height * (fl.y() + 1.0f));
                float sz = fl.z();

                int kernel = 5;
                float texel = 2.0f;
                float shadow_sum = 0;
                int shadow_total = 0;

                for (int dx = -kernel; dx <= kernel; dx++)
                {
                    for (int dy = -kernel; dy <= kernel; dy++)
                    {
                        int ix = (int) (sx + dx * texel);
                        int iy = (int) (sy + dy * texel);
                        shadow_total++;

                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            float stored = shadow_bufs[li][get_index(ix, iy)];
                            if (stored < std::numeric_limits<float>::infinity()
                                && sz > stored + bias) {
                                shadow_sum++;
                            }
                        }
                    }
                }

                float shadow_factor = shadow_sum / shadow_total;

                pixel_color *= (1-0.7 * shadow_factor);


            }

            if (x < 0 || x >= width || y < 0 || y >= height) continue;

            set_pixel(Vector2i(x, y), pixel_color);
        }
    }

}

//TODO: Task 3
void rst::rasterizer::shadow_mapping(const vector<Triangle *> triangles, int light_ind)
{

    auto light_mvp = light_projections[light_ind] * shadow_views[light_ind] * model;

    int total = 0, skipped = 0;

    for (int i = 0; i < triangles.size(); i++) {
        Vector4f v[3] = {
            light_mvp * triangles[i]->v[0],
            light_mvp * triangles[i]->v[1],
            light_mvp * triangles[i]->v[2]
        };
        total++;

        // find NDC
        for (int j = 0; j < 3; j++)
            v[j] /= v[j].w();

        // viewport transform
        for (int j = 0; j < 3; j++) {
            v[j].x() = 0.5f * width  * (v[j].x() + 1.0f);
            v[j].y() = 0.5f * height * (v[j].y() + 1.0f);
        }

        // bounding box
        int min_x = max(0,         (int)floor(min({v[0].x(), v[1].x(), v[2].x()})));
        int max_x = min(width - 1, (int)ceil (max({v[0].x(), v[1].x(), v[2].x()})));
        int min_y = max(0,         (int)floor(min({v[0].y(), v[1].y(), v[2].y()})));
        int max_y = min(height- 1, (int)ceil (max({v[0].y(), v[1].y(), v[2].y()})));

        if (max_x < min_x || max_y < min_y) { skipped++; continue; }

        for (int x = min_x; x <= max_x; x++) {
            for (int y = min_y; y <= max_y; y++) {
                if (!insideTriangle(x + 0.5f, y + 0.5f, v)) continue;

                auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5f, y + 0.5f, v);

                float z = alpha * v[0].z() + beta * v[1].z() + gamma * v[2].z();

                int ind = get_index(x, y);
                if (z < shadow_bufs[light_ind][ind])
                {
                    shadow_bufs[light_ind][ind] = z;
                }

            }
        }
    }

}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) {
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) {
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
    projection = p;
}

void rst::rasterizer::set_lights(const std::vector<light> &l) {
    lights = l;
}

void rst::rasterizer::set_shadow_view(const Eigen::Matrix4f &v) {
    shadow_views.push_back(v);
}

void rst::rasterizer::set_shadow_buffer(const std::vector<float> &shadow_buffer) {
    shadow_bufs.push_back(shadow_buffer);
}

void rst::rasterizer::clear(rst::Buffers buff) {
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(ssaa_frame_buf.begin(), ssaa_frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(ssaa_depth_buf.begin(), ssaa_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    ssaa_frame_buf.resize(4 * w * h);
    ssaa_depth_buf.resize(4 * w * h);
    texture = std::nullopt;
}

float rst::rasterizer::distance(Vector3f x1, Vector3f x2)
{
    float dx = x1.x() - x2.x();
    float dy = x1.y() - x2.y();
    float dz = x1.z() - x2.z();
    return sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
}

int rst::rasterizer::get_index(int x, int y) {
    return (height - y - 1) * width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color) {
    // old index: auto ind = point.y() + point.x() * width;
    int ind = (height - point.y() - 1) * width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader) {
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader) {
    fragment_shader = frag_shader;
}

void rst::rasterizer::add_light_projection(const Eigen::Matrix4f& p, const Eigen::Matrix4f &view, const light l)
{
    lights.push_back(l);
    light_projections.push_back(p);
    shadow_views.push_back(view);
    shadow_bufs.push_back(std::vector<float>(width * height, std::numeric_limits<float>::infinity()));
}
