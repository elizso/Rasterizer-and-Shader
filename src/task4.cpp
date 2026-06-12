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
#include "task3_utils.h"
#include <map>

using namespace std;




int main(int argc, const char **argv) {
    std::vector<Triangle *> TriangleList;
    std::vector<Triangle *> tableTriangles;

    float angle = 0.0f;
    bool shadow = false;

    std::string filename = "output.png";
    rst::Shading shading = rst::Shading::Phong;
    objl::Loader Loader;
    std::string obj_path = "../models/bunny/";


    std::vector<Vector3f> positions;
    std::vector<Vector3f> smooth_normals;

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/bunny/bunny.obj");

    // Map from position -> accumulated normal
    std::map<std::tuple<float,float,float>, Vector3f> pos_to_normal;

    for (auto& mesh : Loader.LoadedMeshes) {
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            Vector3f p0(mesh.Vertices[i+0].Position.X, mesh.Vertices[i+0].Position.Y, mesh.Vertices[i+0].Position.Z);
            Vector3f p1(mesh.Vertices[i+1].Position.X, mesh.Vertices[i+1].Position.Y, mesh.Vertices[i+1].Position.Z);
            Vector3f p2(mesh.Vertices[i+2].Position.X, mesh.Vertices[i+2].Position.Y, mesh.Vertices[i+2].Position.Z);
            Vector3f fn = (p1-p0).cross(p2-p0).normalized();

            for (auto& p : {p0, p1, p2}) {
                auto key = std::make_tuple(p.x(), p.y(), p.z());
                pos_to_normal[key] += fn;
            }
        }
    }

    for (auto& [key, n] : pos_to_normal) n.normalize();

    for (auto& mesh : Loader.LoadedMeshes) {
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            Triangle *t = new Triangle();
            for (int j = 0; j < 3; j++) {
                Vector3f p(mesh.Vertices[i+j].Position.X,
                           mesh.Vertices[i+j].Position.Y,
                           mesh.Vertices[i+j].Position.Z);
                auto key = std::make_tuple(p.x(), p.y(), p.z());
                Vector3f smooth_n = pos_to_normal[key];  // averaged across all faces at this position

                t->setVertex(j, Vector4f(p.x(), p.y(), p.z(), 1.0));
                t->setNormal(j, smooth_n);
                t->setTexCoord(j, Vector2f(mesh.Vertices[i+j].TextureCoordinate.X,
                                            mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }



    float s = 3.0f;  // half-size of the shadow frustum in world units
    float zN = 0.1f, zF = 50.0f;
    Eigen::Matrix4f ortho;
    ortho << 1.0f/s,      0,            0,                    0,
             0,           1.0f/s,       0,                    0,
             0,           0,            -2.0f/(zF-zN),        -(zF+zN)/(zF-zN),
             0,           0,            0,                    1.0f;


    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    Eigen::Vector3f eye_pos = {0, 2, 5};
    auto l1 = light{{-4, 10, 4}, {100, 100, 100}};
    auto l2 = light{{ 4, 10, 4}, {100, 100, 100}};
    std::vector<light> lights = {l1, l2};
    Vector3f target = {0, 0, 0};
    Vector3f up = {0, 1, 0};


    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = blinn_phong_fragment_shader;

    if (argc < 5) {
        std::cout << "Usage: [Shader (texture, normal, blinn-phong, bump)] [Shading "
                     "Frequency (Flat, Gouraud, Phong)] [Shadow (True, False)] [savename.png]"
                  << std::endl;
        return 1;
    } else {
        if (argc == 5) {
            filename = std::string(argv[4]);
        }
        if (std::string(argv[1]) == "texture") {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "bob_diffuse.png";
            r.set_texture(Texture(obj_path + texture_path));
        } else if (std::string(argv[1]) == "normal") {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        } else if (std::string(argv[1]) == "blinn-phong") {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = blinn_phong_fragment_shader;
        } else if (std::string(argv[1]) == "bump") {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }

        if (std::string(argv[2]) == "Flat") {
            std::cout << "Rasterizing using Flat shading\n";
            shading = rst::Shading::Flat;
        } else if (std::string(argv[2]) == "Gouraud") {
            std::cout << "Rasterizing using Goround shading\n";
            shading = rst::Shading::Gouraud;
        } else if (std::string(argv[2]) == "Phong") {
            std::cout << "Rasterizing using Phong shading\n";
            shading = rst::Shading::Phong;
        }

        if (std::string(argv[3]) == "True")
        {
            float plane_y = -1.0f;   // just below bunny feet
            Vector3f v0(-5, plane_y, -5);
            Vector3f v1(-5, plane_y,  5);
            Vector3f v2( 5, plane_y,  5);
            Vector3f v3( 5, plane_y, -5);

            Vector3f normal = triangle_normal(v0, v1, v2);
            Vector3f color (255, 255, 255);

            Triangle *t = new Triangle();
            t->setVertex(0, v0.homogeneous());
            t->setVertex(1, v1.homogeneous());
            t->setVertex(2, v2.homogeneous());
            t->setNormals({normal, normal, normal});
            t->setColors({color, color, color});
            tableTriangles.push_back(t);

            t = new Triangle();
            t->setVertex(0, v0.homogeneous());
            t->setVertex(1, v2.homogeneous());
            t->setVertex(2, v3.homogeneous());
            t->setNormals({normal, normal, normal});
            t->setColors({color, color, color});
            tableTriangles.push_back(t);
            shadow = true;

            r.add_light_projection(ortho, look_at(l1.position, target, up), l1);
            r.add_light_projection(ortho, look_at(l2.position, target, up), l2);
        } else
        {
            r.set_lights(lights);
        }
    }




    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);


    r.clear(rst::Buffers::Color | rst::Buffers::Depth);
    r.set_model(get_model_matrix(angle, {1, 0, 0}, {0.5, -1.8, 0}, 20));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
    if (shadow)
    {
        r.shadow_mapping(TriangleList, 0);
        r.set_model(Eigen::Matrix4f::Identity());
        r.set_fragment_shader(blinn_phong_fragment_shader);
        r.draw(tableTriangles, true, shading, true);

        r.set_model(get_model_matrix(angle, {1, 0, 0}, {0, -1.3, 0}, 20));

    }
    r.set_fragment_shader(active_shader);
    r.draw(TriangleList, true, shading);


    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, image);




    return 0;
}
