#include <vector>

#include "rasterizer.hpp"
#include "Texture.hpp"
#include "Shader.hpp"
#include "global.hpp"
#include "Triangle.hpp"
#include "OBJ_Loader.h"
#include "task3_utils.h"

using namespace std;




int main(int argc, char const *argv[])
{
    objl::Loader Loader;

    std::vector<Triangle *> modelTriangles;
    std::vector<Triangle *> tableTriangles;
    float angle = 140.0f;

    std::string filename = "output.png";
    rst::Shading shading = rst::Shading::Phong;
    std::string obj_path = "../models/spot/";

    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for (auto mesh : Loader.LoadedMeshes) {
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            Triangle *t = new Triangle();
            for (int j = 0; j < 3; j++) {
                t->setColor(j, 255, 255, 240);
                t->setVertex(j, Vector4f(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y,
                                         mesh.Vertices[i + j].Position.Z, 1.0));
                t->setNormal(j, Vector3f(mesh.Vertices[i + j].Normal.X, mesh.Vertices[i + j].Normal.Y,
                                         mesh.Vertices[i + j].Normal.Z));
                t->setTexCoord(
                    j, Vector2f(mesh.Vertices[i + j].TextureCoordinate.X, mesh.Vertices[i + j].TextureCoordinate.Y));
            }
            modelTriangles.push_back(t);
        }
    }
    float y = -0.75f;
    Vector3f v0(-5, y, -5);
    Vector3f v1(-5, y,  3);
    Vector3f v2( 5, y,  3);
    Vector3f v3( 5, y, -5);

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

    int width = 700;
    int height = 700;
    rst::rasterizer r(width, height);
    auto texture_path = "spot_texture.png";
    r.set_texture(Texture(obj_path + texture_path));

    // Replace get_projection_matrix (perspective) with orthographic for the light:
    // Orthographic: manually build the matrix
    // ortho(left, right, bottom, top, near, far)
    float s = 3.0f;  // half-size of the shadow frustum in world units
    float zN = 0.1f, zF = 50.0f;
    Eigen::Matrix4f ortho;
    ortho << 1.0f/s,      0,            0,                    0,
             0,           1.0f/s,       0,                    0,
             0,           0,            -2.0f/(zF-zN),        -(zF+zN)/(zF-zN),
             0,           0,            0,                    1.0f;




    Vector3f eye_pos = {0, 2, 5};
    auto l1 = light{{4, 10, -4}, {50, 50, 50}};
    Vector3f target = {0.122f, 0, -0.146f};  // light aims straight down at cow center
    Vector3f up = {0, 0, -1};
    auto l2 = light{{-4, 9, 4}, {100, 100, 100}};



    function<Vector3f(fragment_shader_payload)> active_shader = texture_fragment_shader;

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    r.clear(rst::Buffers::Color | rst::Buffers::Depth);
    r.set_model(get_model_matrix(angle, {0, 1, 0}, {0, 0, 0}));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
    r.add_light_projection(ortho, look_at(l1.position, target, up), l1);
    r.add_light_projection(ortho, look_at(l2.position, target, up), l2);



    // 1. Shadow map the cow
    r.set_model(get_model_matrix(angle, {0, 1, 0}, {0, 0, 0}));
    r.shadow_mapping(modelTriangles, 0);
    r.shadow_mapping(modelTriangles, 1);

    // 2. Draw table FIRST (so depth buffer is inf when table fragments run)
    r.set_model(Eigen::Matrix4f::Identity());
    r.draw(tableTriangles, true, shading, true);

    // 3. Draw cow SECOND
    r.set_model(get_model_matrix(angle, {0, 1, 0}, {0, 0, 0}));
    r.draw(modelTriangles, true, shading);

    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, image);

    return 0;
}

