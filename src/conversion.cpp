#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Vertex { float x, y, z; };
struct Face   { std::vector<int> indices; };

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ply_to_obj input.ply output.obj\n";
        return 1;
    }

    std::string obj_path = "../models/bunny/";
    std::ifstream in(obj_path + argv[1]);
    if (!in) { std::cerr << "Cannot open " << argv[1] << "\n"; return 1; }

    std::string line;
    int vertex_count = 0, face_count = 0;
    std::vector<std::string> vertex_props;

    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tok; ss >> tok;
        if (tok == "end_header") break;
        if (tok == "element") {
            std::string name; int count;
            ss >> name >> count;
            if (name == "vertex") vertex_count = count;
            if (name == "face")   face_count   = count;
        }
        if (tok == "property") {
            std::string type, name;
            ss >> type >> name;
            if (type != "list") vertex_props.push_back(name);
        }
    }

    int xi=-1, yi=-1, zi=-1;
    for (int i = 0; i < (int)vertex_props.size(); i++) {
        if (vertex_props[i] == "x") xi = i;
        if (vertex_props[i] == "y") yi = i;
        if (vertex_props[i] == "z") zi = i;
    }
    int n_props = (int)vertex_props.size();

    std::cout << "Properties (" << n_props << "): ";
    for (auto& p : vertex_props) std::cout << p << " ";
    std::cout << "\n";

    std::vector<Vertex> verts;
    verts.reserve(vertex_count);
    for (int i = 0; i < vertex_count; i++) {
        if (!std::getline(in, line)) break;
        std::istringstream ss(line);
        std::vector<float> vals(n_props);
        for (auto& f : vals) ss >> f;
        verts.push_back({vals[xi], vals[yi], vals[zi]});
    }

    std::vector<Face> faces;
    faces.reserve(face_count);
    for (int i = 0; i < face_count; i++) {
        if (!std::getline(in, line)) break;
        std::istringstream ss(line);
        int count; ss >> count;
        Face f;
        for (int j = 0; j < count; j++) {
            int idx; ss >> idx;
            f.indices.push_back(idx);
        }
        faces.push_back(f);
    }

    std::ofstream out(obj_path + argv[2]);
    if (!out) { std::cerr << "Cannot write " << argv[2] << "\n"; return 1; }

    out << "# Converted from PLY\n\n";
    for (auto& v : verts)
        out << "v " << v.x << " " << v.y << " " << v.z << "\n";
    out << "\n";

    for (auto& f : faces)
        for (int i = 1; i+1 < (int)f.indices.size(); i++)
            out << "f " << f.indices[0]+1
                << " " << f.indices[i]+1
                << " " << f.indices[i+1]+1 << "\n";

    std::cout << "Written " << verts.size() << " vertices, "
              << faces.size() << " faces to " << argv[2] << "\n";
    return 0;
}