#include <mitsuba/render/mesh_histogram.h>

MTS_NAMESPACE_BEGIN 

std::vector<float> MeshHistogram::compute(const Scene *scene, const std::vector<Point> &points, const std::vector<Vector> &projDirs) {
    // Assmption: Scene only contains a single shape 
    const TriMesh *shape = dynamic_cast<const TriMesh*>(scene->getShapes()[0].get());
    assert(shape);
    size_t numTriangles = shape->getTriangleCount();
    std::vector<float> result(numTriangles, 0.0f);
    std::vector<int> intResult(numTriangles, 0);
    const Triangle *triangles = shape->getTriangles();
    const Point *positions = shape->getVertexPositions();
    for (int i = 0; i < points.size(); ++i) {
        Ray proj(points[i], projDirs[i], -Epsilon, 100.0f, 0.0f);
        Ray proj2(points[i], projDirs[i], -100.0f, Epsilon, 0.0f);
        Intersection its, its2;
        scene->rayIntersect(proj, its);
        scene->rayIntersect(proj2, its2);
        if (its.isValid() && its2.isValid()) {
            if (std::abs(its.t) < std::abs(its2.t)) {
                intResult[its.primIndex]++;
            } else {
                intResult[its2.primIndex]++;
            }
        } else if (its.isValid()) {
            intResult[its.primIndex]++;
        } else if (its2.isValid()) {
            intResult[its2.primIndex]++;
        } else {
            std::cout << "Surface missed!\n";
        }
    }
    // Normalize by triangle areas
    for (int i = 0; i < numTriangles; ++i) 
        result[i] = intResult[i] / (triangles[i].surfaceArea(positions) + 1e-7f);
    return result;
}

MTS_NAMESPACE_END