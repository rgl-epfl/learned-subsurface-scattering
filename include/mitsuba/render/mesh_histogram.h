#pragma once
#if !defined(__MESHHISTOGRAM_H)
#define __MESHHISTOGRAM_H

#include <mitsuba/render/particleproc.h>
#include <mitsuba/render/range.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/mitsuba.h>
#include <mitsuba/core/kdtree.h>
#include <mitsuba/render/scene.h>


MTS_NAMESPACE_BEGIN


class MTS_EXPORT_RENDER MeshHistogram {
public:
    // Output: Vector of histogram values ordered by triangle index
    static std::vector<float> compute(const Scene *scene, const std::vector<Point> &points, const std::vector<Vector> &projDirs);
};


MTS_NAMESPACE_END


#endif
