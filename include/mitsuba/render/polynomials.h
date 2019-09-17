#pragma once
#if !defined(__POLYNOMIALS_H)
#define __POLYNOMIALS_H

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <time.h>
#include <vector>

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/kdtree.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/mitsuba.h>
#include <mitsuba/render/particleproc.h>
#include <mitsuba/render/range.h>
#include <mitsuba/render/renderjob.h>

#include <mitsuba/render/mediumparameters.h>

MTS_NAMESPACE_BEGIN

struct ScatterSamplingRecord;

class MTS_EXPORT_RENDER ConstraintKdTree {

public:
    struct ExtraData {
        Point p, avgP;
        Vector n, avgN;
        size_t sampleCount;
    };

    typedef SimpleKDNode<Point3, ExtraData> TreeNode;
    typedef PointKDTree<SimpleKDNode<Point3, ExtraData>> CTree;

    void build(const std::vector<Point3> &sampledP, const std::vector<Vector3> &sampledN) {
        auto nPoints = sampledP.size();
        m_tree       = CTree(nPoints);

        for (size_t i = 0; i < nPoints; ++i) {
            m_tree[i].setPosition(sampledP[i]);
            ExtraData d;
            d.p           = sampledP[i];
            d.n           = sampledN[i];
            d.sampleCount = 1;
            d.avgN        = Vector(-10, -10, -10);
            d.avgP        = Point(-100, -100, -100);
            m_tree[i].setData(d);
        }
        m_tree.build(true);

        // For each node the tree: Traverse children recursively and get average position, normal as well as sample
        // count sum out
        avgValues(m_tree[0], 0);
        std::cout << "Sample Count " << m_tree[0].getData().sampleCount << std::endl;
        // Gather a random subset of points
        for (size_t i = 0; i < std::min(size_t(32), sampledP.size()); ++i) {
            m_globalPoints.push_back(sampledP[i]);
            m_globalNormals.push_back(sampledN[i]);
        }
    }

    std::tuple<std::vector<Point>, std::vector<Vector>, std::vector<Float>>
    getConstraints(const Point &p, Float kernelEps, const std::function<Float(Float, Float)> &kernel) const {
        // Extract constraints from KD Tree by traversing from the top
        std::vector<Point> positions;
        std::vector<Vector> normals;
        std::vector<Float> sampleWeights;
        getConstraints(p, m_tree[0], 0, m_tree.getAABB(), positions, normals, sampleWeights, kernelEps, kernel);

        // Add the super constraints
        for (size_t i = 0; i < m_globalPoints.size(); ++i) {
            positions.push_back(m_globalPoints[i]);
            normals.push_back(m_globalNormals[i]);
            sampleWeights.push_back(-1.0f);
        }
        return std::make_tuple(positions, normals, sampleWeights);
    }

private:
    std::vector<Point> m_globalPoints;
    std::vector<Vector> m_globalNormals;
    CTree m_tree;
    std::tuple<Point, Vector, size_t> avgValues(TreeNode &node, TreeNode::IndexType index);
    std::pair<Float, Float> getMinMaxDistanceSquared(const Point &p, const AABB &bounds) const;
    void getConstraints(const Point &p, TreeNode node, TreeNode::IndexType index, const AABB &aabb,
                            std::vector<Point> &points, std::vector<Vector> &normals, std::vector<Float> &sampleWeights,
                            Float kernelEps, const std::function<Float(Float, Float)> &kernel) const;
};

class MTS_EXPORT_RENDER PolyUtils {

public:
    static constexpr int nChooseK(int n, int k) {
        return (k == 0 || n == k) ? 1 : nChooseK(n - 1, k - 1) + nChooseK(n - 1, k);
    }

    static constexpr int nPolyCoeffs(int polyOrder) { return nChooseK(3 + polyOrder, polyOrder); }

    static constexpr size_t powerToIndex(size_t dx, size_t dy, size_t dz) {
        // Converts a polynomial degree to a linear coefficient index
        auto d = dx + dy + dz;
        auto i = d - dx;
        auto j = d - dx - dy;
        return i * (i + 1) / 2 + j + d * (d + 1) * (d + 2) / 6;
    }

    static int multinomial(int a, int b, int c) {
        int res   = 1;
        int i     = 1;
        int denom = 1;
        for (int j = 1; j <= a; ++j) {
            res *= i;
            denom *= j;
            i++;
        }
        for (int j = 1; j <= b; ++j) {
            res *= i;
            denom *= j;
            i++;
        }
        for (int j = 1; j <= c; ++j) {
            res *= i;
            denom *= j;
            i++;
        }
        return res / denom;
    }

    static inline float powi(float f, int n) {
        float ret = 1.0f;
        for (int i = 0; i < n; ++i) {
            ret *= f;
        }
        return ret;
    }

    struct Polynomial {
        std::vector<float> coeffs;
        Point refPos;
        Vector refDir;
        bool useLocalDir;
        float scaleFactor;
        int order;
        std::vector<float> normalHistogram;
    };

    struct PolyFitConfig {
        float regularization         = 0.0001f;
        bool useSvd                  = false;
        bool useLightspace           = true;
        int order                    = 3;
        bool hardSurfaceConstraint   = true;
        float globalConstraintWeight = 0.01f;
        float kdTreeThreshold        = 0.0f;
        bool extractNormalHistogram  = false;
        bool useSimilarityKernel     = true;
        float kernelEpsScale         = 1.0f;
    };

    struct PolyFitRecord {
        PolyFitConfig config;
        Point p;
        Vector d;
        Vector n;
        MediumParameters medium;
        float kernelEps;
    };

    static std::tuple<std::vector<std::vector<Point>>, std::vector<std::vector<Vector>>,
                      std::vector<std::vector<float>>>
    getLocalPoints(const std::vector<Point3> &queryLocations, Float kernelEps,
                   const std::string &kernel, const ConstraintKdTree *kdtree);

    static std::tuple<Polynomial, std::vector<Point>, std::vector<Vector>>
    fitPolynomial(const PolyFitRecord &polyfitRecord, const ConstraintKdTree *kdtree);

    static void projectPointsToSurface(const Scene *scene, const Point &refPoint, const Vector &refDir,
                                           ScatterSamplingRecord &sRec, const Eigen::VectorXf &polyCoefficients,
                                           size_t polyOrder, bool useLocalDir, Float scaleFactor, Float kernelEps);

    static Intersection intersectPolynomial(const Ray &ray, const Polynomial &polynomial, Float stepSize,
                                            bool printDebug = false, int nBounce = 0);

    static float getKernelEps(const MediumParameters &medium, int channel = 0, Float kernel_multiplier = 1.0f);

    static inline float getFitScaleFactor(const MediumParameters &medium, int channel = 0, Float kernel_multiplier = 1.0f) {
        return getFitScaleFactor(getKernelEps(medium, channel, kernel_multiplier));
    }

    static inline float getFitScaleFactor(float kernelEps) { return 1.0f / std::sqrt(kernelEps); }

    static inline Float gaussianKernel(Float dist2, Float sigma2) {
        return std::exp(-dist2 / (2 * sigma2)); // Dont do any normalization since number of constraint points is
                                                // adjusted based on sigma2
    }

    template <size_t PolyOrder = 3>
    static void legendreTransform(Eigen::Matrix<float, nPolyCoeffs(PolyOrder), 1> &coeffs) {
        if (PolyOrder == 2) {
            float t[4]     = { 2.8284271247461903, 0.9428090415820634, 0.9428090415820634, 0.9428090415820634 };
            int indices[4] = { 0, 4, 7, 9 };

            float t2[9] = { 1.632993161855452,  1.632993161855452,  1.632993161855452,
                            0.8432740427115678, 0.9428090415820634, 0.9428090415820634,
                            0.8432740427115678, 0.9428090415820634, 0.8432740427115678 };
            float val   = 0.0f;
            for (int i = 0; i < 4; ++i) {
                val += coeffs[indices[i]] * t[i];
            }
            coeffs[0] = val;
            for (int i = 1; i < coeffs.size(); ++i) {
                coeffs[i] = t2[i - 1] * coeffs[i];
            }
        } else if (PolyOrder == 3) {
            float t[4][4]     = { { 2.8284271247461903, 0.9428090415820634, 0.9428090415820634, 0.9428090415820634 },
                              { 1.632993161855452, 0.9797958971132712, 0.5443310539518174, 0.5443310539518174 },
                              { 1.632993161855452, 0.5443310539518174, 0.9797958971132712, 0.5443310539518174 },
                              { 1.632993161855452, 0.5443310539518174, 0.5443310539518174, 0.9797958971132712 } };
            int indices[4][4] = { { 0, 4, 7, 9 }, { 1, 10, 13, 15 }, { 2, 11, 16, 18 }, { 3, 12, 17, 19 } };
            float t2[16]      = { 0.8432740427115678,  0.9428090415820634,  0.9428090415820634,  0.8432740427115678,
                             0.9428090415820634,  0.8432740427115678,  0.427617987059879,   0.48686449556014766,
                             0.48686449556014766, 0.48686449556014766, 0.5443310539518174,  0.48686449556014766,
                             0.427617987059879,   0.48686449556014766, 0.48686449556014766, 0.427617987059879 };

            for (int i = 0; i < 4; ++i) {
                float val = 0.0f;
                for (int j = 0; j < 4; ++j) {
                    val += coeffs[indices[i][j]] * t[i][j];
                }
                coeffs[i] = val;
            }
            for (int i = 4; i < coeffs.size(); ++i) {
                coeffs[i] = t2[i - 4] * coeffs[i];
            }
        } else {
            std::cout << "CANNOT HANDLE POLYNOMIALS OF ORDER != 2 or 3!!\n";
        }
    }

    template<int order>
    static void rotatePolynomial(std::vector<float> &c,
                                    std::vector<float> &c2,
                                    const Vector &s,
                                    const Vector &t,
                                    const Vector &n) {

        c2[0] = c[0];
        c2[1] = c[1]*s[0] + c[2]*s[1] + c[3]*s[2];
        c2[2] = c[1]*t[0] + c[2]*t[1] + c[3]*t[2];
        c2[3] = c[1]*n[0] + c[2]*n[1] + c[3]*n[2];
        c2[4] = c[4]*powi(s[0], 2) + c[5]*s[0]*s[1] + c[6]*s[0]*s[2] + c[7]*powi(s[1], 2) + c[8]*s[1]*s[2] + c[9]*powi(s[2], 2);
        c2[5] = 2*c[4]*s[0]*t[0] + c[5]*(s[0]*t[1] + s[1]*t[0]) + c[6]*(s[0]*t[2] + s[2]*t[0]) + 2*c[7]*s[1]*t[1] + c[8]*(s[1]*t[2] + s[2]*t[1]) + 2*c[9]*s[2]*t[2];
        c2[6] = 2*c[4]*n[0]*s[0] + c[5]*(n[0]*s[1] + n[1]*s[0]) + c[6]*(n[0]*s[2] + n[2]*s[0]) + 2*c[7]*n[1]*s[1] + c[8]*(n[1]*s[2] + n[2]*s[1]) + 2*c[9]*n[2]*s[2];
        c2[7] = c[4]*powi(t[0], 2) + c[5]*t[0]*t[1] + c[6]*t[0]*t[2] + c[7]*powi(t[1], 2) + c[8]*t[1]*t[2] + c[9]*powi(t[2], 2);
        c2[8] = 2*c[4]*n[0]*t[0] + c[5]*(n[0]*t[1] + n[1]*t[0]) + c[6]*(n[0]*t[2] + n[2]*t[0]) + 2*c[7]*n[1]*t[1] + c[8]*(n[1]*t[2] + n[2]*t[1]) + 2*c[9]*n[2]*t[2];
        c2[9] = c[4]*powi(n[0], 2) + c[5]*n[0]*n[1] + c[6]*n[0]*n[2] + c[7]*powi(n[1], 2) + c[8]*n[1]*n[2] + c[9]*powi(n[2], 2);
        if (order > 2) {
            c2[10] = c[10]*powi(s[0], 3) + c[11]*powi(s[0], 2)*s[1] + c[12]*powi(s[0], 2)*s[2] + c[13]*s[0]*powi(s[1], 2) + c[14]*s[0]*s[1]*s[2] + c[15]*s[0]*powi(s[2], 2) + c[16]*powi(s[1], 3) + c[17]*powi(s[1], 2)*s[2] + c[18]*s[1]*powi(s[2], 2) + c[19]*powi(s[2], 3);
            c2[11] = 3*c[10]*powi(s[0], 2)*t[0] + c[11]*(powi(s[0], 2)*t[1] + 2*s[0]*s[1]*t[0]) + c[12]*(powi(s[0], 2)*t[2] + 2*s[0]*s[2]*t[0]) + c[13]*(2*s[0]*s[1]*t[1] + powi(s[1], 2)*t[0]) + c[14]*(s[0]*s[1]*t[2] + s[0]*s[2]*t[1] + s[1]*s[2]*t[0]) + c[15]*(2*s[0]*s[2]*t[2] + powi(s[2], 2)*t[0]) + 3*c[16]*powi(s[1], 2)*t[1] + c[17]*(powi(s[1], 2)*t[2] + 2*s[1]*s[2]*t[1]) + c[18]*(2*s[1]*s[2]*t[2] + powi(s[2], 2)*t[1]) + 3*c[19]*powi(s[2], 2)*t[2];
            c2[12] = 3*c[10]*n[0]*powi(s[0], 2) + c[11]*(2*n[0]*s[0]*s[1] + n[1]*powi(s[0], 2)) + c[12]*(2*n[0]*s[0]*s[2] + n[2]*powi(s[0], 2)) + c[13]*(n[0]*powi(s[1], 2) + 2*n[1]*s[0]*s[1]) + c[14]*(n[0]*s[1]*s[2] + n[1]*s[0]*s[2] + n[2]*s[0]*s[1]) + c[15]*(n[0]*powi(s[2], 2) + 2*n[2]*s[0]*s[2]) + 3*c[16]*n[1]*powi(s[1], 2) + c[17]*(2*n[1]*s[1]*s[2] + n[2]*powi(s[1], 2)) + c[18]*(n[1]*powi(s[2], 2) + 2*n[2]*s[1]*s[2]) + 3*c[19]*n[2]*powi(s[2], 2);
            c2[13] = 3*c[10]*s[0]*powi(t[0], 2) + c[11]*(2*s[0]*t[0]*t[1] + s[1]*powi(t[0], 2)) + c[12]*(2*s[0]*t[0]*t[2] + s[2]*powi(t[0], 2)) + c[13]*(s[0]*powi(t[1], 2) + 2*s[1]*t[0]*t[1]) + c[14]*(s[0]*t[1]*t[2] + s[1]*t[0]*t[2] + s[2]*t[0]*t[1]) + c[15]*(s[0]*powi(t[2], 2) + 2*s[2]*t[0]*t[2]) + 3*c[16]*s[1]*powi(t[1], 2) + c[17]*(2*s[1]*t[1]*t[2] + s[2]*powi(t[1], 2)) + c[18]*(s[1]*powi(t[2], 2) + 2*s[2]*t[1]*t[2]) + 3*c[19]*s[2]*powi(t[2], 2);
            c2[14] = 6*c[10]*n[0]*s[0]*t[0] + c[11]*(2*n[0]*s[0]*t[1] + 2*n[0]*s[1]*t[0] + 2*n[1]*s[0]*t[0]) + c[12]*(2*n[0]*s[0]*t[2] + 2*n[0]*s[2]*t[0] + 2*n[2]*s[0]*t[0]) + c[13]*(2*n[0]*s[1]*t[1] + 2*n[1]*s[0]*t[1] + 2*n[1]*s[1]*t[0]) + c[14]*(n[0]*s[1]*t[2] + n[0]*s[2]*t[1] + n[1]*s[0]*t[2] + n[1]*s[2]*t[0] + n[2]*s[0]*t[1] + n[2]*s[1]*t[0]) + c[15]*(2*n[0]*s[2]*t[2] + 2*n[2]*s[0]*t[2] + 2*n[2]*s[2]*t[0]) + 6*c[16]*n[1]*s[1]*t[1] + c[17]*(2*n[1]*s[1]*t[2] + 2*n[1]*s[2]*t[1] + 2*n[2]*s[1]*t[1]) + c[18]*(2*n[1]*s[2]*t[2] + 2*n[2]*s[1]*t[2] + 2*n[2]*s[2]*t[1]) + 6*c[19]*n[2]*s[2]*t[2];
            c2[15] = 3*c[10]*powi(n[0], 2)*s[0] + c[11]*(powi(n[0], 2)*s[1] + 2*n[0]*n[1]*s[0]) + c[12]*(powi(n[0], 2)*s[2] + 2*n[0]*n[2]*s[0]) + c[13]*(2*n[0]*n[1]*s[1] + powi(n[1], 2)*s[0]) + c[14]*(n[0]*n[1]*s[2] + n[0]*n[2]*s[1] + n[1]*n[2]*s[0]) + c[15]*(2*n[0]*n[2]*s[2] + powi(n[2], 2)*s[0]) + 3*c[16]*powi(n[1], 2)*s[1] + c[17]*(powi(n[1], 2)*s[2] + 2*n[1]*n[2]*s[1]) + c[18]*(2*n[1]*n[2]*s[2] + powi(n[2], 2)*s[1]) + 3*c[19]*powi(n[2], 2)*s[2];
            c2[16] = c[10]*powi(t[0], 3) + c[11]*powi(t[0], 2)*t[1] + c[12]*powi(t[0], 2)*t[2] + c[13]*t[0]*powi(t[1], 2) + c[14]*t[0]*t[1]*t[2] + c[15]*t[0]*powi(t[2], 2) + c[16]*powi(t[1], 3) + c[17]*powi(t[1], 2)*t[2] + c[18]*t[1]*powi(t[2], 2) + c[19]*powi(t[2], 3);
            c2[17] = 3*c[10]*n[0]*powi(t[0], 2) + c[11]*(2*n[0]*t[0]*t[1] + n[1]*powi(t[0], 2)) + c[12]*(2*n[0]*t[0]*t[2] + n[2]*powi(t[0], 2)) + c[13]*(n[0]*powi(t[1], 2) + 2*n[1]*t[0]*t[1]) + c[14]*(n[0]*t[1]*t[2] + n[1]*t[0]*t[2] + n[2]*t[0]*t[1]) + c[15]*(n[0]*powi(t[2], 2) + 2*n[2]*t[0]*t[2]) + 3*c[16]*n[1]*powi(t[1], 2) + c[17]*(2*n[1]*t[1]*t[2] + n[2]*powi(t[1], 2)) + c[18]*(n[1]*powi(t[2], 2) + 2*n[2]*t[1]*t[2]) + 3*c[19]*n[2]*powi(t[2], 2);
            c2[18] = 3*c[10]*powi(n[0], 2)*t[0] + c[11]*(powi(n[0], 2)*t[1] + 2*n[0]*n[1]*t[0]) + c[12]*(powi(n[0], 2)*t[2] + 2*n[0]*n[2]*t[0]) + c[13]*(2*n[0]*n[1]*t[1] + powi(n[1], 2)*t[0]) + c[14]*(n[0]*n[1]*t[2] + n[0]*n[2]*t[1] + n[1]*n[2]*t[0]) + c[15]*(2*n[0]*n[2]*t[2] + powi(n[2], 2)*t[0]) + 3*c[16]*powi(n[1], 2)*t[1] + c[17]*(powi(n[1], 2)*t[2] + 2*n[1]*n[2]*t[1]) + c[18]*(2*n[1]*n[2]*t[2] + powi(n[2], 2)*t[1]) + 3*c[19]*powi(n[2], 2)*t[2];
            c2[19] = c[10]*powi(n[0], 3) + c[11]*powi(n[0], 2)*n[1] + c[12]*powi(n[0], 2)*n[2] + c[13]*n[0]*powi(n[1], 2) + c[14]*n[0]*n[1]*n[2] + c[15]*n[0]*powi(n[2], 2) + c[16]*powi(n[1], 3) + c[17]*powi(n[1], 2)*n[2] + c[18]*n[1]*powi(n[2], 2) + c[19]*powi(n[2], 3);
        }
        for (size_t i = 0; i < c2.size(); ++i) {
            c[i] = c2[i];
        }
    }

    static Vector adjustRayDirForPolynomialTracing(Vector &inDir, const Intersection &its, int polyOrder,
                                                 float polyScaleFactor, int channel = 0);

    static Vector adjustRayForPolynomialTracing(Ray &ray, const Polynomial &polynomial, const Vector &targetNormal);
    static bool adjustRayForPolynomialTracingFull(Ray &ray, const Polynomial &polynomial, const Vector &targetNormal);


    template<int order, typename T>
    static Eigen::Matrix<float, nPolyCoeffs(order), 1> rotatePolynomialEigen(
                                    const T &c,
                                    const Vector &s,
                                    const Vector &t,
                                    const Vector &n) {
        Eigen::Matrix<float, nPolyCoeffs(order), 1> c2;
        c2[0] = c[0];
        c2[1] = c[1]*s[0] + c[2]*s[1] + c[3]*s[2];
        c2[2] = c[1]*t[0] + c[2]*t[1] + c[3]*t[2];
        c2[3] = c[1]*n[0] + c[2]*n[1] + c[3]*n[2];
        c2[4] = c[4]*powi(s[0], 2) + c[5]*s[0]*s[1] + c[6]*s[0]*s[2] + c[7]*powi(s[1], 2) + c[8]*s[1]*s[2] + c[9]*powi(s[2], 2);
        c2[5] = 2*c[4]*s[0]*t[0] + c[5]*(s[0]*t[1] + s[1]*t[0]) + c[6]*(s[0]*t[2] + s[2]*t[0]) + 2*c[7]*s[1]*t[1] + c[8]*(s[1]*t[2] + s[2]*t[1]) + 2*c[9]*s[2]*t[2];
        c2[6] = 2*c[4]*n[0]*s[0] + c[5]*(n[0]*s[1] + n[1]*s[0]) + c[6]*(n[0]*s[2] + n[2]*s[0]) + 2*c[7]*n[1]*s[1] + c[8]*(n[1]*s[2] + n[2]*s[1]) + 2*c[9]*n[2]*s[2];
        c2[7] = c[4]*powi(t[0], 2) + c[5]*t[0]*t[1] + c[6]*t[0]*t[2] + c[7]*powi(t[1], 2) + c[8]*t[1]*t[2] + c[9]*powi(t[2], 2);
        c2[8] = 2*c[4]*n[0]*t[0] + c[5]*(n[0]*t[1] + n[1]*t[0]) + c[6]*(n[0]*t[2] + n[2]*t[0]) + 2*c[7]*n[1]*t[1] + c[8]*(n[1]*t[2] + n[2]*t[1]) + 2*c[9]*n[2]*t[2];
        c2[9] = c[4]*powi(n[0], 2) + c[5]*n[0]*n[1] + c[6]*n[0]*n[2] + c[7]*powi(n[1], 2) + c[8]*n[1]*n[2] + c[9]*powi(n[2], 2);
        if (order > 2) {
            c2[10] = c[10]*powi(s[0], 3) + c[11]*powi(s[0], 2)*s[1] + c[12]*powi(s[0], 2)*s[2] + c[13]*s[0]*powi(s[1], 2) + c[14]*s[0]*s[1]*s[2] + c[15]*s[0]*powi(s[2], 2) + c[16]*powi(s[1], 3) + c[17]*powi(s[1], 2)*s[2] + c[18]*s[1]*powi(s[2], 2) + c[19]*powi(s[2], 3);
            c2[11] = 3*c[10]*powi(s[0], 2)*t[0] + c[11]*(powi(s[0], 2)*t[1] + 2*s[0]*s[1]*t[0]) + c[12]*(powi(s[0], 2)*t[2] + 2*s[0]*s[2]*t[0]) + c[13]*(2*s[0]*s[1]*t[1] + powi(s[1], 2)*t[0]) + c[14]*(s[0]*s[1]*t[2] + s[0]*s[2]*t[1] + s[1]*s[2]*t[0]) + c[15]*(2*s[0]*s[2]*t[2] + powi(s[2], 2)*t[0]) + 3*c[16]*powi(s[1], 2)*t[1] + c[17]*(powi(s[1], 2)*t[2] + 2*s[1]*s[2]*t[1]) + c[18]*(2*s[1]*s[2]*t[2] + powi(s[2], 2)*t[1]) + 3*c[19]*powi(s[2], 2)*t[2];
            c2[12] = 3*c[10]*n[0]*powi(s[0], 2) + c[11]*(2*n[0]*s[0]*s[1] + n[1]*powi(s[0], 2)) + c[12]*(2*n[0]*s[0]*s[2] + n[2]*powi(s[0], 2)) + c[13]*(n[0]*powi(s[1], 2) + 2*n[1]*s[0]*s[1]) + c[14]*(n[0]*s[1]*s[2] + n[1]*s[0]*s[2] + n[2]*s[0]*s[1]) + c[15]*(n[0]*powi(s[2], 2) + 2*n[2]*s[0]*s[2]) + 3*c[16]*n[1]*powi(s[1], 2) + c[17]*(2*n[1]*s[1]*s[2] + n[2]*powi(s[1], 2)) + c[18]*(n[1]*powi(s[2], 2) + 2*n[2]*s[1]*s[2]) + 3*c[19]*n[2]*powi(s[2], 2);
            c2[13] = 3*c[10]*s[0]*powi(t[0], 2) + c[11]*(2*s[0]*t[0]*t[1] + s[1]*powi(t[0], 2)) + c[12]*(2*s[0]*t[0]*t[2] + s[2]*powi(t[0], 2)) + c[13]*(s[0]*powi(t[1], 2) + 2*s[1]*t[0]*t[1]) + c[14]*(s[0]*t[1]*t[2] + s[1]*t[0]*t[2] + s[2]*t[0]*t[1]) + c[15]*(s[0]*powi(t[2], 2) + 2*s[2]*t[0]*t[2]) + 3*c[16]*s[1]*powi(t[1], 2) + c[17]*(2*s[1]*t[1]*t[2] + s[2]*powi(t[1], 2)) + c[18]*(s[1]*powi(t[2], 2) + 2*s[2]*t[1]*t[2]) + 3*c[19]*s[2]*powi(t[2], 2);
            c2[14] = 6*c[10]*n[0]*s[0]*t[0] + c[11]*(2*n[0]*s[0]*t[1] + 2*n[0]*s[1]*t[0] + 2*n[1]*s[0]*t[0]) + c[12]*(2*n[0]*s[0]*t[2] + 2*n[0]*s[2]*t[0] + 2*n[2]*s[0]*t[0]) + c[13]*(2*n[0]*s[1]*t[1] + 2*n[1]*s[0]*t[1] + 2*n[1]*s[1]*t[0]) + c[14]*(n[0]*s[1]*t[2] + n[0]*s[2]*t[1] + n[1]*s[0]*t[2] + n[1]*s[2]*t[0] + n[2]*s[0]*t[1] + n[2]*s[1]*t[0]) + c[15]*(2*n[0]*s[2]*t[2] + 2*n[2]*s[0]*t[2] + 2*n[2]*s[2]*t[0]) + 6*c[16]*n[1]*s[1]*t[1] + c[17]*(2*n[1]*s[1]*t[2] + 2*n[1]*s[2]*t[1] + 2*n[2]*s[1]*t[1]) + c[18]*(2*n[1]*s[2]*t[2] + 2*n[2]*s[1]*t[2] + 2*n[2]*s[2]*t[1]) + 6*c[19]*n[2]*s[2]*t[2];
            c2[15] = 3*c[10]*powi(n[0], 2)*s[0] + c[11]*(powi(n[0], 2)*s[1] + 2*n[0]*n[1]*s[0]) + c[12]*(powi(n[0], 2)*s[2] + 2*n[0]*n[2]*s[0]) + c[13]*(2*n[0]*n[1]*s[1] + powi(n[1], 2)*s[0]) + c[14]*(n[0]*n[1]*s[2] + n[0]*n[2]*s[1] + n[1]*n[2]*s[0]) + c[15]*(2*n[0]*n[2]*s[2] + powi(n[2], 2)*s[0]) + 3*c[16]*powi(n[1], 2)*s[1] + c[17]*(powi(n[1], 2)*s[2] + 2*n[1]*n[2]*s[1]) + c[18]*(2*n[1]*n[2]*s[2] + powi(n[2], 2)*s[1]) + 3*c[19]*powi(n[2], 2)*s[2];
            c2[16] = c[10]*powi(t[0], 3) + c[11]*powi(t[0], 2)*t[1] + c[12]*powi(t[0], 2)*t[2] + c[13]*t[0]*powi(t[1], 2) + c[14]*t[0]*t[1]*t[2] + c[15]*t[0]*powi(t[2], 2) + c[16]*powi(t[1], 3) + c[17]*powi(t[1], 2)*t[2] + c[18]*t[1]*powi(t[2], 2) + c[19]*powi(t[2], 3);
            c2[17] = 3*c[10]*n[0]*powi(t[0], 2) + c[11]*(2*n[0]*t[0]*t[1] + n[1]*powi(t[0], 2)) + c[12]*(2*n[0]*t[0]*t[2] + n[2]*powi(t[0], 2)) + c[13]*(n[0]*powi(t[1], 2) + 2*n[1]*t[0]*t[1]) + c[14]*(n[0]*t[1]*t[2] + n[1]*t[0]*t[2] + n[2]*t[0]*t[1]) + c[15]*(n[0]*powi(t[2], 2) + 2*n[2]*t[0]*t[2]) + 3*c[16]*n[1]*powi(t[1], 2) + c[17]*(2*n[1]*t[1]*t[2] + n[2]*powi(t[1], 2)) + c[18]*(n[1]*powi(t[2], 2) + 2*n[2]*t[1]*t[2]) + 3*c[19]*n[2]*powi(t[2], 2);
            c2[18] = 3*c[10]*powi(n[0], 2)*t[0] + c[11]*(powi(n[0], 2)*t[1] + 2*n[0]*n[1]*t[0]) + c[12]*(powi(n[0], 2)*t[2] + 2*n[0]*n[2]*t[0]) + c[13]*(2*n[0]*n[1]*t[1] + powi(n[1], 2)*t[0]) + c[14]*(n[0]*n[1]*t[2] + n[0]*n[2]*t[1] + n[1]*n[2]*t[0]) + c[15]*(2*n[0]*n[2]*t[2] + powi(n[2], 2)*t[0]) + 3*c[16]*powi(n[1], 2)*t[1] + c[17]*(powi(n[1], 2)*t[2] + 2*n[1]*n[2]*t[1]) + c[18]*(2*n[1]*n[2]*t[2] + powi(n[2], 2)*t[1]) + 3*c[19]*powi(n[2], 2)*t[2];
            c2[19] = c[10]*powi(n[0], 3) + c[11]*powi(n[0], 2)*n[1] + c[12]*powi(n[0], 2)*n[2] + c[13]*n[0]*powi(n[1], 2) + c[14]*n[0]*n[1]*n[2] + c[15]*n[0]*powi(n[2], 2) + c[16]*powi(n[1], 3) + c[17]*powi(n[1], 2)*n[2] + c[18]*n[1]*powi(n[2], 2) + c[19]*powi(n[2], 3);
        }
        return c2;
    }

    // Slow version of polynomial rotation. Retained for understandability.
    template <int order>
    static void rotatePolynomialOld(std::vector<float> &coeffs, std::vector<float> &tmpCoeffs, const Vector &s,
                                    const Vector &t, const Vector &n) {
        assert(tmpCoeffs.size() == coeffs.size());
        for (size_t i = 0; i < tmpCoeffs.size(); ++i) {
            tmpCoeffs[i] = 0;
        }
        for (int l = 0; l <= order; ++l) {
            for (int m = 0; m <= order - l; ++m) {
                for (int k = 0; k <= order - l - m; ++k) {
                    const int pi = powerToIndex(l, m, k);
                    for (int a = 0; a <= order; ++a) {
                        for (int b = 0; b <= order - a; ++b) {
                            for (int c = 0; c <= order - a - b; ++c) {
                                if (l + m + k != a + b + c)
                                    continue;
                                const int pj = powerToIndex(a, b, c);
                                float coeff  = 0.0f;
                                for (int i_1 = 0; i_1 <= l; ++i_1) {
                                    for (int j_1 = 0; j_1 <= l - i_1; ++j_1) {
                                        for (int i_2 = std::max(0, a - i_1 - k); i_2 <= std::min(m, a - i_1); ++i_2) {
                                            for (int j_2 = std::max(0, b - j_1 - k + a - i_1 - i_2);
                                                 j_2 <= std::min(m - i_2, b - j_1); ++j_2) {
                                                const int i_3 = a - i_1 - i_2;
                                                const int j_3 = b - j_1 - j_2;
                                                const int k_1 = l - i_1 - j_1;
                                                const int k_2 = m - i_2 - j_2;
                                                const int k_3 = k - i_3 - j_3;
                                                float term = multinomial(i_1, j_1, k_1) * multinomial(i_2, j_2, k_2) *
                                                             multinomial(i_3, j_3, k_3);
                                                term *= powi(s[0], i_1);
                                                term *= powi(t[0], j_1);
                                                term *= powi(n[0], k_1);
                                                term *= powi(s[1], i_2);
                                                term *= powi(t[1], j_2);
                                                term *= powi(n[1], k_2);
                                                term *= powi(s[2], i_3);
                                                term *= powi(t[2], j_3);
                                                term *= powi(n[2], k_3);
                                                coeff += term;
                                            }
                                        }
                                    }
                                }
                                tmpCoeffs[pj] += coeffs[pi] * coeff;
                            }
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < tmpCoeffs.size(); ++i) {
            coeffs[i] = tmpCoeffs[i];
        }
    }
};

MTS_NAMESPACE_END


#endif
