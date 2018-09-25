#include "registration.h"
#include "registration_kernel.h"

#include <iostream>
#include "Eigen/Geometry"
#include "Eigen/SVD"
#include "fdtree.h"
#include "kernel.h"
#include <chrono>

//提案された姿勢
const float SUG_THETA = 14 * M_PI / 180;
const float SUG_PHI = -146 * M_PI / 180;
//点群抽出のパラメータ
uint16_t CT_THRESHOLD = 40000;
uint16_t LAPLACIAN_THRESHOLD = 40000;
const float CTV_RATE = 0.001f; //CT値が1離れたボクセルを何mm離れたものとみなすか


namespace {
    Vec3 FromEigen(const Eigen::Vector3f &v) {
        return{ v.x(), v.y(), v.z() };
    }
    Eigen::Vector3f ToEigen(const Vec3 u) {
        return{ u.x, u.y, u.z };
    }
    Mat3 FromEigen(const Eigen::Matrix3f &m) {
        Mat3 m3;
        for (int i = 0; i < 3; i++)
        {
            m3.rows[i].x = m.row(i).x();
            m3.rows[i].y = m.row(i).y();
            m3.rows[i].z = m.row(i).z();
        }

        return m3;
    }
    void FindRigidTransformationICP(const std::vector<Vec4> &ref_points, const std::vector<Vec4> &tar_points, Eigen::Matrix3f &rotation, Eigen::Vector3f &translation);
    void RigidTransformationOfPairs(Vec3 *reference, Vec3 *target, size_t n, Eigen::Vector3f &translation, Eigen::Matrix3f &rotation);
    void SamplePoints(
        Volume &vol, Volume &lap, Volume &aff, std::vector<Vec4> &points, Volume &used,
        float lap_min, int ct_min, int aff_max,
        size_t downsample_step);
    void hsvtorgb(float h, float s, float v, float &r, float &g, float &b);

    void FindRigidTransformationICP(const std::vector<Vec4> &ref_points, const std::vector<Vec4> &tar_points, Eigen::Matrix3f &rotation, Eigen::Vector3f &translation)
    {

        for (int iter = 0; iter < 200; iter++) {
            std::vector<Vec3> points_a, points_b;
            
            points_b = cuda_transform_points(tar_points, FromEigen(rotation), FromEigen(translation));
            points_a = cuda_associate_closest(tar_points, points_b, ref_points);
            
            Eigen::Matrix3f rotation_i;
            Eigen::Vector3f translation_i;

            std::cout << "----------" << iter << std::endl;
            ::RigidTransformationOfPairs(points_a.data(), points_b.data(), points_a.size(), translation_i, rotation_i);

          std::cout << "norm " << translation_i.norm() << std::endl;


            rotation = rotation_i * rotation;
            translation = rotation_i * translation + translation_i;

        }
        std::cout << "rotation" << std::endl << rotation << std::endl;
        std::cout << "det = " << rotation.determinant() << std::endl;
        std::cout << "translation" << std::endl << translation << std::endl;
    }

    void RigidTransformationOfPairs(Vec3 *reference, Vec3 *target, size_t n, Eigen::Vector3f &translation, Eigen::Matrix3f &rotation)
    {
        Vec3 reference_avg(0, 0, 0), target_avg (0, 0, 0);
        for (size_t i = 0; i < n; i++) {
            reference_avg = reference_avg + reference[i];
            target_avg = target_avg + target[i];
        }
        reference_avg =  reference_avg / (float)n;
        target_avg = target_avg / (float)n;

        for (size_t i = 0; i < n; i++) {
            reference[i] = reference[i] - reference_avg;
            target[i] = target[i] - target_avg;
        }
        translation = ToEigen(reference_avg - target_avg);

        Eigen::Matrix3f S = Eigen::Matrix3f::Zero();

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 3; k++) {
                    S(j, k) += target[i][j] * reference[i][k];
                }
            }
        }

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
        rotation = svd.matrixV() * svd.matrixU().transpose();
    }


    std::vector<Vec4> ConvertVolumeToPointcloud(Volume &volume, Volume &affection)
    {
        Volume laplacian = cuda_laplacian(volume);
        laplacian.WriteRaw("laplacian.raw");
        std::vector<Vec4> points;
   

        
        auto points_all = cuda_samplepoints(laplacian, volume, CT_THRESHOLD, laplacian, LAPLACIAN_THRESHOLD, affection, 65536); //lever

        size_t DOWNSAMPLE = points_all.size() / 60000;
        for (size_t i = 0; i < points_all.size(); i += DOWNSAMPLE) {
            Vec4 p = points_all.at(i);
            p.w *= CTV_RATE;
            points.push_back(p);
        }
        
        return std::move(points);
    }

    void hsvtorgb(float h, float s, float v, float &r, float &g, float &b) {
        r = v;
        g = v;
        b = v;
        if (s > 0.0f) {
            h *= 6.0f;
            int i = (int)h;
            float f = h - (float)i;
            switch (i) {
            default:
            case 0:
                g *= 1 - s * (1 - f);
                b *= 1 - s;
                break;
            case 1:
                r *= 1 - s * f;
                b *= 1 - s;
                break;
            case 2:
                r *= 1 - s;
                b *= 1 - s * (1 - f);
                break;
            case 3:
                r *= 1 - s;
                g *= 1 - s * f;
                break;
            case 4:
                r *= 1 - s * (1 - f);
                g *= 1 - s;
                break;
            case 5:
                g *= 1 - s;
                b *= 1 - s * f;
                break;
            }
        }
    }
}



void Registration(Volume &vol_ref, Volume &aff_ref, Volume &vol_tar, Volume &aff_tar)
{
    std::vector<Vec4> tar_points = ::ConvertVolumeToPointcloud(vol_tar, aff_tar);
    std::vector<Vec4> ref_points = ::ConvertVolumeToPointcloud(vol_ref, aff_ref);
    
    std::cout << "pointcloud size" << std::endl;
    std::cout << "reference : " << ref_points.size() << std::endl;
    std::cout << "target : " << tar_points.size() << std::endl;

    Eigen::Vector3f rotation_axis(std::cos(SUG_PHI - M_PI / 2), std::sin(SUG_PHI - M_PI / 2), 0);
    Eigen::AngleAxisf aa(SUG_THETA - M_PI / 2, rotation_axis);
    Eigen::Matrix3f rotation = aa.matrix();
    Eigen::Vector3f hoge_axis(1, 1, 1);
    aa = Eigen::AngleAxisf(17 * M_PI / 180, hoge_axis.normalized());
    rotation = aa.matrix() * rotation;

    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    
    ::FindRigidTransformationICP(ref_points, tar_points, rotation, translation);
   
    
    vol_ref.WriteRaw("vol_ref.raw");
    std::cout << "performing transforms" << std::endl;;
    Eigen::Matrix3f rot_inv = rotation.inverse();
    Mat3 rot_inv_m3 = ::FromEigen(rot_inv);
    Vec3 tr_v3 = ::FromEigen(translation);
    Volume vol_tar_moved = cuda_affineinv(vol_tar, vol_ref, rot_inv_m3, tr_v3, 0, 3);
    Volume aff_tar_moved = cuda_affineinv(aff_tar, vol_ref, rot_inv_m3, tr_v3, 0xffff);
    Volume aff_ref_resampled = cuda_affineinv(aff_ref, vol_ref, Mat3::Eye(), {0, 0, 0}, 0xffff); 
    vol_tar_moved.WriteRaw("vol_tar_moved.raw");
    //blend volumes and generate colormap
    std::cout << "blending volumes" << std::endl;
    Volume blend;
    blend.CopyShape(vol_ref);
    std::vector<uint8_t> colormap(blend.Nx() * blend.Ny() * blend.Nz() * 3, 0);
    size_t colormap_ix = 0;
    for (auto ix : blend.Range()) {
        float ct_sum = 0;
        float w_ref = 0, w_tar = 0;

        w_ref = 65535.0f - aff_ref_resampled.at(ix);
        ct_sum += w_ref * vol_ref.at(ix);
        w_tar = 65535.0f - aff_tar_moved.at(ix);
        ct_sum += w_tar * (vol_tar_moved.at(ix));

        if (w_ref + w_tar == 0.0f) {
            blend.at(ix) = 0;
        }
        else {
            blend.at(ix) = (uint16_t)(ct_sum / (w_ref + w_tar));
        }

        if (blend.at(ix) >= 0) {
            float h = 1.0f - w_tar / (w_tar  + w_ref) / 3.0f;
            float x = 1.0f / (1 + exp(-1.5e-4f * ((w_ref + w_tar) / 2 - 12000)));
            float s = x;
            float v = 1.0f;
            float r, g, b;
            ::hsvtorgb(2 * (1 - h), s, v, r, g, b);
            colormap[colormap_ix] = (uint8_t)(255 * r);
            colormap[colormap_ix + 1] = (uint8_t)(255 * g);
            colormap[colormap_ix + 2] = (uint8_t)(255 * b);
        }
        colormap_ix += 3;
    }
    aff_tar_moved.WriteRaw("aff_tar.raw");
    aff_ref_resampled.WriteRaw("aff_ref.raw");
    std::ofstream ofs("colormap.raw", std::ios::binary);
    ofs.write((char *)colormap.data(), blend.Nx() * blend.Ny() * blend.Nz() * 3);
    blend.WriteRaw("blend.raw");
}
