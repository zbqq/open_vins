#ifndef OV_CORE_TRACK_KLT_CUDA_H
#define OV_CORE_TRACK_KLT_CUDA_H


#include "TrackBase.h"


#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>


/**
 * @brief KLT tracking of features (with OpenCV GPU acceleration).
 *
 * This is the implementation of a KLT visual frontend for tracking sparse features.
 * **This leverages OpenCV GPU accelerated methods to do the optical flow**
 * **Thus you need to make sure you build with OpenCV with CUDA enabled**
 * We can track either monocular cameras across time (temporally) along with
 * stereo cameras which we also track across time (temporally) but track from left to right
 * to find the stereo correspondence information also.
 * This uses the [calcOpticalFlowPyrLK](https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp) OpenCV function to do the KLT tracking.
 * Reference GPU code is from the [pyrlk_optical_flow](https://github.com/opencv/opencv/blob/master/samples/gpu/pyrlk_optical_flow.cpp) OpenCV example.
 */
class TrackKLT_GPU : public TrackBase
{

public:

    /**
     * @brief Public default constructor
     * @param camera_k map of camera_id => 3x3 camera intrinic matrix
     * @param camera_d  map of camera_id => 4x1 camera distortion parameters
     * @param camera_fisheye map of camera_id => bool if we should do radtan or fisheye distortion model
     */
    TrackKLT_GPU(std::map<size_t,Eigen::Matrix3d> camera_k,
            std::map<size_t,Eigen::Matrix<double,4,1>> camera_d,
            std::map<size_t,bool> camera_fisheye):
            TrackBase(camera_k,camera_d,camera_fisheye),threshold(10),grid_x(8),grid_y(5),min_px_dist(30) {}

    /**
     * @brief Public constructor with configuration variables
     * @param camera_k map of camera_id => 3x3 camera intrinic matrix
     * @param camera_d  map of camera_id => 4x1 camera distortion parameters
     * @param camera_fisheye map of camera_id => bool if we should do radtan or fisheye distortion model
     * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
     * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
     * @param fast_threshold FAST detection threshold
     * @param gridx size of grid in the x-direction / u-direction
     * @param gridy size of grid in the y-direction / v-direction
     * @param minpxdist features need to be at least this number pixels away from each other
     */
    explicit TrackKLT_GPU(std::map<size_t,Eigen::Matrix3d> camera_k,
            std::map<size_t,Eigen::Matrix<double,4,1>> camera_d,
            std::map<size_t,bool> camera_fisheye,
            int numfeats, int numaruco, int fast_threshold, int gridx, int gridy, int minpxdist):
            TrackBase(camera_k,camera_d,camera_fisheye,numfeats,numaruco),threshold(fast_threshold),grid_x(gridx),grid_y(gridy),min_px_dist(minpxdist) {}


    /**
     * @brief Process a new monocular image
     * @param timestamp timestamp the new image occurred at
     * @param img new cv:Mat grayscale image
     * @param cam_id the camera id that this new image corresponds too
     */
    void feed_monocular(double timestamp, cv::Mat &img, size_t cam_id) override;

    /**
     * @brief Process new stereo pair of images
     * @param timestamp timestamp this pair occured at (stereo is synchronised)
     * @param img_left first grayscaled image
     * @param img_right second grayscaled image
     * @param cam_id_left first image camera id
     * @param cam_id_right second image camera id
     */
    void feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right) override;


protected:

    /**
     * @brief Detects new features in the current image
     * @param img0 image we will detect features on (on device)
     * @param pts0 vector of currently extracted keypoints in this image (on device)
     * @param ids0 vector of feature ids for each currently extracted keypoint
     *
     * Given an image and its currently extracted features, this will try to add new features if needed.
     * Will try to always have the "max_features" being tracked through KLT at each timestep.
     * Passed images should already be grayscaled.
     */
    void perform_detection_monocular(const cv::cuda::GpuMat &d_img0, cv::cuda::GpuMat &d_pts0, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0);

    /**
     * @brief Detects new features in the current stereo pair
     * @param img0 left image we will detect features on (on device)
     * @param img1 right image we will detect features on (on device)
     * @param pts0 left vector of currently extracted keypoints (on device)
     * @param pts1 right vector of currently extracted keypoints (on device)
     * @param ids0 left vector of feature ids for each currently extracted keypoint
     * @param ids1 right vector of feature ids for each currently extracted keypoint
     *
     * This does the same logic as the perform_detection_monocular() function, but we also enforce stereo contraints.
     * So we detect features in the left image, and then KLT track them onto the right image.
     * If we have valid tracks, then we have both the keypoint on the left and its matching point in the right image.
     * Will try to always have the "max_features" being tracked through KLT at each timestep.
     */
    void perform_detection_stereo(const cv::cuda::GpuMat &d_img0, const cv::cuda::GpuMat &d_img1, cv::cuda::GpuMat &d_pts0,
                                  cv::cuda::GpuMat &d_pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

    /**
     * @brief KLT track between two images, and do RANSAC afterwards
     * @param img0 starting image (on device)
     * @param img1 image we want to track too (on device)
     * @param pts0 starting points (on device)
     * @param pts1 points we have tracked (on device)
     * @param mask_out what points had valid tracks
     *
     * This will track features from the first image into the second image.
     * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
     * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
     */
    void perform_matching(const cv::cuda::GpuMat &d_img0, const cv::cuda::GpuMat &d_img1, cv::cuda::GpuMat &d_pts0, cv::cuda::GpuMat &d_pts1, std::vector<uchar> &mask_out);


    /**
     * @brief This function will download a 2d vector of points from a GPU matrix
     * @param d_mat Matrix that resides on the GPU device
     * @param vec Vector of 2d points on the HOST device
     */
    static void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec) {
        vec.resize(d_mat.cols);
        cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
        d_mat.download(mat);
    }

    /**
     * @brief This function will upload a 2d vector of points from to a GPU matrix
     * @param vec Vector of 2d points on the HOST device
     * @param d_mat Matrix that resides on the GPU device
     */
    static void upload(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat) {
        cv::Mat mat(1, (int)vec.size(), CV_32FC2);
        for(size_t i=0; i<vec.size(); i++) {
            mat.at<cv::Vec2f>(0,i)[0] = vec.at(i).x;
            mat.at<cv::Vec2f>(0,i)[1] = vec.at(i).y;
        }
        d_mat.upload(mat);
    }

    /**
     * @brief This function will upload a 2d vector of points from to a GPU matrix
     * @param vec Vector of 2d KeyPoints on the HOST device
     * @param d_mat Matrix that resides on the GPU device
     */
    static void upload(const std::vector<cv::KeyPoint>& vec, cv::cuda::GpuMat& d_mat) {
        cv::Mat mat(1, (int)vec.size(), CV_32FC2);
        for(size_t i=0; i<vec.size(); i++) {
            mat.at<cv::Vec2f>(0,i)[0] = vec.at(i).pt.x;
            mat.at<cv::Vec2f>(0,i)[1] = vec.at(i).pt.y;
        }
        d_mat.upload(mat);
    }

    /**
    * @brief This function will download a vector of chars from a GPU matrix
    * @param d_mat Matrix that resides on the GPU device
    * @param vec Vector of chars on the HOST device
    */
    static void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec) {
        vec.resize(d_mat.cols);
        cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
        d_mat.download(mat);
    }

    // Timing variables
    boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

    // Parameters for our FAST grid detector
    int threshold;
    int grid_x;
    int grid_y;

    // Minimum pixel distance to be "far away enough" to be a different extracted feature
    int min_px_dist;

    // Last set of cuda matrices that are on the current device
    std::map<size_t,cv::cuda::GpuMat> d_pts_last;
    std::map<size_t,cv::cuda::GpuMat> d_img_last;

};




#endif /* OV_CORE_TRACK_KLT_CUDA_H */