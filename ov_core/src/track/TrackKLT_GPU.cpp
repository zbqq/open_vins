#include "TrackKLT_GPU.h"


void TrackKLT_GPU::feed_monocular(double timestamp, cv::Mat &img, size_t cam_id) {

    // Start timing
    rT1 =  boost::posix_time::microsec_clock::local_time();


    pts_last[cam_id].clear();
    ids_last[cam_id].clear();

    // Histogram equalize
    cv::cuda::GpuMat d_img(img);
    cv::cuda::equalizeHist(d_img, d_img);
    d_img.download(img);

    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    if(pts_last[cam_id].empty()) {
        perform_detection_monocular(d_img, d_pts_last[cam_id], pts_last[cam_id], ids_last[cam_id]);
        img_last[cam_id] = img.clone();
        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    perform_detection_monocular(d_img, d_pts_last[cam_id], pts_last[cam_id], ids_last[cam_id]);
    rT2 =  boost::posix_time::microsec_clock::local_time();

    //===================================================================================
    //===================================================================================

    // Debug
    ROS_INFO("current points cam %d = %d",(int)cam_id,(int)pts_last[cam_id].size());

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    std::vector<cv::KeyPoint> pts_left_new;// = pts_last[cam_id];

    // Lets track temporally
    //perform_matching(img_last[cam_id],img,pts_last[cam_id],pts_left_new,mask_ll);
    rT3 =  boost::posix_time::microsec_clock::local_time();

    //===================================================================================
    //===================================================================================

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if(mask_ll.empty()) {
        img_last[cam_id] = img.clone();
        pts_last[cam_id].clear();
        ids_last[cam_id].clear();
        ROS_ERROR("[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....");
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;

    // Loop through all left points
    for(size_t i=0; i<pts_left_new.size(); i++) {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if(pts_left_new[i].pt.x < 0 || pts_left_new[i].pt.y < 0)
            continue;
        // If it is a good track, and also tracked from left to right
        if(mask_ll[i]) {
            good_left.push_back(pts_left_new[i]);
            good_ids_left.push_back(ids_last[cam_id][i]);
        }
    }


    //===================================================================================
    //===================================================================================


    // Update our feature database, with theses new observations
    for(size_t i=0; i<good_left.size(); i++) {
        cv::Point2f npt_l = undistort_point(good_left.at(i).pt, cam_id);
        database->update_feature(good_ids_left.at(i), timestamp, cam_id,
                                 good_left.at(i).pt.x, good_left.at(i).pt.y,
                                 npt_l.x, npt_l.y);
    }

    // Move forward in time
    img_last[cam_id] = img.clone();
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    rT5 =  boost::posix_time::microsec_clock::local_time();

    // Timing information
    //ROS_INFO("[TIME-KLT]: %.4f seconds for detection",(rT2-rT1).total_microseconds() * 1e-6);
    //ROS_INFO("[TIME-KLT]: %.4f seconds for temporal klt",(rT3-rT2).total_microseconds() * 1e-6);
    //ROS_INFO("[TIME-KLT]: %.4f seconds for feature DB update (%d features)",(rT5-rT3).total_microseconds() * 1e-6, (int)good_left.size());
    //ROS_INFO("[TIME-KLT]: %.4f seconds for total",(rT5-rT1).total_microseconds() * 1e-6);


}


void TrackKLT_GPU::feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right) {



}

void TrackKLT_GPU::perform_detection_monocular(const cv::cuda::GpuMat &d_img0, cv::cuda::GpuMat &d_pts0,
                                                std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0) {


    // First compute how many more features we need to extract from this image
    int num_featsneeded = num_features - (int)pts0.size();

    // If we don't need any features, just return
    if(num_featsneeded < 1)
        return;

    // Extract our features (use fast with griding)
    cv::cuda::GpuMat d_pts0_ext;
    //Grider_FAST::perform_griding(img0, pts0_ext, num_featsneeded, grid_x, grid_y, threshold, true);
    cv::Ptr<cv::cuda::FastFeatureDetector> detector = cv::cuda::FastFeatureDetector::create(threshold, true, cv::FastFeatureDetector::TYPE_9_16, num_featsneeded);
    //cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create(1000, 1.2f, 8, 31, 0, 2, 0, 31, 5, true);
    //detector->detectAsync(d_img0, d_pts0_ext);

    // Now detect and download back to this host cpu
    std::vector<cv::KeyPoint> pts0_ext;
    //detector->convert(d_pts0_ext, pts0_ext_3000);
    detector->detect(d_img0, pts0_ext);

//    // Random shuffle and select the points we need
//    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//    std::shuffle(pts0_ext_3000.begin(), pts0_ext_3000.end(), std::default_random_engine(seed));
//    std::vector<cv::KeyPoint> pts0_ext;
//    //std::copy(pts0_ext_3000.begin(),pts0_ext_3000.begin()+((num_featsneeded>(int)pts0_ext_3000.size())?pts0_ext_3000.size():num_featsneeded), pts0_ext.begin());
//    for(size_t i=0; i<pts0_ext_3000.size() && (int)i<num_featsneeded; i++) {
//        pts0_ext.push_back(pts0_ext_3000.at(i));
//    }

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less then grid_px_size points away then existing features
    Eigen::MatrixXd grid_2d;
    grid_2d.resize((int)(d_img0.cols/min_px_dist)+10,(int)(d_img0.rows/min_px_dist)+10);
    for(auto& kpt : pts0) {
        grid_2d((int)(kpt.pt.x/min_px_dist),(int)(kpt.pt.y/min_px_dist)) = 1;
    }

    // Now, reject features that are close a current feature
    for(auto& kpt : pts0_ext) {
        // See if there is a point at this location
        if(grid_2d((int)(kpt.pt.x/min_px_dist),(int)(kpt.pt.y/min_px_dist)) == 1)
            continue;
        // Update the grid as this location being taken
        grid_2d((int)(kpt.pt.x/min_px_dist),(int)(kpt.pt.y/min_px_dist)) = 1;
        // Update our "active points" vector!
        pts0.push_back(kpt);
        ids0.push_back(currid++);
    }

    // Finally upload this new matrix of keypoints
    upload(pts0, d_pts0);


}


void TrackKLT_GPU::perform_detection_stereo(const cv::cuda::GpuMat &d_img0, const cv::cuda::GpuMat &d_img1,
                                            cv::cuda::GpuMat &d_pts0, cv::cuda::GpuMat &d_pts1,
                                            std::vector<size_t> &ids0, std::vector<size_t> &ids1) {


}


void TrackKLT_GPU::perform_matching(const cv::cuda::GpuMat &d_img0, const cv::cuda::GpuMat &d_img1,
                                    cv::cuda::GpuMat &d_pts0, cv::cuda::GpuMat &d_pts1,
                                    std::vector<uchar> &mask_out) {

//    // We must have equal vectors
//    assert(kpts0.size() == kpts1.size());
//
//    // Return if we don't have any points
//    if(kpts0.empty() || kpts1.empty())
//        return;
//
//    // Convert keypoints into points (stupid opencv stuff)
//    std::vector<cv::Point2f> pts0, pts1;
//    for(size_t i=0; i<kpts0.size(); i++) {
//        pts0.push_back(kpts0.at(i).pt);
//        pts1.push_back(kpts1.at(i).pt);
//    }
//
//    // If we don't have enough points for ransac just return empty
//    // We set the mask to be all zeros since all points failed RANSAC
//    if(pts0.size() < 10) {
//        for(size_t i=0; i<pts0.size(); i++)
//            mask_out.push_back((uchar)0);
//        return;
//    }
//
//    // Now do KLT tracking to get the valid new points
//    int pyr_levels = 3;
//    std::vector<uchar> mask_klt;
//    std::vector<float> error;
//    cv::Size win_size(11, 11);
//    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01);
//    cv::calcOpticalFlowPyrLK(img0, img1, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
//
//    // Do RANSAC outlier rejection
//    std::vector<uchar> mask_rsc;
//    cv::findFundamentalMat(pts0, pts1, cv::FM_RANSAC, 1, 0.999, mask_rsc);
//
//    // Loop through and record only ones that are valid
//    for(size_t i=0; i<mask_klt.size(); i++) {
//        auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i])? 1 : 0);
//        mask_out.push_back(mask);
//    }
//
//    // Copy back the updated positions
//    for(size_t i=0; i<pts0.size(); i++) {
//        kpts0.at(i).pt = pts0.at(i);
//        kpts1.at(i).pt = pts1.at(i);
//    }



}



