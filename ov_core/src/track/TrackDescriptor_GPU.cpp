#include "TrackDescriptor_GPU.h"



void TrackDescriptor_GPU::feed_monocular(double timestamp, cv::Mat &img, size_t cam_id) {

    // Start timing
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // Histogram equalize
    cv::cuda::GpuMat d_img(img);
    cv::cuda::equalizeHist(d_img, d_img);
    //d_img.download(img);
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // If we are the first frame (or have lost tracking), initialize our descriptors
    if(pts_last.find(cam_id)==pts_last.end() || pts_last[cam_id].empty()) {
        perform_detection_monocular(d_img, pts_last[cam_id], d_desc_last[cam_id], ids_last[cam_id]);
        img_last[cam_id] = img.clone();
        return;
    }

    // Our new keypoints and descriptor for the new image
    std::vector<cv::KeyPoint> pts_new;
    cv::cuda::GpuMat d_desc_new;
    std::vector<size_t> ids_new;

    // First, extract new descriptors for this new image
    perform_detection_monocular(d_img, pts_new, d_desc_new, ids_new);
    rT2 =  boost::posix_time::microsec_clock::local_time();

    //===================================================================================
    //===================================================================================

    // Our matches temporally
    std::vector<cv::DMatch> matches_ll;

    // Lets match temporally
    robust_match(pts_last[cam_id],pts_new,d_desc_last[cam_id],d_desc_new,matches_ll);
    rT3 =  boost::posix_time::microsec_clock::local_time();


    //===================================================================================
    //===================================================================================

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    cv::Mat good_desc_left;

    // Count how many we have tracked from the last time
    int num_tracklast = 0;

    // Download the extracted descriptors for the last extraction
    cv::Mat desc_new;
    d_desc_new.download(desc_new);

    // Loop through all current left to right points
    // We want to see if any of theses have matches to the previous frame
    // If we have a match new->old then we want to use that ID instead of the new one
    for(size_t i=0; i<pts_new.size(); i++) {

        // Loop through all left matches, and find the old "train" id
        int idll = -1;
        for(size_t j=0; j<matches_ll.size(); j++){
            if(matches_ll[j].trainIdx == (int)i) {
                idll = matches_ll[j].queryIdx;
            }
        }

        // If we found a good stereo track from left to left, and right to right
        // Then lets replace the current ID with the old ID
        // We also check that we are linked to the same past ID value
        if(idll != -1) {
            good_left.push_back(pts_new[i]);
            good_desc_left.push_back(desc_new.row((int)i));
            good_ids_left.push_back(ids_last[cam_id][idll]);
            num_tracklast++;
        } else {
            // Else just append the current feature and its unique ID
            good_left.push_back(pts_new[i]);
            good_desc_left.push_back(desc_new.row((int)i));
            good_ids_left.push_back(ids_new[i]);
        }

    }
    rT4 =  boost::posix_time::microsec_clock::local_time();


    //===================================================================================
    //===================================================================================


    // Update our feature database, with theses new observations
    for(size_t i=0; i<good_left.size(); i++) {
        cv::Point2f npt_l = undistort_point(good_left.at(i).pt, cam_id);
        database->update_feature(good_ids_left.at(i), timestamp, cam_id,
                                 good_left.at(i).pt.x, good_left.at(i).pt.y,
                                 npt_l.x, npt_l.y);
    }

    // Debug info
    //ROS_INFO("LtoL = %d | good = %d | fromlast = %d",(int)matches_ll.size(),(int)good_left.size(),num_tracklast);

    // Upload the new set of descriptors for the next timestep
    cv::cuda::GpuMat d_good_desc_left;
    d_good_desc_left.upload(good_desc_left);

    // Move forward in time
    img_last[cam_id] = img.clone();
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    d_desc_last[cam_id] = d_good_desc_left;
    rT5 =  boost::posix_time::microsec_clock::local_time();

    // Our timing information
    //ROS_INFO("[TIME-DESC]: %.4f seconds for upload",(rT1-rT0).total_microseconds() * 1e-6);
    //ROS_INFO("[TIME-DESC]: %.4f seconds for detection",(rT2-rT1).total_microseconds() * 1e-6);
    //ROS_INFO("[TIME-DESC]: %.4f seconds for matching",(rT3-rT2).total_microseconds() * 1e-6);
    //ROS_INFO("[TIME-DESC]: %.4f seconds for merging",(rT4-rT3).total_microseconds() * 1e-6);
    //ROS_INFO("[TIME-DESC]: %.4f seconds for feature DB update (%d features)",(rT5-rT4).total_microseconds() * 1e-6, (int)good_left.size());
    //ROS_INFO("[TIME-DESC]: %.4f seconds for total",(rT5-rT0).total_microseconds() * 1e-6);


}

void TrackDescriptor_GPU::feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right) {



}


void TrackDescriptor_GPU::perform_detection_monocular(const cv::cuda::GpuMat &d_img0, std::vector<cv::KeyPoint>& pts0,
                                                        cv::cuda::GpuMat &d_desc0, std::vector<size_t>& ids0) {

    // Assert that we need features
    assert(pts0.empty());

    // Extract points and their descriptors
    detector->detect(d_img0, pts0);
    detector->compute(d_img0, pts0, d_desc0);

    // For all good matches, lets append our id vector
    // Set our IDs to be unique IDs here, will later replace with corrected ones, after temporal matching
    for(size_t i=0; i<pts0.size(); i++) {
        currid++;
        ids0.push_back(currid);
    }

}

void TrackDescriptor_GPU::robust_match(std::vector<cv::KeyPoint>& pts0, std::vector<cv::KeyPoint> pts1,
                                       cv::cuda::GpuMat& d_desc0, cv::cuda::GpuMat& d_desc1, std::vector<cv::DMatch>& matches) {

    // Our 1to2 and 2to1 match vectors
    std::vector<std::vector<cv::DMatch> > matches0to1, matches1to0;

    // Match descriptors (return 2 nearest neighbours)
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    matcher->knnMatch(d_desc0, d_desc1, matches0to1, 2);
    matcher->knnMatch(d_desc1, d_desc0, matches1to0, 2);

    // Do a ratio test for both matches
    robust_ratio_test(matches0to1);
    robust_ratio_test(matches1to0);

    // Finally do a symmetry test
    std::vector<cv::DMatch> matches_good;
    robust_symmetry_test(matches0to1, matches1to0, matches_good);

    // Convert points into points for RANSAC
    std::vector<cv::Point2f> pts0_rsc, pts1_rsc;
    for(size_t i=0; i<matches_good.size(); i++) {
        // Get our ids
        int index_pt0 = matches_good.at(i).queryIdx;
        int index_pt1 = matches_good.at(i).trainIdx;
        // Push back just the 2d point
        pts0_rsc.push_back(pts0[index_pt0].pt);
        pts1_rsc.push_back(pts1[index_pt1].pt);
    }

    // If we don't have enough points for ransac just return empty
    if(pts0_rsc.size() < 10)
        return;

    // Do RANSAC outlier rejection
    std::vector<uchar> mask_rsc;
    cv::findFundamentalMat(pts0_rsc, pts1_rsc, cv::FM_RANSAC, 1, 0.99, mask_rsc);

    // Loop through all good matches, and only append ones that have passed RANSAC
    for(size_t i=0; i<matches_good.size(); i++) {
        // Skip if bad ransac id
        if (mask_rsc[i] != 1)
            continue;
        // Else, lets append this match to the return array!
        matches.push_back(matches_good.at(i));
    }


}

void TrackDescriptor_GPU::robust_ratio_test(std::vector<std::vector<cv::DMatch> >& matches) {

    // Loop through all matches
    for(auto matchIterator=matches.begin(); matchIterator!= matches.end(); ++matchIterator)
    {
        // If 2 NN has been identified, else remove this feature
        if (matchIterator->size() > 1) {
            // check distance ratio, remove it if the ratio is larger
            if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > knn_ratio ) {
                matchIterator->clear();
            }
        } else {
            // does not have 2 neighbours, so remove it
            matchIterator->clear();
        }
    }

}

void TrackDescriptor_GPU::robust_symmetry_test(std::vector<std::vector<cv::DMatch> >& matches1,
                                           std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& good_matches) {

    // for all matches image 1 -> image 2
    for (auto matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1) {

        // ignore deleted matches
        if (matchIterator1->empty() || matchIterator1->size() < 2)
            continue;

        // for all matches image 2 -> image 1
        for (auto matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2) {
            // ignore deleted matches
            if (matchIterator2->empty() || matchIterator2->size() < 2)
                continue;

            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
                // add symmetrical match
                good_matches.emplace_back(cv::DMatch((*matchIterator1)[0].queryIdx,(*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance));
                // next match in image 1 -> image 2
                break;
            }
        }
    }

}
