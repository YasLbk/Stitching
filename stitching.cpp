#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

int cornerHarris_Blocksize = 2;
int cornerHarris_aperturesize = 3;
double cornerHarris_k = 0.04;


/**
 * @brief IF DEBUG flag is true display the intermidiate results
 *
 */
#define DEBUG false

/**
 * @brief Display the image
 *
 * @param img: input opencv image
 * @param title: title string of the window
 */
void display_image(const Mat &img, string title = "Display_Window") {
  namedWindow(title, WINDOW_AUTOSIZE);
  imshow(title, img);
}

/**
 * @brief read the input images from the command line arguments; assuming only
 * two input images
 *
 * @param inputs : output array of Mat
 * @param argv   : command line arguments
 */

void read_input_images(vector<Mat> &inputs, char **argv) {
  inputs[0] = imread(argv[1], IMREAD_COLOR);
  inputs[1] = imread(argv[2], IMREAD_COLOR);
  if (!inputs[0].data || !inputs[1].data) {
    cout << "Check the input images \n";
    exit(0);
  }

  cout << "Displaying input images\n";
  display_image(inputs[0], "im-1");
  display_image(inputs[1], "im-2");
}

/**
 * @brief  detect corners
 *
 * @param imgt1 : input image
 * @param window_name
 * @param corners : pointer to a vector of corners
 * @param threshold threshold value filtering corner detection
 * 
 */
void detect_corners(Mat& imgt1, string window_name, vector<Point> *corners,
                    int threshold) {
  Mat src, dst, normalized, scaled;
  cvtColor(imgt1, src, 6, 1);

  cornerHarris(src, dst, cornerHarris_Blocksize, cornerHarris_aperturesize,
               cornerHarris_k, BORDER_DEFAULT);

  normalize(dst, normalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
  // convertScaleAbs(normalized, scaled);
  for (int i = 0; i < normalized.rows; i++) {
    for (int j = 0; j < normalized.cols; j++) {
      if ((int)normalized.at<float>(i, j) > threshold) {
        circle(imgt1, Point(j, i), 5, Scalar(10, 10, 255), 2, 8, 0);
        corners->push_back(Point(j, i));
        // cout << corners->size() << endl;
        circle(scaled, Point(j, i), 5, Scalar(0, 10, 255), 2, 8, 0);
      }
    }
  }
  // display_image(imgt1, window_name);
}

/**
 * @brief  get matches between corners of both img
 *
 * @param inputs : input images
 * @param corners : vector of corners for each image
 * @param matches : stores pairs of matching corners
 * @param THRESH_SUM : filters better matching points
 * @param window_size 
 * 
 */
void get_matches(const vector<Mat> &inputs, vector<vector<Point>> &corners,
                 vector<pair<int, int>> &matches, float THRESH_SUM = 7000,
                 int window_size = 7) {
  Point w(window_size, window_size);
  vector<vector<int>> asso(corners.size());
  // for a corner i
  for (int i = 0; i < corners.size(); ++i) {
    asso[i].resize(corners[i].size(), -1);
    for (int j = 0; j < corners[i].size(); ++j) {
      // for a corner j
      if (corners[i][j].x > (window_size + 2) &&
          corners[i][j].x < inputs[i].size().width - (window_size - 2) &&
          corners[i][j].y > (window_size + 2) &&
          corners[i][j].y < inputs[i].size().height - (window_size - 2)) {
        // cout << "j=" << j << "\n";
        Point corner1 = corners[i][j];
        Rect roi1(corner1 - w, corner1 + w);
        for (int k = 0; k < corners.size(); ++k) {
          if (i == k) {
            continue;
          }
          for (int l = 0; l < corners[k].size(); ++l) {
            if (corners[k][l].x > (window_size + 2) &&
                corners[k][l].x < inputs[k].size().width - (window_size - 2) &&
                corners[k][l].y > (window_size + 2) &&
                corners[k][l].y < inputs[k].size().height - (window_size - 2)) {
              Mat diff;
              Point corner2 = corners[k][l];
              Rect roi2(corner2 - w, corner2 + w);

              absdiff(inputs[i](roi1), inputs[k](roi2), diff);
              multiply(diff, diff, diff);
              float sume = sum(diff).val[0];
              //cout << i << j << k << l << sume << endl;
              if (sume < THRESH_SUM) {
                asso[i][j] = l;
              }
            }
          }
        }
      }
    }
  }


  for (int j = 0; j < corners[0].size(); ++j) {
    if (asso[0][j] != -1 && asso[1][asso[0][j]] == j) {
      matches.push_back(pair<int, int>(j, asso[0][j]));
      // circle(inputs[0], corners[0][j] , 100 ,Scalar(0,0,0) );
      // circle(inputs[1], corners[1][asso[0][j]] , 100 ,Scalar(0,0,0) );
    }
  }
}

/**
 * @brief  show merge of Two Images
 *
 * @param imageOne : input images
 * @param imagetwo  
 * 
 */
Mat showTwoImages(Mat &imageOne, Mat &imageTwo) {
  int totalCol = imageOne.cols + imageTwo.cols;
  int totalRow = max(imageOne.rows, imageTwo.rows);

  Mat totalImage(Size(totalCol, totalRow), imageOne.type());

  imageOne.copyTo(totalImage(Rect(0, 0, imageOne.cols, imageOne.rows)));
  imageTwo.copyTo(
      totalImage(Rect(imageOne.cols, 0, imageTwo.cols, imageTwo.rows)));
  display_image(totalImage);
  return totalImage;
}

/**
 * @brief  show Two Images
 *
 * @param inputs : input images
 * @param merger : merge image of both input
 * @param corners : vector of corners for each image
 * @param matches : stores pairs of matching corners
 * 
 */
void vis_matches(vector<Mat> inputs, Mat &merger, vector<vector<Point>> &corners,
                 vector<pair<int, int>> &matches) {
  int offset = inputs[0].cols;
  for (const auto it : matches) {
    int a = it.first;
    int b = it.second;
    //cout << a << "  " << b << endl << endl;
    // circle(merger, corners[0][a], 100, Scalar(0, 0, 0));
    // circle(merger, corners[1][b]+Point(offset,0), 100, Scalar(0, 0, 0));
    line(merger, corners[0][a], corners[1][b] + Point(offset, 0),
         Scalar(0, 255, 0), 2, 8, 0);
  }
  // display_image(merger);
}
/**
 * @brief  thsi function calculates offsets coord : 
 * x and y parametrs in order to merge accordingly both inputs
 *
 * @param inputs : input images
 * @param merger : merge image of both input
 * @param corners : vector of corners for each image
 * @param matches : stores pairs of matching corners
 * @param offset : vector containing translating parameters
 */

void compute_mean_translation(vector<Mat>& inputs, Mat &merger, vector<vector<Point>> corners,
                              vector<pair<int, int>> matches, int offset[2]) {
  int offsetp = inputs[0].cols;
  Point img1;
  Point img2;
  for (const auto it : matches) {
    int a = it.first;
    int b = it.second;
    //cout << a << "  " << b << endl << endl;
    img1 += corners[0][a];
    img2 += corners[1][b] + Point(offsetp, 0);
  }
  int taille = matches.size();
  img1 = img1 / taille;
  img2 = img2 / taille;
  //cout << img1 << img2 << endl;
  circle(merger, img1, 100, Scalar(0, 0, 0));
  circle(merger, img2, 100, Scalar(0, 0, 0));
  offset[0] = img2.x - img1.x;
  offset[1] = img1.y - img2.y;
  
}
/**
 * @brief  this function display each input image at the right
 * place in the global output image using offset/translation param
 *
 * @param inputs : input images
 * @param merger : merge image of both input
 * @param result_stitched : output
 * @param offset : vector containing translating parameters
 */
void vis_stitched(vector<Mat>& inputs, Mat &merger, Mat result_stitched,
                  int offset[2]) {
  int diffx = offset[0];
  int diffy = offset[1];
  Mat output = Mat::zeros(inputs[1].rows + diffy,
                          inputs[0].cols + inputs[1].cols - diffx,
                          merger.type());  // inputs[1].cols + 210

  Rect roi = Rect(0, 0, inputs[0].cols, inputs[0].rows);

  Mat roiImg = output(roi);
  inputs[0].copyTo(roiImg);
  Rect roi2 = Rect(output.cols - inputs[1].cols, output.rows - inputs[1].rows,
                   inputs[1].cols, inputs[1].rows);

  Mat roiImg2 = output(roi2);

  inputs[1].copyTo(roiImg2);

  display_image(output);
  //display_image(merger, "Merge of both");
  result_stitched=output;
}


/**********************************************MAIN*************************/
int main(int argc, char **argv) {
  if (argc < 3) {
    cout << " Usage: ./stitching [required] input_image_01 [required] "
            "input_image_02 [optional] output_image \n";
    return -1;
  }
  // decalre required variables
  vector<Mat> inputs(2);
  vector<vector<Point>> corners(inputs.size());
  vector<pair<int, int>> matches;
  Mat result_stitched;
  int offset[2];

  // read the input images
  read_input_images(inputs, argv);

  // [TODO] write a function to detect corners
  detect_corners(inputs[0], "Detect features 1", &corners[0], 120);
  detect_corners(inputs[1], "Detect features 2", &corners[1], 120);

  get_matches(inputs, corners, matches);
  Mat merger = showTwoImages(inputs[0], inputs[1]);

  // [TODO] write a function to visualize matches
  vis_matches(inputs, merger, corners, matches);

  // [TODO] write a function to compute mean translated point
  compute_mean_translation(inputs, merger, corners, matches, offset);

  // [TODO] write a function to visualize mosaic // assumes only translation
  // motion between the images
  read_input_images(inputs, argv);
  vis_stitched(inputs, merger, result_stitched, offset);

  // write the output image
  if (argc == 4) {
    imwrite(argv[3], result_stitched);
  }
  waitKey(0);
  return 0;
}