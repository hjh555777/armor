#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

struct LightBar
{
    RotatedRect rect;
};

bool sortByX(const LightBar &a, const LightBar &b)
{
    return a.rect.center.x < b.rect.center.x;
}

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "摄像头打开失败！" << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame))
    {
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Mat mask;
        inRange(hsv, Scalar(90, 120, 120), Scalar(120, 255, 255), mask);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(mask, mask, kernel);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<LightBar> bars;
        for (auto &cnt : contours)
        {
            if (contourArea(cnt) < 20)
                continue;
            RotatedRect r = minAreaRect(cnt);
            float w = r.size.width;
            float h = r.size.height;
            if (h < w)
                swap(w, h);
            float ratio = h / (w + 1e-6);
            if (ratio > 2.5 && ratio < 10)
            {
                LightBar bar;
                bar.rect = r;
                bars.push_back(bar);
            }
        }

        if (bars.size() >= 2)
        {
            sort(bars.begin(), bars.end(), sortByX);
            for (int i = 0; i < (int)bars.size() - 1; i++)
            {
                LightBar left = bars[i];
                LightBar right = bars[i + 1];
                float dy = abs(left.rect.center.y - right.rect.center.y);
                float avgH = (left.rect.size.height + right.rect.size.height) / 2;
                if (dy < avgH * 0.4)
                {
                    Point2f armorCenter = (left.rect.center + right.rect.center) / 2;
                    vector<Point2f> pts;
                    pts.push_back(left.rect.points[0]);
                    pts.push_back(left.rect.points[1]);
                    pts.push_back(right.rect.points[2]);
                    pts.push_back(right.rect.points[3]);
                    vector<Point> poly;
                    for (auto p : pts)
                        poly.push_back(Point(p));
                    polylines(frame, poly, true, Scalar(0, 255, 0), 2);
                    circle(frame, armorCenter, 3, Scalar(0, 0, 255), -1);

                    Point2f imgCenter(frame.cols / 2.0, frame.rows / 2.0);
                    float dx = armorCenter.x - imgCenter.x;
                    float dy = armorCenter.y - imgCenter.y;
                    cout << "装甲偏移 dx:" << dx << " dy:" << dy << endl;

                    Rect roiRect(
                        armorCenter.x - avgH * 0.4,
                        armorCenter.y - avgH * 0.4,
                        avgH * 0.8,
                        avgH * 0.8);
                    roiRect &= Rect(0, 0, frame.cols, frame.rows);
                    Mat numRoi = frame(roiRect);
                    imshow("数字ROI", numRoi);
                }
            }
        }

        imshow("原始画面", frame);
        imshow("颜色掩码", mask);
        if (waitKey(1) == 27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
