#include "RL.h"

int main()
{
	IplImage* img = cvLoadImage("C:/Users/54053/Desktop/InterBox/1.jpg");
	
	CRLDeconv deconv;
	CvMat* psf =  deconv.GetPSF(20, 50);
	
	IplImage* imgF = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 3);
	IplImage* ch1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
	IplImage* ch2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
	IplImage* ch3 = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);

	cvConvert(img,imgF);
	cvSplit(imgF, ch1, ch2, ch3, NULL);
	
	IplImage* res1 = deconv.RLdeconvolution(ch1, psf, 10);
	IplImage* res2 = deconv.RLdeconvolution(ch2, psf, 10);
	IplImage* res3 = deconv.RLdeconvolution(ch3, psf, 10);

	cvMerge(res1, res2, res3, NULL, imgF);
	cvConvert(imgF, img);

	cvSaveImage("C:/Users/54053/Desktop/InterBox/final.jpg", img);
	return 0;
}