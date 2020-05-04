#include "..\..\Include\OpenSource\OpenCV\OpenCV2.4.11\opencv.hpp"

class CRLDeconv
{
public:
	CRLDeconv();
	~CRLDeconv();

	IplImage* RLdeconvolution(IplImage* imageFloat, CvMat* psf, int iterNum);		//Richardson-Lucy反卷积去模糊
	CvMat* GetPSF(float angle, int length);		//根据角度和长度得到运动模糊核

protected:
	int sign(float x);				//符号函数
	void GetMeshGrid(int xStart, int xEnd, int yStart, int yEnd, CvMat** X, CvMat** Y);		//得到网格点坐标
	

};
