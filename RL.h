#include "..\..\Include\OpenSource\OpenCV\OpenCV2.4.11\opencv.hpp"

class CRLDeconv
{
public:
	CRLDeconv();
	~CRLDeconv();

	IplImage* RLdeconvolution(IplImage* imageFloat, CvMat* psf, int iterNum);		//Richardson-Lucy�����ȥģ��
	CvMat* GetPSF(float angle, int length);		//���ݽǶȺͳ��ȵõ��˶�ģ����

protected:
	int sign(float x);				//���ź���
	void GetMeshGrid(int xStart, int xEnd, int yStart, int yEnd, CvMat** X, CvMat** Y);		//�õ����������
	

};
