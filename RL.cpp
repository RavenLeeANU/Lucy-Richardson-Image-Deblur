#include "RL.h"

using namespace std;

int g_linewdt = 1;
double g_epsKernel = 2.2204e-16;			//运动模糊核补偿参数
float g_epsRL = 0.01f;

CRLDeconv::CRLDeconv()
{

}

CRLDeconv::~CRLDeconv()
{

}

IplImage* CRLDeconv::RLdeconvolution(IplImage* imageFloat, CvMat* psf, int iterNum)
{

	//判断是否为空指针
	if (imageFloat == NULL)
	{
		return NULL;
	}

	//初始化路径点
	IplImage* Y = cvCreateImage(cvGetSize(imageFloat), imageFloat->depth, 1); //预测点
	IplImage* J1 = cvCloneImage(imageFloat);//前一次迭代点
	IplImage* J2 = cvCloneImage(imageFloat);//前两次迭代点
	IplImage* wI = imageFloat;//模糊图像

	//IplImage* reBlurred = cvCreateImage(cvGetSize(imageFloat), imageFloat->depth, 1); //RL函数中的第一次卷积结果
	IplImage* imR = cvCreateImage(cvGetSize(imageFloat), imageFloat->depth, 1); //RL函数中的第二次卷积结果

	//计算迭代加速因子的相关变量
	IplImage* T1 = cvCreateImage(cvGetSize(imageFloat), imageFloat->depth, imageFloat->nChannels);
	IplImage* T2 = cvCreateImage(cvGetSize(imageFloat), imageFloat->depth, imageFloat->nChannels);

	double sum1 = 0;
	double sum2 = 0;

	double lambda = 0;			//迭代加速因子

	//赋值操作使用的指针
	float* ptrJ1 = NULL;
	float* ptrJ2 = NULL;
	float* ptrY = NULL;
	float* ptrT1 = NULL;
	float* ptrT2 = NULL;
	float* ptrImR = NULL;
	float* ptrRe = NULL;
	float* ptrWI = NULL;

	//RL迭代
	bool ping = true;//乒乓操作flag
	for (int it = 0; it < iterNum; it++)
	{
		sum1 = 0;
		sum2 = 0;

		//需要有前2次的迭代点才能计算出加速因子
		if (it > 1 && ping)
		{
			//计算加速因子
			for (int i = 0; i < T1->height; ++i)
			{
				ptrT1 = (float*)(T1->imageData + i * T1->widthStep);
				ptrT2 = (float*)(T2->imageData + i * T2->widthStep);

				for (int j = 0; j < T1->width; ++j){
					sum1 += ptrT1[j] * ptrT2[j];
					sum2 += ptrT2[j] * ptrT2[j];
				}
			}
			lambda = sum1 / (sum2 + g_epsRL);
		}
		else if (it > 1 && !ping){
			for (int i = 0; i < T1->height; ++i)
			{
				ptrT1 = (float*)(T1->imageData + i * T1->widthStep);
				ptrT2 = (float*)(T2->imageData + i * T2->widthStep);

				for (int j = 0; j < T1->width; ++j){
					sum1 += ptrT2[j] * ptrT1[j];
					sum2 += ptrT1[j] * ptrT1[j];
				}
			}
			lambda = sum1 / (sum2 + g_epsRL);
		}

		if (ping){
			//计算预测点
			//cvSub(J1, J2, temp);
			//cvScaleAdd(temp, cvScalarAll(lambda), J1, Y);

			for (int i = 0; i < Y->height; ++i)
			{
				ptrJ1 = (float*)(J1->imageData + i * J1->widthStep);
				ptrJ2 = (float*)(J2->imageData + i * J2->widthStep);
				ptrY = (float*)(Y->imageData + i * Y->widthStep);

				for (int j = 0; j < Y->width; ++j)
				{
					ptrY[j] = (lambda * (ptrJ1[j] - ptrJ2[j]) + ptrJ1[j]);
					if (ptrY[j] < 0)//判断是否有负值，如果有则置为零
					{
						ptrY[j] = 0;
					}
				}
			}

			//RL函数中的第一次卷积
			cvFilter2D(Y, T1, psf);

			//cvFilter2D(Y, J2, psf);
			//reBlurred = J2;

			//RL函数中的第二次卷积
			for (int i = 0; i < wI->height; ++i)
			{
				ptrRe = (float*)(T1->imageData + i * T1->widthStep);
				ptrWI = (float*)(wI->imageData + i * wI->widthStep);
				ptrImR = (float*)(imR->imageData + i * imR->widthStep);

				for (int j = 0; j < wI->width; ++j)
				{
					ptrImR[j] = ptrWI[j] / (ptrRe[j] + g_epsRL);
					ptrRe[j] = ptrImR[j] + g_epsRL;
				}
			}

			cvFilter2D(T1, imR, psf);
			//cvFilter2D(reBlurred, T2, psf);
			//imR = T2;

			for (int i = 0; i < Y->height; ++i)
			{
				ptrJ1 = (float*)(J1->imageData + i * J1->widthStep);
				ptrJ2 = (float*)(J2->imageData + i * J2->widthStep);
				ptrImR = (float*)(imR->imageData + i * imR->widthStep);
				ptrY = (float*)(Y->imageData + i * Y->widthStep);
				ptrT1 = (float*)(T1->imageData + i * T1->widthStep);
				ptrT2 = (float*)(T2->imageData + i * T2->widthStep);

				for (int j = 0; j < Y->width; ++j)
				{
					//ptrJ2[j] = ptrJ1[j];
					ptrJ2[j] = ptrImR[j] * ptrY[j];

					//ptrT2[j] = ptrT1[j];
					ptrT2[j] = ptrJ1[j] - ptrY[j];
				}
			}
		}
		else{
			//计算预测点
			//cvSub(J1, J2, temp);
			//cvScaleAdd(temp, cvScalarAll(lambda), J1, Y);

			for (int i = 0; i < Y->height; ++i)
			{
				ptrJ1 = (float*)(J1->imageData + i * J1->widthStep);
				ptrJ2 = (float*)(J2->imageData + i * J2->widthStep);
				ptrY = (float*)(Y->imageData + i * Y->widthStep);

				for (int j = 0; j < Y->width; ++j)
				{
					ptrY[j] = (lambda * (ptrJ2[j] - ptrJ1[j]) + ptrJ2[j]);
					if (ptrY[j] < 0)//判断是否有负值，如果有则置为零
					{
						ptrY[j] = 0;
					}
				}
			}

			//RL函数中的第一次卷积
			cvFilter2D(Y, T2, psf);

			//cvFilter2D(Y, J2, psf);
			//reBlurred = J2;

			//RL函数中的第二次卷积
			for (int i = 0; i < wI->height; ++i)
			{
				ptrRe = (float*)(T2->imageData + i * T2->widthStep);
				ptrWI = (float*)(wI->imageData + i * wI->widthStep);
				ptrImR = (float*)(imR->imageData + i * imR->widthStep);

				for (int j = 0; j < wI->width; ++j)
				{
					ptrImR[j] = ptrWI[j] / (ptrRe[j] + g_epsRL);
					ptrRe[j] = ptrImR[j] + g_epsRL;
				}
			}

			cvFilter2D(T2, imR, psf);
			//cvFilter2D(reBlurred, T2, psf);
			//imR = T2;

			for (int i = 0; i < Y->height; ++i)
			{
				ptrJ1 = (float*)(J1->imageData + i * J1->widthStep);
				ptrJ2 = (float*)(J2->imageData + i * J2->widthStep);
				ptrImR = (float*)(imR->imageData + i * imR->widthStep);
				ptrY = (float*)(Y->imageData + i * Y->widthStep);
				ptrT1 = (float*)(T1->imageData + i * T1->widthStep);
				ptrT2 = (float*)(T2->imageData + i * T2->widthStep);

				for (int j = 0; j < Y->width; ++j)
				{
					//ptrJ2[j] = ptrJ1[j];
					ptrJ1[j] = ptrImR[j] * ptrY[j];

					//ptrT2[j] = ptrT1[j];
					ptrT1[j] = ptrJ1[j] - ptrY[j];
				}
			}
		}
		ping = !ping;

	}

	//释放内存
	cvReleaseImage(&T1);
	cvReleaseImage(&T2);
	cvReleaseImage(&Y);
	cvReleaseImage(&J2);
	//cvReleaseImage(&reBlurred);
	cvReleaseImage(&imR);

	return J1;
}

int CRLDeconv::sign(float x)
{
	if (x < 0)
	{
		return -1;
	}
	else if (abs(x) < FLT_EPSILON)
	{
		return 0;
	}

	return 1;
}

void CRLDeconv::GetMeshGrid(int xStart, int xEnd, int yStart, int yEnd, CvMat** X, CvMat** Y)
{
	//如果预先开辟了内存就先释放掉，避免内存泄漏
	if (*X)
	{
		cvReleaseMat(X);
	}
	if (*Y)
	{
		cvReleaseMat(Y);
	}


	int xLength = abs(xStart - xEnd) + 1;
	int yLength = abs(yStart - yEnd) + 1;


	vector<int> tempX, tempY;

	if (xStart > xEnd)
	{
		for (int i = xStart; i >= xEnd; i--)
		{
			tempX.push_back(i);
		}
	}
	else
	{
		for (int i = xStart; i <= xEnd; i++)
		{
			tempX.push_back(i);
		}
	}


	for (int j = yStart; j <= yEnd; j++)
	{
		tempY.push_back(j);
	}

	CvMat* matX = cvCreateMat(1, xLength, CV_32S);
	CvMat* matY = cvCreateMat(yLength, 1, CV_32S);

	for (int i = 0; i < xLength; i++)
	{
		CV_MAT_ELEM(*matX, int, 0, i) = tempX[i];
	}

	for (int i = 0; i < yLength; i++)
	{
		CV_MAT_ELEM(*matY, int, i, 0) = tempY[i];
	}

	*X = cvCreateMat(yLength, xLength, CV_32S);
	*Y = cvCreateMat(yLength, xLength, CV_32S);

	cvRepeat(matX, *X);
	cvRepeat(matY, *Y);

	//释放内存
	cvReleaseMat(&matX);
	cvReleaseMat(&matY);

}

CvMat* CRLDeconv::GetPSF(float angle, int length)
{
	/*计算模糊核的旋转半径len以及弧度制角度phi，设置模糊核宽度*/
	float len = max(length, 1);//不到1按1算
	float half = (len - 1) / 2;
	float multiple = angle / 180;
	float phi = (multiple - int(multiple)) * CV_PI;//换成弧度制

	float cosphi = cos(phi);
	float sinphi = sin(phi);
	int xsign = sign(cosphi);
	int linewdt = g_linewdt;

	/*根据旋转半径和角度计算四分之一模糊核网格*/
	int sx = int(half * cosphi + linewdt * xsign - len * g_epsKernel);
	int sy = int(half * sinphi + linewdt - len * g_epsKernel);

	//得到网格坐标点
	CvMat* X = NULL;
	CvMat* Y = NULL;
	GetMeshGrid(0, sx, 0, sy, &X, &Y);

	/*计算网络点到模糊方向的距离*/
	CvMat* dist2line = cvCreateMat(X->rows, X->cols, CV_32F);

	CvMat* XFloat = cvCreateMat(X->rows, X->cols, CV_32F);
	CvMat* YFloat = cvCreateMat(Y->rows, Y->cols, CV_32F);

	cvConvert(X, XFloat);
	cvConvert(Y, YFloat);

	//向量乘以标量
	CvMat* zeros = cvCreateMat(XFloat->rows, XFloat->cols, CV_32F);

	CvMat* sinX = cvCreateMat(X->rows, X->cols, CV_32F);
	CvMat* cosY = cvCreateMat(Y->rows, Y->cols, CV_32F);

	cvScaleAdd(XFloat, cvScalarAll(sinphi), zeros, sinX);
	cvScaleAdd(YFloat, cvScalarAll(cosphi), zeros, cosY);
	cvSub(cosY, sinX, dist2line);

	/*计算网格点到原点的距离*/
	CvMat* rad = cvCreateMat(X->rows, X->cols, CV_32F);
	CvMat* radSquare = cvCreateMat(X->rows, X->cols, CV_32F);

	CvMat* XSquare = cvCreateMat(X->rows, X->cols, CV_32F);
	cvMul(XFloat, XFloat, XSquare);
	CvMat* YSquare = cvCreateMat(Y->rows, Y->cols, CV_32F);
	cvMul(YFloat, YFloat, YSquare);

	cvAdd(XSquare, YSquare, radSquare);
	cvPow(radSquare, rad, 0.5);

	/*定位异常网格点*/
	std::vector<int> lastpixIndex;
	for (int i = 0; i < rad->rows; i++)
	{
		for (int j = 0; j < rad->cols; j++)
		{

			if ((CV_MAT_ELEM(*rad, float, i, j)  > half * (1 - FLT_EPSILON))
				&& (abs(CV_MAT_ELEM(*dist2line, float, i, j)) < linewdt * (1 + FLT_EPSILON)))
			{
				lastpixIndex.push_back(i * rad->cols + j);
			}
		}
	}

	/*处理异常网格点到模糊方向的距离*/
	for (int i = 0; i < lastpixIndex.size(); i++)
	{
		int index = lastpixIndex[i];
		int rowIndex = index / X->cols;
		int colIndex = index % X->cols;

		float x2lastpixTemp = half - abs((CV_MAT_ELEM(*XFloat, float, rowIndex, colIndex)
			+ CV_MAT_ELEM(*dist2line, float, rowIndex, colIndex)*sinphi) / cosphi);

		float dist2lineTemp = CV_MAT_ELEM(*dist2line, float, rowIndex, colIndex);

		CV_MAT_ELEM(*dist2line, float, rowIndex, colIndex) = sqrt(dist2lineTemp * dist2lineTemp
			+ x2lastpixTemp * x2lastpixTemp);
	}

	CvMat* dist2lineAbs = cvCloneMat(dist2line);
	for (int i = 0; i < dist2lineAbs->rows; i++)
	{
		for (int j = 0; j < dist2lineAbs->cols; j++)
		{
			CV_MAT_ELEM(*dist2lineAbs, float, i, j) = abs(CV_MAT_ELEM(*dist2line, float, i, j));
		}
	}

	/*确定权重*/
	CvMat* tempMat = cvCreateMat(dist2line->rows, dist2line->cols, CV_32F);
	CvMat* tempMat2 = cvCreateMat(dist2line->rows, dist2line->cols, CV_32F);
	cvSet(tempMat, cvScalarAll(linewdt));
	cvAddS(tempMat, cvScalarAll(g_epsKernel), tempMat2);

	cvSub(tempMat2, dist2lineAbs, dist2line);

	//把除了线宽之外的值都置为零
	for (int i = 0; i < dist2line->rows; i++)
	{
		for (int j = 0; j < dist2line->cols; j++)
		{
			if (CV_MAT_ELEM(*dist2line, float, i, j) < 0)
			{
				CV_MAT_ELEM(*dist2line, float, i, j) = 0;
			}
		}
	}

	/*填充并且得到完整模糊核*/
	CvMat* h = cvCreateMat(dist2line->rows * 2 - 1, dist2line->cols * 2 - 1, CV_32F);
	cvZero(h);

	CvMat* hTL = cvCreateMat(dist2line->rows, dist2line->cols, CV_32F);
	cvFlip(dist2line, hTL, -1);//关于原点做180度中心对称

	CvMat* hBR = cvCreateMat(dist2line->rows, dist2line->cols, CV_32F);
	cvCopy(dist2line, hBR);

	//给模糊核分块赋值
	for (int i = 0; i < dist2line->rows; i++)
	{
		for (int j = 0; j < dist2line->cols; j++)
		{
			CV_MAT_ELEM(*h, float, i, j) = CV_MAT_ELEM(*hTL, float, i, j);
		}
	}
	for (int i = 0; i < dist2line->rows; i++)
	{
		for (int j = 0; j < dist2line->cols; j++)
		{
			CV_MAT_ELEM(*h, float, i + dist2line->rows - 1, j + dist2line->cols - 1) = CV_MAT_ELEM(*hBR, float, i, j);
		}
	}

	/*归一化，判断旋转*/
	CvMat* hNorm = cvCreateMat(h->rows, h->cols, CV_32F);
	CvScalar sumH = cvSum(h);
	CvMat* temp = cvCreateMat(h->rows, h->cols, CV_32F);
	cvSet(temp, cvScalarAll(sumH.val[0] + g_epsKernel * len *len));
	cvDiv(h, temp, hNorm);


	if (cosphi > 0)
	{
		cvFlip(hNorm, h, 1);
	}


	//释放内存
	cvReleaseMat(&X);
	cvReleaseMat(&Y);
	cvReleaseMat(&XFloat);
	cvReleaseMat(&YFloat);
	cvReleaseMat(&dist2line);
	cvReleaseMat(&rad);
	cvReleaseMat(&radSquare);

	cvReleaseMat(&zeros);
	cvReleaseMat(&sinX);
	cvReleaseMat(&cosY);
	cvReleaseMat(&XSquare);
	cvReleaseMat(&YSquare);
	cvReleaseMat(&dist2lineAbs);
	cvReleaseMat(&tempMat);
	cvReleaseMat(&tempMat2);

	cvReleaseMat(&hTL);
	cvReleaseMat(&hBR);
	cvReleaseMat(&hNorm);
	cvReleaseMat(&temp);


	return h;

}


