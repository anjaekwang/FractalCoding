#define _CRT_SECURE_NO_WARNINGS


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

#include <algorithm>
#include <time.h>



using namespace std;
using namespace cv;

#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j<img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0���� ���Ե� ������

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}
#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)


int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx)*(1.0 - dy)*A + dx*(1.0 - dy)*B
		+ (1.0 - dx)*dy*C + dx*dy*D;

	return((int)(value + 0.5));
}


typedef struct EncodingResult {									//���ڵ� ��� ����ü
	int idx; int flag = 0; EncodingResult* a[4] = {NULL,}; int gt; int avg; int err; double alpha; int x, y;
}EResult;

EncodingResult* ERAlloc(int a) {//���ڵ� ��� ����ü ���� �Ҵ��Լ�
	EncodingResult* tmp;
	tmp = (EncodingResult*)calloc(a, sizeof(EncodingResult));		//size
	return(tmp);
}


void ERFree2(EncodingResult* image, int size) {//���ڵ���� ����ü �����Ҵ������ �޸� ���� �Լ�
	for (int i = 0; i < size; i++)
	{
		if (image[i].flag == 1)
		{
			for (int j = 0; j < 4; j++)
				free(image[i].a[j]);
		}
	}

	free(image);
}

void DownSize2(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y += 2)
	{
		for (int x = 0; x < width; x += 2)
		{
			if (x + 1 > width || y + 1 > height || x / 2 < 0 || y / 2 < 0) continue;
			else img_out[y / 2][x / 2] = (img[y][x] + img[y][x + 1] + img[y + 1][x] + img[y + 1][x + 1]) / 4;
		}
	}
}

// img�� t_img�� ����, y,x�� img�� ������ġ
int GetError(int** img, int** t_img, int height, int width, int t_h, int t_w, int y, int x)
{
	// err : ���ȼ����� ���밪�� ���Ѱ�.
	int temp = 0;
	for (int a = 0; a < t_h; a++)
	{
		for (int b = 0; b < t_w; b++)
		{
			if (y + a >= height || x + b >= width) continue;
			else temp += abs(t_img[a][b] - img[y + a][x + b]);
		}
	}
	return temp;
}


//Geometric Transform  a: des, b:src
//1 a[y][x] = b[y][x]
//2 a[y][x] = b[N-1-y][x]
//3 a[y][x] = b[y][N-1-x]
//4 a[y][x] = b[N-1-y][N-1-x]
//5 a[y][x] = b[x][y]
//6 a[y][x] = b[N-1-x][y]
//7 a[y][x] = b[x][N-1-y]
//8 a[y][x] = b[N-1-x][N-1-y]


void GT1(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[y][x] = src[y][x];
		}
	}
}

void GT2(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[y][x] = src[height - 1 - y][x];
		}
	}
}

void GT3(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[y][x] = src[y][width - 1 - x];
		}
	}
}
void GT4(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[y][x] = src[height - 1 - y][width - 1 - x];
		}
	}
}
void GT5(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[x][y] = src[y][x];
		}
	}

}

void GT6(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[y][x] = src[height - 1 - x][y];
		}
	}
}

void GT7(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[y][x] = src[x][width - 1 - y];
		}
	}
}
void GT8(int** src, int** des, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			des[y][x] = src[height - 1 - x][width - 1 - y];
		}
	}
}

void selectGT(int** in_img, int height, int width, int num, int** out_img)
{
	switch (num)
	{
	case 1:
		GT1(in_img, out_img, height, width); break;
	case 2:
		GT2(in_img, out_img, height, width); break;
	case 3:
		GT3(in_img, out_img, height, width); break;
	case 4:
		GT4(in_img, out_img, height, width); break;
	case 5:
		GT5(in_img, out_img, height, width); break;
	case 6:
		GT6(in_img, out_img, height, width); break;
	case 7:
		GT7(in_img, out_img, height, width); break;
	case 8:
		GT8(in_img, out_img, height, width); break;
	default:
		printf("Isom default", num); break;
	}
}

//����� ����� ���ϴ� �Լ�
int GetBlockAvg(int** img, int height, int width, int y, int x, int N)
{
	int sum = 0;
	//������ǥ  y(height),x(width)
	if (y < 0 || y + N > height || x < 0 || x + N > width) return 0;
	else
	{
		for (int i = y; i < y + N; i++) 
		{
			for (int j = x; j <x + N; j++) 
			{
				sum += img[i][j];
			}
		}
		return (int)(sum / (N*N));
	}
}


//���������� ���°�.
void RemoveMean(int** Block, int height, int width, int avg)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Block[y][x] -= avg;
		}
	}

}
//����� �̹����� �о���°�
void ReadBloack(int** img, int y, int x, int dy, int dx, int** block)
{
	for (int i = y; i < y + dy; i++)
	{
		for (int j = x; j < x + dx; j++)
		{
			block[i - y][j - x] = img[i][j];
		}
	}

}
//�̹����� ����� ���°�.
void WriteBloack(int** img, int y, int x, int dy, int dx, int** block)
{
	for (int i = y; i < y + dy; i++)
	{
		for (int j = x; j < x + dx; j++)
		{
			img[i][j] = block[i - y][j - x];
		}
	}
}


// ����� ���� ���ϴ� �Լ�
void AlphaMultiply(int** img, int** temp, int N, double alpha)
{
	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++)
		{
			temp[y][x] = (int)(alpha*img[y][x] + 0.5); //+0.5�� ����� �ս� ���δ�.
		}
	}
}


void Copy_img(int** img1, int** img2, int height, int width)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img1[y][x] = img2[y][x];
		}
	}
}
double PSNR(int** im1, int** im2, int height, int width)
{
	double err = 0.0;
	for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
		err += ((double)im1[i][j] - im2[i][j]) * (im1[i][j] - im2[i][j]);
	}

	err = err / (width*height);

	return(10.0 * log10(255 * 255.0 / err));
}

bool cmp(const EResult &a, const EResult &b)  //��������
{
	if (a.err > b.err) return true;
	else if (a.err == b.err) return a.err > b.err;
	else return false;
}
bool cmp2(const EResult &a, const EResult &b) // ��������
{
	if (a.idx < b.idx) return true;
	else if (a.idx == b.idx) return a.err < b.err;
	else return false;
}

bool WriteParameter(const char* name, EncodingResult* A, int size) {//txt�� ����ü ������ ���Ϸ� ����
	FILE* fp = fopen(name, "w");
	if (fp == NULL) {
		printf("\n Failure in fopen!!"); return false;
	}
	int n = 0;
	for (int j = 0; j < size; j++)
	{
		fprintf(fp, "%d %d %d %d %d %f %d\n", A[j].idx, A[j].x, A[j].y, A[j].gt, A[j].avg, A[j].alpha, A[j].flag);
		if (A[j].flag == 1)
		{
			for (int i = 0; i < 4; i++)
				fprintf(fp, "%d %d %d %d %f\n", A[j].a[i]->x, A[j].a[i]->y, A[j].a[i]->gt, A[j].a[i]->avg, A[j].a[i]->alpha);
		}
	}
	fclose(fp);
	return true;
}

bool ReadParameter(const char* name, EncodingResult* A, int size) {
	FILE* fp = fopen(name, "r");

	if (fp == NULL) {
		printf("\n Failure in fopen!!"); return false;
	}
	for (int i = 0; i < size; i++)
	{
		fscanf(fp, "%d%d%d%d%d%lf%d", &(A[i].idx), &(A[i].x), &(A[i].y), &(A[i].gt), &(A[i].avg), &(A[i].alpha), &(A[i].flag));
		if (A[i].flag == 1)
		{
			for (int j = 0; j < 4; j++)
			{
				A[i].a[j] = (EResult*)calloc(1, sizeof(EResult));
				fscanf(fp, "%d%d%d%d%lf", &(A[i].a[j]->x), &(A[i].a[j]->y), &(A[i].a[j]->gt), &(A[i].a[j]->avg), &(A[i].a[j]->alpha)); //���⼭���� �ٽ�
			}
		}
	}
	fclose(fp);
	return true;
}

// Dblock �̹��� ��ü���� ���� ���� ��� ��ȯ
EResult TemplateMatching(int** img, int height, int width, int** block, int N)
{
	EResult result;

	int** Dblock = IntAlloc2(2 * N, 2 * N);
	int** Dblock2 = IntAlloc2(N, N);								//�ٿ����¡
	int** Dblock3 = IntAlloc2(N, N);								//GT�� Dblock
	int** Dblock4 = IntAlloc2(N, N);								//alpha������ Dblock

	int err = INT_MAX;

	result.avg = GetBlockAvg(block, N, N, 0, 0, N);
	RemoveMean(block, N, N, result.avg);

	for (int y = 0; y < height - 2 * N; y++)
	{
		for (int x = 0; x < width - 2 * N; x++)
		{
			ReadBloack(img, y, x, 2 * N, 2 * N, Dblock);
			DownSize2(Dblock, 2 * N, 2 * N, Dblock2);
			int tempAvg = GetBlockAvg(Dblock2, N, N, 0, 0, N);
			for (int num = 1; num <= 8; num++)						//GT ����  
			{
				selectGT(Dblock2, N, N, num, Dblock3);
				RemoveMean(Dblock3, N, N, tempAvg);
				for (double alpha = 0.3; alpha <= 1.0; alpha += 0.1) //alpha����
				{
					AlphaMultiply(Dblock3, Dblock4, N, alpha); 
					int curErr = GetError(block, Dblock4, N, N, N, N, 0, 0);
					if (err > curErr)
					{
						err = curErr; 
						result.x = x;
						result.y = y;
						result.alpha = alpha;
						result.gt = num;
						result.err = err;
					}
				}
			} 
		}
	}
	IntFree2(Dblock, 2 * N, 2 * N);
	IntFree2(Dblock2, N, N);
	IntFree2(Dblock3, N, N);
	IntFree2(Dblock4, N, N);

	return result;
}


void encoding(int** img, int height, int width, int N, int size) 
{
	int** block = IntAlloc2(N, N);
	int errN = size * 0.2; // ���� 20% ���� ����
	int i = 0;
	clock_t start, finish;


	EResult* en_result = ERAlloc(size);

	for (int y = 0; y < height / N; y++)
	{
		for (int x = 0; x < width / N; x++)
		{
			ReadBloack(img, N * y, N * x, N, N, block); // ������� 
			en_result[i] = TemplateMatching(img, height, width, block, N);
			en_result[i].idx = i;
			i++;
		}
	}
	sort(en_result, en_result + size, cmp);

	//���� 20% �����κ�
	for (int i = 0; i < errN; i++)
	{
		en_result[i].flag = 1;

		int** Eblock = IntAlloc2(N, N);
		int** Eblock2 = IntAlloc2(N / 2, N / 2);

		int idx = en_result[i].idx;
		ReadBloack(img, (idx / (height/N)) * N, (idx % (width/N)) * N, N, N, Eblock);	 // ���������� �ش��ϴ� 8*8 ��� �̹��� ������ 
		for (int j = 0; j < 4; j++)
		{
			ReadBloack(Eblock,  ((int)(j/2) * 4) , (j % 2) * 4, N/2, N/2, Eblock2);		// 4*4 ��� �̹��� ������
			en_result[i].a[j] = (EResult*)calloc(1, sizeof(EResult));
			*en_result[i].a[j] = TemplateMatching(img, height, width, Eblock2, N/2);	// ���ڵ� �ٽ��� 
		}
		IntFree2(Eblock, N, N);
		IntFree2(Eblock2, N / 2, N / 2);
	}
	sort(en_result, en_result + size, cmp2);
	WriteParameter("encoding.txt", en_result, size);									//����ü�� ����� ������ txt�������� ����.

	ERFree2(en_result, size);
	IntFree2(block, N, N);
}

// ���ڵ� �Լ�
void Decoding(EncodingResult* en_Result, int** dec, int height, int width, int N, int size) 
{
	int** block = IntAlloc2(N * 2, N * 2);
	int** block2 = IntAlloc2(N, N);
	int** block3 = IntAlloc2(N, N);

	int** mblock = IntAlloc2(N, N);
	int** mblock2 = IntAlloc2(N / 2, N / 2);
	int** mblock3 = IntAlloc2(N / 2, N / 2);

	int** dec_tmp = IntAlloc2(height, width);

	for (int i = 0; i< size; i++)
	{

		int y = (i / (height / N)) * N;  //�̹��� ����ġ 
		int x = (i % (width / N)) * N;

		if (en_Result[i].flag == 0)
		{
			ReadBloack(dec, en_Result[i].y, en_Result[i].x, N * 2, N * 2, block);			//x,y��ǥ�� ���ũ���� 2�踸ŭ �о��
			DownSize2(block, N * 2, N * 2, block2);											//�ٿ����¡
			int avg = GetBlockAvg(block2, N, N, 0, 0, N);									//��հ� ���
			RemoveMean(block2, N, N, avg);													//��հ� ����
			selectGT(block2, N, N, en_Result[i].gt, block3);					            //����� gt����
			AlphaMultiply(block3, block3, N, en_Result[i].alpha);							//alpha ������.
			RemoveMean(block3, N, N, -en_Result[i].avg);									//����� ��հ��� ������
			WriteBloack(dec_tmp, y, x, N, N, block3);										//img_dec_tmp�� x,y�� ��ǥ�� ���ũ�⸸ŭ ó�����̹����� ����.
		}
		else
		{
			for (int j = 0; j < 4; j++)
			{
				ReadBloack(dec, en_Result[i].a[j]->y, en_Result[i].a[j]->x, N, N, mblock);		
				DownSize2(mblock, N, N, mblock2);												
				int avg = GetBlockAvg(mblock2, N / 2, N / 2, 0, 0, N / 2);									
				RemoveMean(mblock2, N / 2, N / 2, avg);													
				selectGT(mblock2, N / 2, N / 2, en_Result[i].a[j]->gt, mblock3);					            
				AlphaMultiply(mblock3, mblock3, N / 2, en_Result[i].a[j]->alpha);								
				RemoveMean(mblock3, N / 2, N / 2, -en_Result[i].a[j]->avg);									
				WriteBloack(dec_tmp, y + ((int)(j / 2) * 4), x + ((j % 2) * 4), N / 2, N / 2, mblock3);					
			}
		}
	}

	Copy_img(dec, dec_tmp, height, width);				//image_dec_tmp�� �����ϱ����� dec�� ����

	IntFree2(block, N * 2, N * 2);					   //�޸� ����
	IntFree2(block2, N, N);
	IntFree2(block3, N, N);
	IntFree2(mblock, N, N);			
	IntFree2(mblock2, N / 2, N / 2);
	IntFree2(mblock3, N / 2, N / 2);
	IntFree2(dec_tmp, height, width);
}



void main()
{
	int height, width, N = 8;
	int** img = ReadImage("LENA256.bmp", &height, &width);
	int size = (width / N) *  (height / N);
	int** image_dec = IntAlloc2(height, width);								//���ڵ� ��ȯ�Ҷ����� ��� img
	EncodingResult* en_result = ERAlloc(size);								//���ڵ��� ������ ���������� ����ü ���� �Ҵ�
	ReadParameter("encoding.txt", en_result, size);							//���ڵ� �� ������ �������ִ� txt���� �о��

	for (int i = 0; i< height; i++)
		for (int j = 0; j < width; j++)
			image_dec[i][j] = 128;											//image_dec 128�� �ʱ�ȭ


	clock_t start, finish;
	start = clock();
	encoding(img, height, width, N, size);

	for (int i = 0; i < 10; i++) {
		printf("--- %d��° ���ڵ� ���� ---\n", i);
		printf("PSNR = %f\n", PSNR(img, image_dec, height, width));
		Decoding(en_result, image_dec, height, width, N, size);				//���ڵ� ����
	}

	finish = clock();
	printf("%f�� \n", (double)(finish - start) / CLOCKS_PER_SEC);
	
	ImageShow("���ڵ�", image_dec, width, height);							//���ڵ� ���
	ImageShow("����", img, width, height);									//�̹��� ���� ���

	system("pause");
	IntFree2(img, width, height);
	IntFree2(image_dec, height, width);
	ERFree2(en_result, size);

}
