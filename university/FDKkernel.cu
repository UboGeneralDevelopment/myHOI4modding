
#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>
#include <time.h>


#define PI 3.1415926535897932384626433832795f

#define threadN 512 //現在の CUDA では 512 が MAX
#define sinoBlock 200 //GPUメモリが足りなければ減らす。shared メモリの関係で threadN 以下の値を設定

//shepp-logan の CUDA Kernel
__global__ void convolution_GPU(int W, int H, float *S, float d, float D, //float *q){
	float *shepp, float *q) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id < W*H) {
		int i = id / W; //よこ
		int j = id - i*W; //たて
		float s = 0;
		for (int k = 0; k<W; k++)
			if (j > k)
				s += q[i*W + k] * shepp[j - k];
			else
				s += q[i*W + k] * shepp[k - j];
		S[id] = s;
	}
}

//パネルの横方向に畳み込み積分
void convolution(int W, int H, int P, float **S, float d, float D, float shiftU, float shiftV) {
	//(W, H): ディテクタのピクセル数．P:投影像数．S:投影像．
	//d:回転軸に射影したピクセルサイズ．D:線減から回転軸までの距離

	int i, j, k;

	float D2 = D*D;
	float Hh = 0.5f*(1 - H); //ピクセル中心へにシフト量
	float Wh = 0.5f*(1 - W); //ピクセル中心へにシフト量

	float *cosG = (float*)malloc(H*W * sizeof(float)); //テーブル
	float *shepp = (float*)malloc(W * sizeof(float)); //テーブル
	float *q = (float*)malloc(H*W * sizeof(float)); //一時計算領域

	float* d_shepp; //GPU用
	float *d_q; //GPU用
	float *d_S; //GPU用
	cudaMalloc((void**)&d_shepp, sizeof(float)*W);
	cudaMalloc((void**)&d_q, sizeof(float)*H*W);
	cudaMalloc((void**)&d_S, sizeof(float)*H*W);

	int counter = 0;
	int percent = 0;

	for (i = 0; i<H; i++) {
		float v = d*(i + Hh - shiftV);
		float L = D2 + v*v;

		for (j = 0; j<W; j++) {
			float s = d*(j + Wh - shiftU);
			cosG[i*W + j] = (float)(D / sqrt(L + s*s));
		}
	}

	for (i = 0; i<W; i++)
		shepp[i] = 2.0f / (PI*PI*d*(1.0f - 4 * i*i)); //d は一個約分でキャンセル
	cudaMemcpy(d_shepp, shepp, sizeof(float)*W, cudaMemcpyHostToDevice);

	int blockN = (H*W + threadN - 1) / threadN; //ブロック数

												//projection image を 一枚づつ送って畳み込む
	for (i = 0; i<P; i++) {
		for (j = 0; j<H; j++)
			for (k = 0; k<W; k++)
				q[j*W + k] = S[i][j*W + k] * cosG[j*W + k];

		cudaMemcpy(d_q, q, sizeof(float)*H*W, cudaMemcpyHostToDevice);

		convolution_GPU << < blockN, threadN >> >(W, H, d_S, d, D, d_shepp, d_q);

		cudaMemcpy(S[i], d_S, sizeof(float)*H*W, cudaMemcpyDeviceToHost);

		if (++counter * 100 > percent*P) {
			printf("\r %3d %%", ++percent);
			fflush(stdout);
		}
	}

	free(cosG);
	free(shepp);
	free(q);

	cudaFree(d_shepp);
	cudaFree(d_q);
	cudaFree(d_S);

	printf("\n");
}

//backprojection の CUDA Kernel
__global__ void backprojection_GPU(int W, int H, int P, float *S,
	float dI, float D, float dW, float dH, float s2, float C,
	float *rotS, float *rotC,
	int XY, float *p, float *F,
	float shiftA, float shiftU, float shiftV) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id < XY*XY) {
		float x1 = p[id * 3];
		float y1 = p[id * 3 + 1];
		if (x1*x1 + y1*y1 < s2) {
			float z = p[id * 3 + 2];

			float f = 0;
			float rot = shiftA*dW * 2 / W;
			for (int k = 0; k<P; k++) { //一周足す
				float x = (x1 - shiftA)*rotC[k] - y1*rotS[k] + shiftA;
				float y = (x1 - shiftA)*rotS[k] + y1*rotC[k];

				float U = D / (D + y);

				float s = x*U + dW;
				float ds = s*dI - 0.5f + shiftU;
				int n = (int)ds;  //detector ID
				if (n >= 0 && n <= W - 2) {
					float v = z*U + dH;
					float dv = v*dI - 0.5 + shiftV;
					int m = (int)dv;  //detector ID
					if (m >= 0 && m <= H - 2) {
						ds -= n; //for linear interpolation on detector along X
						dv -= m; //for linear interpolation on detector along Z

						int idx = (k*H + m)*W + n;
						f += U*U*((1.0f - dv)*((1.0f - ds)*S[idx] + ds*S[idx + 1]) +
							dv*((1.0f - ds)*S[idx + W] + ds*S[idx + W + 1]));
					}
				}
			}
			F[id] += C*f;
		}
	}
}


//逆投影 各ボクセルからパネルに当たりに行く
void backprojection(int XY, int Z, float **F, float sXY, float sZ,
	int W, int H, int P, float **S,
	float d, float D,
	float shiftA, float shiftU, float shiftV) {
	//(XY, XY, Z): CTボリュームのボクセル数．(sXY, sXY, sZ): ボリュームのサイズの半分

	int i, j, k, l;

	float s2 = sXY*sXY;
	float dW = 0.5f*d*W;
	float dH = 0.5f*d*H;

	float pitch = 2.0f*sXY / XY; //ボクセルのサイズ
	float dI = 1.0f / d; //割り算の回数を減らすため
	float C = 2 * PI / P; //係数．積分の微小量をまとめたもの．

						  //計算時間の節約に sin, cos のテーブルを作る
	float *rotS = (float*)malloc(P * sizeof(float));
	float *rotC = (float*)malloc(P * sizeof(float));

	int counter = 0;
	int percent = 0;

	float *p = (float*)malloc(XY*XY * 3 * sizeof(float)); //評価点の座標の配列

	float* d_S;
	float* d_p;
	float* d_F;
	float* d_rotS;
	float* d_rotC;

	cudaMalloc((void**)&d_S, sizeof(float)*sinoBlock*H*W);
	cudaMalloc((void**)&d_p, sizeof(float) * 3 * XY*XY);
	cudaMalloc((void**)&d_F, sizeof(float)*XY*XY);
	cudaMalloc((void**)&d_rotS, sizeof(float)*P);
	cudaMalloc((void**)&d_rotC, sizeof(float)*P);

	for (i = 0; i<P; i++) {
		float a = PI*(2.0f*i / P + 0.5f);
		rotS[i] = (float)sin(a);
		rotC[i] = (float)cos(a);
	}
	cudaMemcpy(d_rotS, rotS, sizeof(float)*P, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rotC, rotC, sizeof(float)*P, cudaMemcpyHostToDevice);
	free(rotS);
	free(rotC);

	for (i = 0; i<Z; i++)
		for (j = 0; j<XY*XY; j++)
			F[i][j] = 0;

	int blockN = (XY*XY + threadN - 1) / threadN;

	for (i = 0; i<P; i += sinoBlock) { //sinogram は 全部 GPU memory に上がらないので sinoBlock 枚ずつ足しこむ
		int sB = sinoBlock;
		if (sinoBlock > P - i)
			sB = P - i;
		for (j = 0; j<sB; j++)
			cudaMemcpy(&(d_S[j*H*W]), S[i + j], sizeof(float)*H*W, cudaMemcpyHostToDevice);

		for (j = 0; j<Z; j++) { //各 z で、一スライス毎に GPU kernel を呼ぶ
			float z = j*pitch - sZ; //z coordinate on image
			for (k = 0; k<XY; k++) {
				float y = k*pitch - sXY; //y coordinate on image
				for (l = 0; l<XY; l++) {
					float x = l*pitch - sXY; //x coordinate on image

					int id = 3 * (k*XY + l);

					p[id] = x;
					p[id + 1] = y;
					p[id + 2] = z;
				}
			}

			cudaMemcpy(d_p, p, sizeof(float) * 3 * XY*XY, cudaMemcpyHostToDevice);
			cudaMemcpy(d_F, F[j], sizeof(float)*XY*XY, cudaMemcpyHostToDevice);

			backprojection_GPU << < blockN, threadN >> >(W, H, sB, d_S,
				dI, D, dW, dH, s2, C,
				&(d_rotS[i]), &(d_rotC[i]),
				XY, d_p, d_F,
				shiftA, shiftU, shiftV);

			cudaMemcpy(F[j], d_F, sizeof(float)*XY*XY, cudaMemcpyDeviceToHost);

			counter += sB;
			if (counter * 100 > percent*Z*P) {
				percent = (int)(counter * 100 / (Z*P));
				printf("\r %3d %%", percent);
				fflush(stdout);
			}
		}
	}

	free(p);
	cudaFree(d_S);
	cudaFree(d_p);
	cudaFree(d_F);
	cudaFree(d_rotS);
	cudaFree(d_rotC);

	printf("\n");
}

int main(int argc, char** argv) {
	int i, j, k;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("#GPU = %d\n", deviceCount);
	int device;
	int useGPU = 0;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d (%s) has compute capability %d.%d.\nsharedMem = %dKB\nglobalMem = %dMB\n",
			device, deviceProp.name,
			deviceProp.major, deviceProp.minor,
			deviceProp.sharedMemPerBlock / 1024,
			deviceProp.totalGlobalMem / 1024 / 1024);
	}
	printf("Using GPU %d\n", useGPU);
	cudaSetDevice(useGPU);

	char in_path[1000], out_name[1000];
	float s2o, s2d;
	int W, H, P;
	float detP;

	int N;
	float scaleW, scaleH;
	float shiftA, shiftU, shiftV;

	float dW, dH;
	float sizeR, sizeH;
	int zN;
	float **ct, **sino;
	FILE *in, *out, *para;//, *out1;
	float delta;

	char name[1000];

	unsigned short *buff;

	time_t timeS, timeE;

	time(&timeS);

	//パラメータをファイルから読込
	para = fopen(argv[1], "r");
	fscanf(para, "%s", in_path);
	fscanf(para, "%s", out_name);
	fscanf(para, "%f %f", &s2d, &s2o);
	fscanf(para, "%d %d %d %f", &W, &H, &P, &detP);
	fscanf(para, "%d %f %f", &N, &scaleW, &scaleH);
	fscanf(para, "%f %f %f", &shiftA, &shiftU, &shiftV);
	fclose(para);

	//detector size
	dW = detP*W;
	dH = detP*H;

	//image size (cylinder)
	sizeR = (float)(0.5f*s2o*dW / sqrt(s2d*s2d + 0.25f*dW*dW)); //radius
	sizeH = 0.5f*(s2o/*-sizeR*/) / s2d*dH; //height/2
	sizeR *= scaleW;
	sizeH *= scaleH;


	//#volxels in z-axis
	zN = (int)(sizeH*N / sizeR);

	printf("Volume Size = %f x %f x %f\n", 2 * sizeR, 2 * sizeR, 2 * sizeH);
	printf("Origin = (%f, %f %f)\n", -sizeR, -sizeR, -sizeH);
	printf("#voxels = %d x %d x %d\n", N, N, zN);
	printf("Voxel size = %f\n", 2 * sizeR / N);

	//tomogram
	ct = (float**)malloc(zN * sizeof(float*));
	for (i = 0; i<zN; i++)
		ct[i] = (float*)malloc(N*N * sizeof(float));

	//sinogram
	sino = (float**)malloc(P * sizeof(float*));
	for (i = 0; i<P; i++)
		sino[i] = (float*)malloc(W*H * sizeof(float));

	in = fopen(in_path, "rb");
	buff = new unsigned short[W];
	for (i = 0; i<P; i++) {
		for (j = 0; j<H; j++) {
			fread(buff, 2, W, in);
			for (k = 0; k<W; k++)
				sino[i][W*(H - j - 1) + k] = (float)(-log((float)buff[k] / 65535));
		}
	}
	delete[] buff;
	fclose(in);

	delta = (float)(detP*s2o / s2d); //scaled detector pitch

	printf("Convolution\n");
	convolution(W, H, P, sino, delta, s2o, shiftU, -shiftV);

	printf("Backprojection\n");
	backprojection(N, zN, ct, sizeR, sizeH,
		W, H, P, sino, delta, s2o, (shiftA - shiftU)*detP, shiftU, -shiftV);
	for (i = 0; i<P; i++)
		free(sino[i]);
	free(sino);

	printf("Writing\n");

	sprintf(name, "%s-float-O(%.6g,%.6g,%.6g)-%dx%dx%d-%.6gx%.6gx%.6g.raw",
		out_name, -sizeR, -sizeR, -sizeH, N, N, zN, 2 * sizeR / N, 2 * sizeR / N, 2 * sizeR / N);
	out = fopen(name, "wb");
	for (i = 0; i<zN; i++)
		fwrite(ct[i], 4, N*N, out);
	fclose(out);

	for (i = 0; i<zN; i++)
		free(ct[i]);
	free(ct);

	time(&timeE);
	printf("Timing = %f sec.\n", difftime(timeE, timeS));

	return 0;
}
