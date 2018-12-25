
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<string>
#include<fstream>
#include<math.h>


int main(int argc, char** argv) {
	int i, j, k, l;
	FILE *volumef1, *reliabilityf1, *volumef2, *reliabilityf2, *para;

	//ボリューム、信頼度、回転行列とかそのほかパラメータ

	/*
	設定ファイルパラメータ

	1ファイル1のパス
	1ファイル2のパス
	出力ファイル名
	ボリュームのボクセル数XYZ、

	*/

	char in_path_1[1000], in_path_2[1000], out_name[1000];

	//入力ファイルと出力ファイル名
	int voxels[3];
	//double vsize;

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	fscanf(para, "%s", in_path_1);
	fscanf(para, "%s", in_path_2);
	fscanf(para, "%s", out_name);
	fscanf(para, "%d %d %d", &voxels[0], &voxels[1], &voxels[2]);
	fclose(para);

	//ボリュームの格納容器　xy平面サイズvoxels1[0] * voxels1[1]　x一列サイズvoxel1[0]
	float *volume_1 = new float[voxels[0] * voxels[1] * voxels[2]];

	float *volume_2 = new float[voxels[0] * voxels[1] * voxels[2]];

	float *fusioned = new float[voxels[0] * voxels[1] * voxels[2]];

	//ボリューム読み込む
	if ((volumef1 = fopen(in_path_1, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}
	printf("volume 1 load success\n");
	fread(volume_1, sizeof(float), voxels[0] * voxels[1] * voxels[2], volumef1);
	fclose(volumef1);


	if ((volumef2 = fopen(in_path_2, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}
	printf("volume 2 load success\n");
	fread(volume_2, sizeof(float), voxels[0] * voxels[1] * voxels[2], volumef2);//ボリューム読み込み
	fclose(volumef2);

	printf("fusion start\n");

	float r1, r2;

	for (i = 0; i < voxels[2]; i++) {
		printf("progress %d / %d\n", i, voxels[2]);
		for (j = 0; j < voxels[1]; j++) {
			for (k = 0; k < voxels[0]; k++) {
				/*
				ここで信頼度合成計算
				*/
				r1 = volume_1[voxels[0] * voxels[1] * i + voxels[0] * j + k];
				r2 = volume_2[voxels[0] * voxels[1] * i + voxels[0] * j + k] + 0.0005;

				fusioned[voxels[0] * voxels[1] * i + voxels[0] * j + k] = r1*r2;
			}
		}
	}

	printf("Writing fusioned volume\n");
	char name[1000];
	sprintf(name, "%s-float-(%dx%dx%d).raw", out_name, voxels[0], voxels[1], voxels[2]);
	printf("%s", name);

	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(fusioned, sizeof(float), voxels[0] * voxels[1] * voxels[2], out);

	printf("\nWriting fusioned volume end\n");

	fclose(out);


	delete[] volume_1;
	delete[] volume_1;
	delete[] fusioned;
	return 0;
}