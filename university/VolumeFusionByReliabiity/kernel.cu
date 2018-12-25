
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

int main(int argc, char** argv){
	int i, j, k, l;
	FILE *volumef1, *reliabilityf1, *volumef2, *reliabilityf2, *para;

	//ボリューム、信頼度、回転行列とかそのほかパラメータ
	
	/*
	設定ファイルパラメータ

	1ボリュームファイルのパス
	1信頼度ファイルのパス
	2ボリュームファイルのパす
	2信頼度ファイルのパス
	出力ファイル名
	ボリュームのボクセル数XYZ、

	*/

	char in_path_v1[1000], in_path_r1[1000], in_path_v2[1000], in_path_r2[1000], out_name[1000];
	
	//入力ファイルと出力ファイル名
	int voxels[3];
	//double vsize;

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	fscanf(para, "%s", in_path_v1);
	fscanf(para, "%s", in_path_r1);
	fscanf(para, "%s", in_path_v2);
	fscanf(para, "%s", in_path_r2);
	fscanf(para, "%s", out_name);
	fscanf(para, "%d %d %d", &voxels[0], &voxels[1], &voxels[2]);
	fclose(para);

	//ボリュームの格納容器　xy平面サイズvoxels1[0] * voxels1[1]　x一列サイズvoxel1[0]
	float *volume_1 = new float[voxels[0] * voxels[1] * voxels[2]];

	float *reliability_1 = new float[voxels[0] * voxels[1] * voxels[2]];
	
	float *volume_2 = new float[voxels[0] * voxels[1] * voxels[2]];

	float *reliability_2 = new float[voxels[0] * voxels[1] * voxels[2]];

	float *fusioned = new float[voxels[0] * voxels[1] * voxels[2]];

	float *fusioned_reliability = new float[voxels[0] * voxels[1] * voxels[2]];

	//ボリューム読み込む
	if ((volumef1 = fopen(in_path_v1, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}
	printf("volume 1 load success\n");
	fread(volume_1, sizeof(float), voxels[0] * voxels[1] * voxels[2], volumef1);
	fclose(volumef1);


	if ((volumef2 = fopen(in_path_v2, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}
	printf("volume 2 load success\n");
	fread(volume_2, sizeof(float), voxels[0] * voxels[1] * voxels[2], volumef2);//ボリューム読み込み
	fclose(volumef2);


	if ((reliabilityf1 = fopen(in_path_r1, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}
	printf("reliability 1 load success\n");
	fread(reliability_1, sizeof(float), voxels[0] * voxels[1] * voxels[2], reliabilityf1);//ボリューム読み込み
	fclose(reliabilityf1);


	if ((reliabilityf2 = fopen(in_path_r2, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}
	printf("reliability 2 load success\n");
	fread(reliability_2, sizeof(float), voxels[0] * voxels[1] * voxels[2], reliabilityf2);//ボリューム読み込み
	fclose(reliabilityf2);


	printf("fusion start\n");

	float r1, r2, tr;
	float v1, v2, vf, vr;
	for (i = 0; i < voxels[2]; i++) {
		printf("progress %d / %d\n", i, voxels[2]);
		for (j = 0; j < voxels[1]; j++) {
			for (k = 0; k < voxels[0]; k++) {

				r1 = reliability_1[voxels[0] * voxels[1] * i + voxels[0] * j + k];
				r2 = reliability_2[voxels[0] * voxels[1] * i + voxels[0] * j + k];

				vr = (r1*r2) / (r1 + r2);

				fusioned_reliability[voxels[0] * voxels[1] * i + voxels[0] * j + k] = vr;

				r1 = 1 / r1;
				r2 = 1 / r2;
				tr = r1 + r2;
				r1 = r1 / tr;
				r2 = r2 / tr;

				v1 = volume_1[voxels[0] * voxels[1] * i + voxels[0] * j + k];
				v2 = volume_2[voxels[0] * voxels[1] * i + voxels[0] * j + k];

				vf = v1*r1 + v2*r2;

				fusioned[voxels[0] * voxels[1] * i + voxels[0] * j + k] = vf;

			}
		}
	}
	char name[1000];
	FILE *out;

	printf("Writing fusioned volume\n");

	sprintf(name, "%s-float-(%dx%dx%d).raw", out_name, voxels[0], voxels[1], voxels[2]);
	printf("%s", name);


	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(fusioned, sizeof(float), voxels[0] * voxels[1] * voxels[2], out);

	printf("\nWriting fusioned volume end\n");

	fclose(out);

	printf("Writing fusioned reliability\n");

	sprintf(name, "%s-reliability-float-(%dx%dx%d).raw", out_name, voxels[0], voxels[1], voxels[2]);
	printf("%s", name);


	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(fusioned_reliability, sizeof(float), voxels[0] * voxels[1] * voxels[2], out);

	printf("\nWriting fusioned reliability end\n");

	fclose(out);

	/*
	printf("bit adusted ver writing\n");

	for (i = 0; i < voxels[2]; i++) {
		printf("progress %d / %d\n", i, voxels[2]);
		for (j = 0; j < voxels[1]; j++) {
			for (k = 0; k < voxels[0]; k++) {


				fusioned[voxels[0] * voxels[1] * i + voxels[0] * j + k] = 1000*fusioned[voxels[0] * voxels[1] * i + voxels[0] * j + k];
			}
		}
	}
	sprintf(name, "%s-float-adjusted-(%dx%dx%d).raw", out_name, voxels[0], voxels[1], voxels[2]);
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(fusioned, sizeof(float), voxels[0] * voxels[1] * voxels[2], out);

	printf("\nWriting fusioned volume end\n");

	fclose(out);
*/

	delete[] volume_1;
	delete[] volume_1;
	delete[] reliability_1;
	delete[] reliability_2;
	delete[] fusioned;
	delete[] fusioned_reliability;
    return 0;
}
