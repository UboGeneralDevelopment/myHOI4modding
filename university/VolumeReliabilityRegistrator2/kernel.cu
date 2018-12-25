
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
方針

位置合わせそのものと、位置合わせ変換行列はVGSTUDIOで出力する。

変換を行う第2ボリュームと、変換行列、そのほか変換に必要なデータを読み込み

第1ボリュームと同サイズの出力ボリュームを作り、その各ボクセルが第2ボリューム中で対応する位置を変換行列に従って計算し、第2ボリュームの値をとってきて代入していく。
第2ボリューム上に対応する値がなければとりあえず放置。

信頼度も同様に変換。

返還後のボリュームと信頼度ボリュームを出力

*/

int main(int argc, char** argv){
	int i, j, k, l;
	FILE *in_volume, *in_reliability, *para;
	//ボリューム、信頼度、回転行列とかそのほかパラメータ
	/*
	設定ファイルパラメータ
	
	ボリュームファイルのパス
	信頼度ファイルのパス
	出力ファイル名
	入力ボリュームのボクセル数XYZ、出力ボリュームのボクセル数XYZ
	入力ボリュームのボクセルサイズ、出力ボリュームのボクセルサイズ
	変換行列。ボリューム１をボリューム2以降に位置合わせしていく。

	必要な変換行列は、位置合わせ先のボリュームから、位置合わせするボリュームへの変換で、位置合わせするボリュームから位置合わせ先への変換ではないので注意。
	基本的に座標は実サイズ座標系で計算していく。第一ボリュームのこれこれ位置が第二ボリュームのここここ位置に対応するという感じ。
	第1ボリュームに第2ボリュームを位置合わせしていく。
	出力は第一ボリューム。

	*/

	char in_path_name_v[1000], in_path_name_r[1000], out_name[1000];
	//入力ファイルと出力ファイル名
	int v_in[3], v_out[3];
	//v_inが第2(入力)ボリュームで、v_outが出力
	double v_in_size, v_out_size;
	//入力ボクセル数と入力ボクセルサイズ
	double affin[4][4];
	//入力変換行列。逆変換もVGSTUDIOから容易に求まる。

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	fscanf(para, "%s", in_path_name_v);
	fscanf(para, "%s", in_path_name_r);
	fscanf(para, "%s", out_name);
	fscanf(para, "%d %d %d %d %d %d", &v_in[0], &v_in[1], &v_in[2], &v_out[0], &v_out[1], &v_out[2]);
	fscanf(para, "%lf %lf", &v_in_size, &v_out_size);

	fscanf(para, "%lf %lf %lf %lf", &affin[0][0], &affin[1][0], &affin[2][0], &affin[3][0]);
	fscanf(para, "%lf %lf %lf %lf", &affin[0][1], &affin[1][1], &affin[2][1], &affin[3][1]);
	fscanf(para, "%lf %lf %lf %lf", &affin[0][2], &affin[1][2], &affin[2][2], &affin[3][2]);
	fscanf(para, "%lf %lf %lf %lf", &affin[0][3], &affin[1][3], &affin[2][3], &affin[3][3]);
	
	fclose(para);

	i = 0;
	j = 0;
	k = 0;
	double tx, ty, tz;
	double cx, cy, cz;

	float temp_1_1, temp_1_2, temp_1_3, temp_1_4, temp_2_1, temp_2_2;
	float y0, y1;

	//ボリュームの格納容器　xy平面サイズvoxels1[0] * voxels1[1]　x一列サイズvoxel1[0]
	float *ct_volume = new float[v_in[0]* v_in[1]* v_in[2]];	
	//変換後のボリューム容器
	float *transformed_volume = new float[v_out[0] * v_out[1] * v_out[2]];

	//変換前ボリューム読み込む
	if ((in_volume = fopen(in_path_name_v, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}

	printf("volume load success\n");

	fread(ct_volume, 4, v_in[0] * v_in[1] * v_in[2], in_volume);//ボリューム読み込み

	fclose(in_volume);
	
	printf("transformation starts\n");


	for (i = 0; i < v_out[2]; i++) {
		printf("progress %d / %d\n", i, v_out[2]);
		for (j = 0; j < v_out[1]; j++) {
			for (k = 0; k < v_out[0]; k++) {
				//VGstudioでは座標中心はボリューム中心に置かれ、座標軸の変換を求めることができる。これに基づき各ボクセルの対応を計算
				//座標系には、第2ボリューム実座標系（原点ボリューム中心）、第2ボリュームボクセル座標系（原点ボリューム端点）、出力ボリューム実座標系、出力ボリュームボクセル座標系の4つがある。

				//出力ボリューム実座標系における、各ボクセル実座標
				tx = v_out_size*(k + 0.5) - (v_out_size*v_out[0] / 2);
				ty = v_out_size*(j + 0.5) - (v_out_size*v_out[1] / 2);
				tz = v_out_size*(i + 0.5) - (v_out_size*v_out[2] / 2);

				//出力ボリュームのボクセル座標を変換し、第2ボリューム実座標系での座標を計算。
				cx = affin[0][0] * tx + affin[1][0] * ty + affin[2][0] * tz + affin[3][0];
				cy = affin[0][1] * tx + affin[1][1] * ty + affin[2][1] * tz + affin[3][1];
				cz = affin[0][2] * tx + affin[1][2] * ty + affin[2][2] * tz + affin[3][2];

				//第2ボリューム実座標系の原点をボリュームの端点へ移動。
				cx = cx + (v_in_size*v_in[0] / 2);
				cy = cy + (v_in_size*v_in[1] / 2);
				cz = cz + (v_in_size*v_in[2] / 2);

				//座標系の距離をボクセル単位距離に変換、また座標原点を(0,0,0)のボクセルの中心としたときの座標を求める。
				cx = cx / v_in_size - 0.5;
				cy = cy / v_in_size - 0.5;
				cz = cz / v_in_size - 0.5;

				if (cx > 1 && cx < (v_in[0] - 2) && cy > 1 && cy < (v_in[1] - 2) && cz > 1 && cz < (v_in[2] - 2)) {
					y0 = ct_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_1 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = ct_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_2 = y0 + (cz - (int)cz)*(y1 - y0);
					
					y0 = ct_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_3 = y0 + (cz - (int)cz)*(y1 - y0);
					
					y0 = ct_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_4 = y0 + (cz - (int)cz)*(y1 - y0);


					temp_2_1 = temp_1_1 + (cy - (int)cy)*(temp_1_2 - temp_1_1);

					temp_2_2 = temp_1_3 + (cy - (int)cy)*(temp_1_4 - temp_1_3);


					transformed_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = temp_2_1 + (cx - (int)cx)*(temp_2_2 - temp_2_1);
				
				}
				else {
					transformed_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = 0;
				}

				/*
				cxv = (int)(0.5 + (cx / vsize1));
				cyv = (int)(0.5 + (cy / vsize1));
				czv = (int)(0.5 + (cz / vsize1));

				//printf("%d %d %d :\n", cxv, cyv, czv);
				
				if ( cxv > 0 && cxv < voxels1[0]  && cyv > 0 && cxv < voxels1[1] && czv > 0 && czv < voxels1[2]) {
					transformed_volume[voxels2[0] * voxels2[1] * i + voxels2[0] * j + k] = ct_volume[cxv + voxels1[0] * cyv + voxels1[0] * voxels1[1] * czv];
				}
				else {
					transformed_volume[voxels2[0] * voxels2[1] * i + voxels2[0] * j + k] = 0;
				}

				//printf("%d %d %d : %d\n", k, j, i, transformed_volume[voxels2[0] * voxels2[1] * i + voxels2[0] * j + k]);
				*/

			}
		}
	}

	printf("Writing transformed volume\n");
	char name[1000];
	sprintf(name, "%s-(%dx%dx%d).raw", out_name, v_out[0], v_out[1], v_out[2]);
	printf("%s", name);

	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(transformed_volume, sizeof(float), v_out[0] * v_out[1] * v_out[2], out);
	
	printf("\nWriting transformed volume end\n");
	
	fclose(out);

	delete[] transformed_volume;
	delete[] ct_volume;
	

	//信頼度の変換

	//ボリュームの格納容器　xy平面サイズvoxels1[0] * voxels1[1]　x一列サイズvoxel1[0]
	float *r_volume = new float[v_in[0] * v_in[1] * v_in[2]];
	//変換後のボリューム容器
	float *transformed_r_volume = new float[v_out[0] * v_out[1] * v_out[2]];

	//変換前ボリューム読み込む
	if ((in_reliability = fopen(in_path_name_r, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}

	printf("reliability load success\n");

	fread(r_volume, sizeof(float), v_in[0] * v_in[1] * v_in[2], in_reliability);//ボリューム読み込み

	fclose(in_reliability);

	printf("transformation starts\n");

	for (i = 0; i < v_out[2]; i++) {
		printf("progress %d / %d\n", i, v_out[2]);
		for (j = 0; j < v_out[1]; j++) {
			for (k = 0; k < v_out[0]; k++) {

				//出力ボリューム実座標系における、各ボクセル実座標
				tx = v_out_size*(k + 0.5) - (v_out_size*v_out[0] / 2);
				ty = v_out_size*(j + 0.5) - (v_out_size*v_out[1] / 2);
				tz = v_out_size*(i + 0.5) - (v_out_size*v_out[2] / 2);

				//出力ボリュームのボクセル座標を変換し、第2ボリューム実座標系での座標を計算。
				cx = affin[0][0] * tx + affin[1][0] * ty + affin[2][0] * tz + affin[3][0];
				cy = affin[0][1] * tx + affin[1][1] * ty + affin[2][1] * tz + affin[3][1];
				cz = affin[0][2] * tx + affin[1][2] * ty + affin[2][2] * tz + affin[3][2];

				//第2ボリューム実座標系の原点をボリュームの端点へ移動。
				cx = cx + (v_in_size*v_in[0] / 2);
				cy = cy + (v_in_size*v_in[1] / 2);
				cz = cz + (v_in_size*v_in[2] / 2);

				//座標系の距離をボクセル単位距離に変換、また座標原点を(0,0,0)のボクセルの中心としたときの座標を求める。
				cx = cx / v_in_size - 0.5;
				cy = cy / v_in_size - 0.5;
				cz = cz / v_in_size - 0.5;

				if (cx > 1 && cx < (v_in[0] - 2) && cy > 1 && cy < (v_in[1] - 2) && cz > 1 && cz < (v_in[2] - 2)) {
					y0 = r_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_1 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = r_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_2 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = r_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_3 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = r_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_4 = y0 + (cz - (int)cz)*(y1 - y0);


					temp_2_1 = temp_1_1 + (cy - (int)cy)*(temp_1_2 - temp_1_1);

					temp_2_2 = temp_1_3 + (cy - (int)cy)*(temp_1_4 - temp_1_3);


					transformed_r_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = temp_2_1 + (cx - (int)cx)*(temp_2_2 - temp_2_1);

				}
				else {
					transformed_r_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = 0;
				}

			}
		}
	}

	printf("Writing transformed reliability volume\n");
	
	sprintf(name, "%s-reliability-(%dx%dx%d).raw", out_name, v_out[0], v_out[1], v_out[2]);
	printf("%s", name);

	
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(transformed_r_volume, sizeof(float), v_out[0] * v_out[1] * v_out[2], out);

	printf("\nWriting transformed reliability end\n");

	fclose(out);

	delete[] transformed_r_volume;
	delete[] r_volume;
	system("pause");
	return 0;
}
