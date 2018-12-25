
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<string>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#define STEP 1
#define PI 3.1415926535897932384626433832795f //円周率
#define Filedivide 1　//読み込みファイルの分割。結局使わなかった。
#define ThreadsPerBlock 512 //現在の CUDA では 512 が MAX
#define angle_div 50 //評価時の分割数

/*
方針
valuemapを放り込んで、それを評価するためだけのプログラム
*/



int main(int argc, char** argv) {//argcとかには起動時に渡す変数がはいる。

	int i, j, k, l;//便利にカウントなどに使う数字

	int in_offset = 0;//オフセットがあれば
	int valuemap_phi_div, valuemap_theta_div;//valuemapの分割数。頂点除外なので、phiは本当は+1となっている

	FILE *in_valuemap_file;//入力ファイル用のファイル容器

	printf("enter value map size in form of phi theta\n");//valuemapの縦と横の長さ入力
	scanf_s("%d %d", &valuemap_phi_div, &valuemap_theta_div);

	printf("enter offset if there is\n");//オフセットがあれば入力。
	scanf_s("%d", &in_offset);

	int limit = 10;
	printf("enter limit max 10\n");
	scanf_s("%d", &limit);
	float ban = 0.1;
	printf("enter threshold default 0.1\n");
	scanf_s("%f", &ban);

	if ((in_valuemap_file = fopen(argv[1], "rb")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};
	printf("file_open_success\n");

	float *valuemap = new float[valuemap_phi_div*valuemap_theta_div];
	float *trajectory_index = new float[valuemap_phi_div*valuemap_theta_div];//評価軌道ごとに評価値を収納する。評価軌道軸の数は投影角度の数ととりあえず一致させる。

	for (i = 0; i < valuemap_phi_div*valuemap_theta_div; i++) {
		
		valuemap[i] = 0;
		trajectory_index[i] = 0;
	}
	printf("memory_success\n");

	fseek(in_valuemap_file, in_offset, SEEK_SET);//読み込み時にオフセット移動

	fread(valuemap, sizeof(float), valuemap_phi_div*valuemap_theta_div, in_valuemap_file);//ボリューム読み込み
	fclose(in_valuemap_file);
	printf("loading_success\n");
	
	/*
	for (i = 0; i < valuemap_phi_div*valuemap_theta_div; i++) {
		printf("%lf\n", valuemap[i]);
	}
	system("pause");
	*/

	printf("\n projection_trajectory_analysis_start\n");
	
	float v_phi_y = 0, v_theta_x = 0;//value_map上での角度座標
	int x, y;
	float axis_phi = 0, axis_theta = 0;//trajectoryの軸角度。
	float t = 0;//軌道の角度。開店する。
	float trajectory_value = 0;

	//各軌道ごとに評価値を計算し、indexに収納。評価を行う軌道の数は投影像の数と一致。
	for (i = 0; i < valuemap_phi_div; i++) {

		axis_phi = i*PI / valuemap_phi_div;

		for (j = 0; j < valuemap_theta_div; j++) {

			axis_theta = j*PI / valuemap_theta_div;
			trajectory_value = 0;

			for (t = 0; t < PI; t += 0.05) {//軸上で半回転
				v_phi_y = acos(-sin(axis_phi)*cos(t));
				v_theta_x = PI/2 + atan((-sin(axis_theta)*cos(axis_phi)*cos(t) + cos(axis_theta)*sin(t)) / (cos(axis_theta)*cos(axis_phi)*cos(t) - sin(axis_theta)*sin(t)));
				y = (int)(0.5+(valuemap_phi_div*v_phi_y / PI));
				x = (int)(0.5+(valuemap_theta_div*v_theta_x / PI));
				trajectory_value += valuemap[y*valuemap_theta_div + x];
			}
			printf("trajectory value %f phi %f theta %f\n", trajectory_value, axis_phi, axis_theta);
			trajectory_index[valuemap_theta_div*i + j] = trajectory_value;


		}
	}//これで各軌道ごとの輝度値の合計が収納された。


	//軌道の中でよさげなものをいくつか選ぶ。最高のものをまず選んでからその周囲角度でない軌道を順に選んでいく。繰り返し数はとりあえず４．
	float trajectory_lanking[10][3];

	float axis_phi_max = 0;//よい軌道の一時保管庫
	float axis_theta_max = 0;//よい軌道の一時保管庫
	float trajectory_max = 0;//一番悪い軌道
	for (k = 0; k < valuemap_theta_div*valuemap_phi_div;k++) {
		if (trajectory_max < trajectory_index[k]) {
			trajectory_max = trajectory_index[k];
		}
	}
	for (k = 0; k < 10; k++) {//ランキング初期化
		trajectory_lanking[k][0] = trajectory_max;
		trajectory_lanking[k][1] = 0;
		trajectory_lanking[k][2] = 0;
	}

	int test = 0;

	for (k = 0; k < limit; k++) {
		trajectory_value =  trajectory_max;

		for (i = 0; i < valuemap_phi_div; i++) {
			axis_phi = i*PI / valuemap_phi_div;
			for (j = 0; j < valuemap_theta_div; j++) {
				axis_theta = j*PI / valuemap_theta_div;
				test = 0;
				//検証する軌道が既存の軌道とかぶっていないかの検定を行う
				for (l = 0; l < limit; l++) {
					if ((axis_phi >(trajectory_lanking[l][1] - ban) && axis_phi < (trajectory_lanking[l][1] + ban)) || (axis_theta >(trajectory_lanking[l][2] - ban) && axis_theta < (trajectory_lanking[l][2] + ban))) {
						test = 1;
					}
				}

				if (test < 0.5) {
					if (trajectory_value > trajectory_index[i*valuemap_theta_div + j]) {
						trajectory_value = trajectory_index[i*valuemap_theta_div + j];
						axis_phi_max = axis_phi;
						axis_theta_max = axis_theta;
					}
				}
			}
		}

		trajectory_lanking[k][0] = trajectory_value;
		trajectory_lanking[k][1] = axis_phi_max;
		trajectory_lanking[k][2] = axis_theta_max;
		printf("value_%d_%f axis phi_%f theta_%f\n", k,trajectory_lanking[k][0], trajectory_lanking[k][1], trajectory_lanking[k][2]);

	}




	printf("Writing trajectory\n");
	for (k = 0; k < limit; k++) {
		for (t = 0; t < PI; t += 0.05) {
			v_phi_y = acos(-sin(trajectory_lanking[k][1])*cos(t));
			v_theta_x = PI/2 + atan((-sin(trajectory_lanking[k][2])*cos(trajectory_lanking[k][1])*cos(t) + cos(trajectory_lanking[k][2])*sin(t)) / (cos(trajectory_lanking[k][2])*cos(trajectory_lanking[k][1])*cos(t) - sin(trajectory_lanking[k][2])*sin(t)));
			y = (int)(0.5+(valuemap_phi_div*v_phi_y / PI));
			x = (int)(0.5+(valuemap_theta_div*v_theta_x / PI));

			valuemap[y*valuemap_theta_div + x] = k*100;
	
		}
	}



	char name[1000];
	sprintf(name, "%s-trajectory-double-(axis_phi_%f_theta_%f_pdiv_%d_tdiv_%d).raw", argv[1],axis_phi, axis_theta, valuemap_phi_div, valuemap_theta_div);

	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(valuemap, sizeof(float), valuemap_phi_div*valuemap_theta_div, out);
	fclose(out);
	printf("\nwriting_end\n\n");


	////////////////メモリ開放///////////////
	delete[] trajectory_index;
	delete[] valuemap;

	////////////////メモリ開放///////////////

	printf("program_end\n");
	system("pause");

	return 0;
}
