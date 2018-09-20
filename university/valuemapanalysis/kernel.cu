
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

	int in_offset;//オフセットがあれば
	int valuemap_phi_div, valuemap_theta_div;//valuemapの分割数。頂点除外なので、phiは本当は+1となっている

	FILE *in_valuemap_file;//入力ファイル用のファイル容器

	printf("enter value map size in form of phi theta\n");//valuemapの縦と横の長さ入力
	scanf_s("%d %d", &valuemap_phi_div, &valuemap_theta_div);

	printf("enter offset if there is\n");//オフセットがあれば入力。
	scanf_s("%d", &in_offset);

	if ((in_valuemap_file = fopen(argv[1], "rb")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};
	printf("file_open_success\n");

	double *valuemap = new double[valuemap_phi_div*valuemap_theta_div];
	double *trajectory_index = new double[valuemap_phi_div*valuemap_theta_div];//評価軌道ごとに評価値を収納する。評価軌道軸の数は投影角度ととりあえず一致させる。
	for (i = 0; i < valuemap_phi_div*valuemap_theta_div;i++) {
		valuemap[i] = 0;
		trajectory_index[i] = 0;
	}
	printf("memory_success\n");

//	fseek(in_valuemap_file, in_offset, SEEK_SET);//読み込み時にオフセット移動

	fread(valuemap, sizeof(double), valuemap_phi_div*valuemap_theta_div, in_valuemap_file);//ボリューム読み込み
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
	double trajectory_value = 0;
	int maxx, maxy;

	//各軌道ごとに評価値を計算し、indexに収納
	for (i = 0; i < valuemap_phi_div; i++) {
		axis_theta = 0;
		axis_phi += PI / (valuemap_phi_div+1);

		for (j = 0; j < valuemap_theta_div; j++) {
			trajectory_value = 0;
			for (t = 0; t < PI; t += 0.05) {//軸上で半回転
				v_phi_y = acos(-sin(axis_phi)*cos(t));
				v_theta_x = PI/2 + atan((-sin(axis_theta)*cos(axis_phi)*cos(t) + cos(axis_theta)*sin(t)) / (cos(axis_theta)*cos(axis_phi)*cos(t) - sin(axis_theta)*sin(t)));
				y = (int)(valuemap_phi_div)*(v_phi_y / PI);
				x = (int)(valuemap_theta_div)*(v_theta_x / PI);
				trajectory_value += valuemap[y*valuemap_theta_div + x];
			}
			printf("trajectory value %lf phi %f theta %f\n", trajectory_value, axis_phi, axis_theta);
			trajectory_index[valuemap_theta_div*i + j] = trajectory_value;

			axis_theta += PI / valuemap_theta_div;
		}
	}//これで各軌道ごとの輝度値の合計が収納された。

	trajectory_value = trajectory_index[0];


	//軌道の中でよさげなものを選ぶ
	for (i = 0; i < valuemap_phi_div; i++) {
		for (j = 0; j < valuemap_theta_div; j++) {
			if (trajectory_value > trajectory_index[i*valuemap_theta_div + j]) {
				trajectory_value = trajectory_index[i*valuemap_theta_div + j];
				axis_phi = i*PI / (valuemap_phi_div + 1);
				axis_theta = j*PI / valuemap_theta_div;
			}
		}
	}
	printf("max_value_%lf axis phi_%f theta_%f\n", trajectory_value, axis_phi, axis_theta);

	printf("Writing trajectory\n");
	for (t = 0; t < PI; t += 0.05) {
		v_phi_y = acos(-sin(axis_phi)*cos(t));
		v_theta_x = PI/2 + atan((-sin(axis_theta)*cos(axis_phi)*cos(t) + cos(axis_theta)*sin(t)) / (cos(axis_theta)*cos(axis_phi)*cos(t) - sin(axis_theta)*sin(t)));
		y = (int)(valuemap_phi_div)*(v_phi_y / PI);
		x = (int)(valuemap_theta_div)*(v_theta_x / PI);

		valuemap[y*valuemap_theta_div + x] = 0;
	
	}


	char name[1000];
	sprintf(name, "%s-trajectory-double-(axis_phi_%f_theta_%f_pdiv_%d_tdiv_%d).raw", argv[1],axis_phi, axis_theta, valuemap_phi_div, valuemap_theta_div);

	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(valuemap, sizeof(double), valuemap_phi_div*valuemap_theta_div, out);
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
