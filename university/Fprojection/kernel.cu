
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

#include<stdio.h>
#include<string>
#include<fstream>
#include<math.h>

#define STEP 1
#define Filedivide 1
#define ThreadsPerBlock 512 //現在の CUDA では 512 が MAX


//数GBは確保可能らしいが、floatでぶち込んでるのであふれてる可能性はある。とりあえずはそのまま作る。

__global__ void forward_marching_GPU(unsigned short *d_input_volume, float *d_proj1, float *d_ray_position, float step_x, float step_y, float step_z, Params params) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int j;
	float v = 0;//輝度の格納庫
	//レイ一つ一つについて独立計算しているため、idは一つでいい。
		for (j = 0; j < params.projection_radius / 2.0 / STEP; j++) {//合計何ステップを踏むかをjで表している
			int x = (int)(d_ray_position[idx * 3] + j*step_x);//ここの一ステップごとに、xyz方向に何ボクセル分進むかを計算し、intで丸められて最終的な点座標を得ている。
			int y = (int)(d_ray_position[idx * 3 + 1] + j*step_y);
			int z = (int)(d_ray_position[idx * 3 + 2] + j*step_z);
			if (x > 0 && y > 0 && z > 0 && x < params.voxels_x && y < params.voxels_y && z < params.voxels_z) {
				//光線上の点が、ボリュームの置いてあるボックス空間にある時、
				v += d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];//ｖにその座表情にあるトモグラフのボクセルの値を加算していく。
			}
			int x2 = (int)(d_ray_position[idx * 3] - j*step_x);//逆方向にもマーチングする。
			int y2 = (int)(d_ray_position[idx * 3 + 1] - j*step_y);
			int z2 = (int)(d_ray_position[idx * 3 + 2] - j*step_z);
			if (x2 > 0 && y2 > 0 && z2 > 0 && x2 < params.voxels_x && y2 < params.voxels_y && z2 < params.voxels_z) {
				//光線上の点が、ボリュームの置いてあるボックス空間にある時、
				v += d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];//ｖにその座表にあるトモグラフのボクセルの値を加算していく。
			}
		}
		d_proj1[idx] = v;
}


int main(int argc, char** argv) {//argcとかには起動時に渡す変数がはいる。

	int i, j, k;//便利にカウントなどに使う数字
	Params params;//パラメータ構造体
	FILE *in, *para;//入力ファイル用のファイル容器
						  //入力ファイルと内部変数間の数字のやり取りの一時的なバッファ

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	fscanf(para, "%s", params.in_path_name);//設定ファイルからパラメータ読み込み。スペースを空けると次に行くのでファイル名にスペースはつけないように。
	fscanf(para, "%s", params.out_name);
	fscanf(para, "%d", &params.in_offset);
	fscanf(para, "%f %f", &params.source_object_distance, &params.source_detector_distance);
	fscanf(para, "%d %d %d", &params.voxels_x, &params.voxels_y, &params.voxels_z);
	fscanf(para, "%d %d", &params.pixels_x, &params.pixels_y);
	fscanf(para, "%f %f", &params.pixel_pitch_x, &params.pixel_pitch_y);
	fclose(para);

	printf("input projection angle in form of phi theta\n");//初期の投影角度設定
	scanf_s("%f %f", &params.projection_phi, &params.projection_theta);

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n pixels %d %d\n angle %f %f\n",
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.pixels_x, params.pixels_y,
		params.projection_phi, params.projection_theta);//パラメータ書き出し。



	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//コマンドラインからの実行ではフルのパスを指定しなければ見つからない。しかも\は2回かく。あるいは設定テキストを直接プログラムに放り投げてもよい。
	}
	printf("load_success\n");
	
	//平面サイズ×厚みの1次元配列作成
	unsigned short *input_volume = new unsigned short[params.voxels_y*params.voxels_x*params.voxels_z];
	printf("memory_success\n");

	fseek(in, params.in_offset, SEEK_SET);
	fread(input_volume, 2, params.voxels_x*params.voxels_y*params.voxels_z, in);
	fclose(in);
	printf("loading_success\n");


	//投影像のキャンバス作成
	params.projection_radius = params.voxels_x * params.voxels_x + params.voxels_y *params.voxels_y + params.voxels_z * params.voxels_z;
	params.projection_radius = sqrt(params.projection_radius);//投影像の一片の長さ
	params.projection_sides = (int)params.projection_radius;//ボクセル長さ、つまり整数に丸めた場合。

	float *proj1 = new float[params.projection_sides*params.projection_sides];
	printf("projection_canvas_successfully_created\n");

	//投影レイのスタート位置を格納する容器を作成。xyz座標で設定
	float *ray_position = new float[params.projection_sides*params.projection_sides * 3];//キャンバスサイズの3倍

	printf("ray_successfully_created\n");

	//座標上でのレイのスタート位置を設定。iがキャンバス上でのy座標,jがキャンバス上でのx座標に対応
	for (i = 0; i < params.projection_sides; i++) {
		for (j = 0; j < params.projection_sides; j++) {
			ray_position[(i*params.projection_sides + j) * 3] = j - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 1] = i - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 2] = 0;
		}
	}

	//スタート座標をx,y軸に対してtheta,phiだけ回転させる
	float a, b, c;
	for (i = 0; i < params.projection_sides*params.projection_sides; i++) {
		//まずx軸に対して回転
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = a;
		ray_position[i * 3 + 1] = cos(params.projection_theta)*b;//-sin(params.projection_theta)*c
		ray_position[i * 3 + 2] = sin(params.projection_theta)*b;//+cos(params.projection_theta)*c
		//次にy軸に対して回転
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = cos(params.projection_phi)*a+sin(params.projection_phi)*c;
		ray_position[i * 3 + 1] = b;
		ray_position[i * 3 + 2] = -sin(params.projection_phi)*a + cos(params.projection_phi)*c;
		//ボリュームの中心座標に平行移動
		ray_position[i * 3] = ray_position[i * 3] + params.voxels_x / 2.0 - 0.5;
		ray_position[i * 3 + 1] = ray_position[i * 3 + 1] + params.voxels_y / 2.0 - 0.5;
		ray_position[i * 3 + 2] = ray_position[i * 3 + 2] + params.voxels_z / 2.0 - 0.5;
		//printf("ray_2_position(%f,%f,%f)\n", ray_position[i * 3], ray_position[i * 3 + 1], ray_position[i * 3 + 1]);
	}
	
	float ray_step[3] = { cos(params.projection_theta)*sin(params.projection_phi)*STEP ,-sin(params.projection_theta)*STEP ,cos(params.projection_theta)*cos(params.projection_phi)*STEP };
	//ここでレイの方向を決定。ついでに、ボクセル長さがそのまま座標系の長さに対応しているため、そのままステップとして用いれる。

	printf("projection_sides_%d ray_marching_sum_%d\n", params.projection_sides,params.projection_sides*params.projection_sides);



	//ここからGPUに転送して計算していく。

	float* d_proj1;
	float* d_ray_position;
	unsigned short* d_input_volume;

	cudaMalloc(&d_proj1, sizeof(float)*params.projection_sides*params.projection_sides);
	cudaMalloc(&d_ray_position,sizeof(float)*params.projection_sides*params.projection_sides * 3);
	cudaMalloc(&d_input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z);	

	cudaMemcpy(d_proj1, proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ray_position, ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);
	
	printf("ray_marching_start\n");
	
	//ブロック数設定とGPU呼び出し
	int BlockNum = (params.projection_sides*params.projection_sides + ThreadsPerBlock - 1) / ThreadsPerBlock;
	forward_marching_GPU <<< BlockNum, ThreadsPerBlock >>> ( d_input_volume,  d_proj1, d_ray_position, ray_step[0], ray_step[1], ray_step[2], params);

	//投影像の回収とCUDAメモリ開放
	cudaMemcpy(proj1, d_proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyDeviceToHost);

	cudaFree(d_input_volume);
	cudaFree(d_proj1);
	cudaFree(d_ray_position);



	printf("ray_marching_end\n");

	printf("Writing\n");//ここから先は書き出し。書き出しファイルに形式の指定などはない。
	char name[1000];
	sprintf(name, "%s-float-%dx%d-(%f-%f-%f).raw", params.out_name, params.projection_sides, params.projection_sides,ray_step[0], ray_step[1], ray_step[2]);
	printf("%s", name);
	
	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	
	//outに投影キャンバスをぶち込む。とりあえずはフロートで
	fwrite(proj1, sizeof(float), params.projection_sides*params.projection_sides, out);
	
	fclose(out);
	printf("\nwriting_end\n");

//コマンドラインからの実行では出力ファイルはプログラムのフォルダ内にできる。直接テキストを投げ込むとテキストのあるフォルダにできる。
	

	/*
	順投影の改良方針
	全方向から見れる。線源とディテクタ距離の指定がめんどいので、平行投影を作る。
	基本的に、入力トモグラフの1ボクセルを単位距離とし、トモグラフのはじのボクセルを座標中心とする座標で操作を行う。
	(0,0,0)にボクセルの端が来る。((辺の長さ-1)/2)でボックス真ん中座標となる。座標はフロート
	投影の角度設定は緯度と経度を用いる。z座標方向に開店後、xy平面上のある軸上で回転させる。
	投影角度の指定にもとづき、キャンバス上の各光線の出発座標位置を変換する。これはフロートでよい。
	光線のステップサイズとマーチング距離を決定する。ステップ距離は変えられるがとりあえずボクセルサイズ、距離は球体の直径。
	
	ここからGPUで行いたい。
	ボリュームと投影像、投影出発点のメモリを動的にCUDAに確保し、コピー、あとはレイの方向やステップ回数、レイの個数など必要な数字をぶち込んで計算する。
	おわったら、元のデータ容器にCUDAからコピーして、CUDAのメモリを開放。
	完了

	終了したら出力する
	*/



	delete[] input_volume;
	delete[] proj1;
	delete[] ray_position;
	printf("program_end\n");

	return 0;
}