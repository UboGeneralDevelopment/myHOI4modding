
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

#include<stdio.h>
#include<string>
#include<fstream>
#include<math.h>

#define STEP 1
#define PI 3.1415926535897932384626433832795f //円周率
#define Filedivide 1　//読み込みファイルの分割。結局使わなかった。
#define ThreadsPerBlock 512 //現在の CUDA では 512 が MAX


/*
方針

とりあえず、縦と横の分割数を決定して分割数だけ画像を生成させる。方向検討は分割方向とは依存する必要性はないが、とりあえずは検討方向は分割方向と一致させる。

また、取ってくるのは透過率の合計ではなく、その最大値のみを利用。より透過像"らしく"なる

とりあえず、画像一枚につき何かしらの評価値を算定し、それを各投影角度換算を行って合計したものをその角度における合計の評価値として計算する。
画像一枚の評価値は、輝度値*ピクセル数の合計とする。評価値を算定する部分を分割して改良がしやすいようにする。

ファイル読み込み終了時点で、投影角度枚数と試行角度回数設定を行う。
投影キャンバス作成→レイマーチング計算→評価値算出→メモリ開放　までを一連のプロセスとしてモジュール化。

評価値の再比較検討を行い最終回転軸角度の算出及びその角度におけるデモ投影像の作成。

投影出発ファイルと投影像ファイルはmain内で作成して、回転設定だけ外部関数に投げる。


現状、複数枚数の生産ができない。ふぁっく

*/

__global__ void forward_marching_GPU(unsigned short *d_input_volume, float *d_proj1, float *d_ray_position, float step_x, float step_y, float step_z, Params params) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int j;
	float v = 0;//輝度の格納庫
	d_proj1[idx] = 0;
	//レイ一つ一つについて独立計算しているため、idは一つでいい。
		for (j = 0; j < params.projection_radius / 2.0 / STEP; j++) {//合計何ステップを踏むかをjで表している
			int x = (int)(d_ray_position[idx * 3] + j*step_x);//ここの一ステップごとに、xyz方向に何ボクセル分進むかを計算し、intで丸められて最終的な点座標を得ている。
			int y = (int)(d_ray_position[idx * 3 + 1] + j*step_y);
			int z = (int)(d_ray_position[idx * 3 + 2] + j*step_z);
			if (x > 0 && y > 0 && z > 0 && x < params.voxels_x && y < params.voxels_y && z < params.voxels_z) {
				//光線上の点が、ボリュームの置いてあるボックス空間にある時、
				//v += d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];//ｖにその座表情にあるトモグラフのボクセルの値を加算していく。
				if (v < d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y]) {//最大値をとる。
					v = d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];
				}
			}
			int x2 = (int)(d_ray_position[idx * 3] - j*step_x);//逆方向にもマーチングする。
			int y2 = (int)(d_ray_position[idx * 3 + 1] - j*step_y);
			int z2 = (int)(d_ray_position[idx * 3 + 2] - j*step_z);
			if (x2 > 0 && y2 > 0 && z2 > 0 && x2 < params.voxels_x && y2 < params.voxels_y && z2 < params.voxels_z) {
				//光線上の点が、ボリュームの置いてあるボックス空間にある時、
				//v += d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];//ｖにその座表にあるトモグラフのボクセルの値を加算していく。
				if (v < d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y]) {//最大値をとる。
					v = d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];
				}
			}
		}
		d_proj1[idx] = v;
}


void ray_start_setting(Params params, float *ray_position) {
	int i, j;
	//初期化。座標中心上においてレイのスタート位置を設定。iがキャンバス上でのy座標,jがキャンバス上でのx座標に対応
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
		ray_position[i * 3] = cos(params.projection_phi)*a + sin(params.projection_phi)*c;
		ray_position[i * 3 + 1] = b;
		ray_position[i * 3 + 2] = -sin(params.projection_phi)*a + cos(params.projection_phi)*c;
		//ボリュームの中心座標に平行移動
		ray_position[i * 3] = ray_position[i * 3] + params.voxels_x / 2.0 - 0.5;
		ray_position[i * 3 + 1] = ray_position[i * 3 + 1] + params.voxels_y / 2.0 - 0.5;
		ray_position[i * 3 + 2] = ray_position[i * 3 + 2] + params.voxels_z / 2.0 - 0.5;
		//printf("ray_2_position(%f,%f,%f)\n", ray_position[i * 3], ray_position[i * 3 + 1], ray_position[i * 3 + 1]);
	}
}





int main(int argc, char** argv) {//argcとかには起動時に渡す変数がはいる。

	int i, j, k;//便利にカウントなどに使う数字
	Params params;//パラメータ構造体
	FILE *in, *para;//入力ファイル用のファイル容器
						  //入力ファイルと内部変数間の数字のやり取りの一時的なバッファ

	//////////////////////パラメータ読み込み//////////////////////////

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};
	
	//設定ファイルからパラメータ読み込み。スペースを空けると次に行くのでファイル名にスペースはつけないように。
	fscanf(para, "%s", params.in_path_name);
	fscanf(para, "%s", params.out_name);
	fscanf(para, "%d", &params.in_offset);
	fscanf(para, "%f %f", &params.source_object_distance, &params.source_detector_distance);//多分使わない
	fscanf(para, "%d %d %d", &params.voxels_x, &params.voxels_y, &params.voxels_z);
	fscanf(para, "%f %f", &params.projection_phi, &params.projection_theta);//あまりつかわない
	fclose(para);

	printf("projection angle division number in form of phi theta\n");//投影角度の数の設定
	scanf_s("%d %d", &params.projection_div_phi, &params.projection_div_theta);

	params.projection_radius = sqrt(params.voxels_x * params.voxels_x + params.voxels_y *params.voxels_y + params.voxels_z * params.voxels_z);//投影像の一片の長さ
	params.projection_sides = (int)params.projection_radius;//ボクセル長さ、つまり整数に丸めた場合。

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n angle division phi %f theta %f\n projection radius %f/n projection sides %d",
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.projection_div_phi, params.projection_div_theta,
		params.projection_radius,params.projection_sides);//パラメータ書き出し。
	//////////////////////パラメータ読み込み終了//////////////////////////

	//////////////////////ファイル読み込み//////////////////////////
	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//コマンドラインからの実行ではフルのパスを指定しなければ見つからない。しかも\は2回かく。あるいは設定テキストを直接プログラムに放り投げてもよい。
	}
	printf("load_success\n");
	
	//平面サイズ×厚みの1次元配列作成
	unsigned short *input_volume = new unsigned short[params.voxels_y*params.voxels_x*params.voxels_z];
	printf("memory_success\n");

	fseek(in, params.in_offset, SEEK_SET);//読み込み時にオフセット移動
	fread(input_volume, 2, params.voxels_x*params.voxels_y*params.voxels_z, in);//読み込み～～
	fclose(in);
	printf("loading_success\n");
	//////////////////////ファイル読み込み終了//////////////////////////
	
	float ray_step[3];//レイの格納容器作成
	char name[1000];

	//////////////////////繰り返し対象////////////////////////
	//////////////////////投影方向の設定////////////////////////

	for (i = 0; i < params.projection_div_phi ; i++ ) {
		params.projection_phi = PI*(1.0f*i / params.projection_div_phi);
		//for (j = 0; j < params.projection_div_theta; j++) {
			//params.projection_theta = PI*(1.0f*j / params.projection_div_theta);
			//ここで投影キャンバス作成、GPUでの投影像作成、評価値の計算まで計算が繰り返される。

	//ここの方向の繰り返し設定を実装しなければがなければ自動的に初期設定一枚のみがはしる。

	//////////////////////投影方向の設定終了////////////////////////


	//////////////////////投影キャンバス初期化、投影スタート位置、レイ設定キャンバス作成//////////////////////////
	printf("projection_setting_start\n");


	float *proj1 = new float[params.projection_sides*params.projection_sides];//投影像のキャンバス作成

	//投影レイのスタート位置を格納する容器を作成。xyz座標で設定
	float *ray_position = new float[params.projection_sides*params.projection_sides * 3];//キャンバスサイズの3倍
	//ここから、スタート位置、レイ、パラメータを投げ込んで開店したスタート位置とレイを受け取る関数を作る。
	
	ray_start_setting(params, ray_position);
	
	ray_step[0] = { cos(params.projection_theta)*sin(params.projection_phi)*STEP};
	ray_step[1] = { -sin(params.projection_theta)*STEP};
	ray_step[2] = { cos(params.projection_theta)*cos(params.projection_phi)*STEP };
	//ここでレイの方向を決定。ついでに、ボクセル長さがそのまま座標系の長さに対応しているため、そのままステップとして用いれる。
	
	printf("ray,ray_canvas,ray_start_positions are successfully_created\n");
	//////////////////////投影キャンバス作成終了//////////////////////////


	//////////////////GPU転送およびレイマーチング計算///////////////////////

	float* d_proj1;
	float* d_ray_position;
	unsigned short* d_input_volume;
	//GPUメモリ確保
	cudaMalloc(&d_proj1, sizeof(float)*params.projection_sides*params.projection_sides);
	cudaMalloc(&d_ray_position,sizeof(float)*params.projection_sides*params.projection_sides * 3);
	cudaMalloc(&d_input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z);	
	//GPUにデータ転送
	cudaMemcpy(d_proj1, proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ray_position, ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);
	
	printf("ray_marching_start\n");
	
	//ブロック数設定とGPU計算呼び出し
	int BlockNum = (params.projection_sides*params.projection_sides + ThreadsPerBlock - 1) / ThreadsPerBlock;
	forward_marching_GPU <<< BlockNum, ThreadsPerBlock >>> ( d_input_volume,  d_proj1, d_ray_position, ray_step[0], ray_step[1], ray_step[2], params);

	//投影像の回収とCUDAメモリ開放
	cudaMemcpy(proj1, d_proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyDeviceToHost);

	cudaFree(d_input_volume);
	cudaFree(d_proj1);
	cudaFree(d_ray_position);
	printf("ray_marching_end\n");

	//////////////////////GPU転送おわり//////////////////////////


	//////////////////////評価値の計算////////////////////

	//////////////////////評価値の計算終了/////////////////////////


	//////////////////////書き出し(オプション)//////////////////////////

	printf("Writing\n");//ここから先は書き出し。書き出しファイルに形式の指定などはない。

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

	delete[] proj1;
	delete[] ray_position;	

		//}
	}
	//////////////////////書き出し終了//////////////////////////
	//////////////////////繰り返し対象終了////////////////////////


	//////////////////////評価値の比較検討////////////////////

	//////////////////////評価値の比較検討終了////////////////




	delete[] input_volume;




	printf("program_end\n");

	return 0;
}