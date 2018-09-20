
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
#define angle_div 50 //評価時の分割数

/*
方針

縦と横の分割数については、最大分割数設定後、格緯度ごとに自動生成する。方向検討は分割方向とは必ずしも依存しない。
頂点における撮像回転は意味がないため、分割数xに対して実際にはx+1分割し、その最初と最後の頂点位置での撮像を行わない。

取ってくるのは透過率の合計ではなく、その最大値のみを利用。より透過像"らしく"なる。
最大値のみをとるので、当映像はuint16,unsigne intファイル。仮に投下率の合計を出したければ平均する。

書き出す透過率と角度のマップはdoubleで出力。

ファイル読み込み終了時点で、投影角度枚数と試行角度回数設定を行う。
投影キャンバス作成→レイマーチング計算→評価値算出→メモリ開放　までを一連のプロセスとしてモジュール化。
投影出発ファイルと投影像ファイルはmain内で作成して、回転設定だけ外部関数に投げる。

フラグメントがあるので、mallocとかfreeはfor文の外。
ボリュームのGPUへの読み込みは一回のみ。

*/

__global__ void forward_marching_GPU(unsigned short *d_input_volume, unsigned short *d_proj1, float *d_ray_position, float step_x, float step_y, float step_z, Params params) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int j;
	int x, y, z, x2, y2, z2;
	float vf = 0.0;
	unsigned short v = 0;//輝度の格納庫
	d_proj1[idx] = 0;
	//レイ一つ一つについて独立計算しているため、idは一つでいい。
		for (j = 0; j < params.projection_radius / 2.0 / STEP; j++) {//合計何ステップを踏むかをjで表している
			x = (int)(d_ray_position[idx * 3] + j*step_x);//ここの一ステップごとに、xyz方向に何ボクセル分進むかを計算し、intで丸められて最終的な点座標を得ている。
			y = (int)(d_ray_position[idx * 3 + 1] + j*step_y);
			z = (int)(d_ray_position[idx * 3 + 2] + j*step_z);
			if (x > 0 && y > 0 && z > 0 && x < params.voxels_x && y < params.voxels_y && z < params.voxels_z) {
				//光線上の点が、ボリュームの置いてあるボックス空間にある時、
				//v += d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];//ｖにその座表情にあるトモグラフのボクセルの値を加算していく。
				if (vf < d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y]) {//最大値をとる。
					vf = d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];
				}
			}
			x2 = (int)(d_ray_position[idx * 3] - j*step_x);//逆方向にもマーチングする。
			y2 = (int)(d_ray_position[idx * 3 + 1] - j*step_y);
			z2 = (int)(d_ray_position[idx * 3 + 2] - j*step_z);
			if (x2 > 0 && y2 > 0 && z2 > 0 && x2 < params.voxels_x && y2 < params.voxels_y && z2 < params.voxels_z) {
				//光線上の点が、ボリュームの置いてあるボックス空間にある時、
				//v += d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];//ｖにその座表にあるトモグラフのボクセルの値を加算していく。
				if (vf < d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y]) {//最大値をとる。
					vf = d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];
				}
			}
		}
		v = (int)vf;
		d_proj1[idx] = v;
}


void ray_start_setting(Params params, float *ray_position, float ray_phi, float ray_theta) {
	int i, j;
	//初期化。座標中心上においてレイのスタート位置を設定。iがキャンバス上でのy座標,jがキャンバス上でのx座標に対応。
	for (i = 0; i < params.projection_sides; i++) {
		for (j = 0; j < params.projection_sides; j++) {
			ray_position[(i*params.projection_sides + j) * 3] = j - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 1] = i - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 2] = 0;
		}
	}

	//スタート座標をy,z軸に対してtheta,phiだけ回転させたのち、平行移動させる
	float a, b, c;
	for (i = 0; i < params.projection_sides*params.projection_sides; i++) {
		//まずy軸にそって回転
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = cos(ray_phi)*a + sin(ray_phi)*c;
		ray_position[i * 3 + 1] = b;
		ray_position[i * 3 + 2] = -sin(ray_phi)*a + cos(ray_phi)*c;
	 //次にz軸にそって回転
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = cos(ray_theta)*a - sin(ray_theta)*b;
		ray_position[i * 3 + 1] = sin(ray_theta)*a + cos(ray_theta)*b;
		ray_position[i * 3 + 2] = c;

		//ボリュームの中心座標に平行移動
		ray_position[i * 3] = ray_position[i * 3] + params.voxels_x / 2.0 - 0.5;
		ray_position[i * 3 + 1] = ray_position[i * 3 + 1] + params.voxels_y / 2.0 - 0.5;
		ray_position[i * 3 + 2] = ray_position[i * 3 + 2] + params.voxels_z / 2.0 - 0.5;
		//printf("ray_2_position(%f,%f,%f)\n", ray_position[i * 3], ray_position[i * 3 + 1], ray_position[i * 3 + 1]);
	}

}



int main(int argc, char** argv) {//argcとかには起動時に渡す変数がはいる。

	int i, j, k, l;//便利にカウントなどに使う数字
	Params params;//パラメータ構造体
	FILE *in, *para;//入力ファイル用のファイル容器
						  //入力ファイルと内部変数間の数字のやり取りの一時的なバッファ

	//////////////////////パラメータ読み込みとパラメータ設定//////////////////////////

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	//設定ファイルからパラメータ読み込み。スペースを空けると次に行くのでファイル名にスペースはつけない。
	fscanf(para, "%s", params.in_path_name);
	fscanf(para, "%s", params.out_name);
	fscanf(para, "%d", &params.in_offset);
	fscanf(para, "%f %f", &params.source_object_distance, &params.source_detector_distance);//多分使わない
	fscanf(para, "%d %d %d", &params.voxels_x, &params.voxels_y, &params.voxels_z);
	fscanf(para, "%f %f", &params.projection_phi, &params.projection_theta);//方向の初期設定。あまりつかわない
	fclose(para);

	params.projection_div_phi = 4;//とりあえず定義。したのscanでよみこませてもよい。
	params.projection_div_theta = 4;

	printf("projection angle division number in form of phi theta\n");//投影角度の枚数の設定。ここで読み込む内容はよく変更する。
	scanf_s("%d %d", &params.projection_div_phi, &params.projection_div_theta);

	params.projection_radius = sqrt(params.voxels_x * params.voxels_x + params.voxels_y *params.voxels_y + params.voxels_z * params.voxels_z);//投影像の一片の長さ
	params.projection_sides = (int)params.projection_radius;//ボクセル長さ、つまり整数に丸めた場合。

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n start angle phi %f theta %f\n angle division phi %d theta %d\n projection radius %f projection sides %d",
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.projection_phi, params.projection_theta,
		params.projection_div_phi, params.projection_div_theta,
		params.projection_radius, params.projection_sides);//パラメータ書き出し。




	//////////////////////読み込みと各種メモリ確保,GPU転送//////////////////////////
	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//コマンドラインからの実行ではフルのパスを指定しなければ見つからない。しかも\は2回かく。あるいは設定テキストを直接プログラムに放てもよい。
	}
	printf("load_success\n");

	unsigned short *input_volume = new unsigned short[params.voxels_y*params.voxels_x*params.voxels_z];
	//平面サイズ×厚みの1次元配列作成
	printf("memory_success\n");

	fseek(in, params.in_offset, SEEK_SET);//読み込み時にオフセット移動
	fread(input_volume, 2, params.voxels_x*params.voxels_y*params.voxels_z, in);//ボリューム読み込み
	fclose(in);
	printf("loading_success\n");

	unsigned short *proj1 = new unsigned short[params.projection_sides*params.projection_sides];
	//投影像のメモリ確保。CPU側
	float *ray_position = new float[params.projection_sides*params.projection_sides * 3];
	//キャンバスサイズの3倍の投影レイのスタート位置を格納するメモリ。CPU側。xyz座標で設定
	float ray_step[3];
	//レイの格納容器作成
	float ray_phi = 0.0;
	float ray_theta = 0.0;
	//レイの方向初期設定
	char name[1000];
	//書き出し用の名前
	double *valuemap = new double[params.projection_div_phi*params.projection_div_theta];
	for (k = 0; k < params.projection_div_phi*params.projection_div_theta; k++) {
		valuemap[k] = 0;
	}
	//評価値のメモリ

	printf("projection_setting_start\n");
	//////////////////////以下GPUでのメモリ確保
	unsigned short* d_proj1;
	float* d_ray_position;
	unsigned short* d_input_volume;
	//GPUメモリ確保
	cudaMalloc(&d_proj1, sizeof(unsigned short)*params.projection_sides*params.projection_sides);
	cudaMalloc(&d_ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3);
	cudaMalloc(&d_input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z);

	//ボリュームを先にGPUへと転送
	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);




	/////////////////////////繰り返し対象。投影を作り、評価し、評価マップを作製////////////////////////////////////

	for (i = 0; i < params.projection_div_phi; i++) {//繰り返し回数の設定。phi方向,theta方向回転で二重ループになっている。。
		//theta方向回転時の初期設定。phiに少しずつ移動させる。
		ray_theta = 0;
		ray_phi += PI / (params.projection_div_phi + 1);

		for (j = 0; j < params.projection_div_theta; j++) {

			printf("phi(%f)_theta(%f)\n",ray_phi, ray_theta);
			//////////////////////投影メモリ、投影スタート位置設定、レイ方向設定の初期化//////////////////////////

			for (k = 0; k < params.projection_sides*params.projection_sides; k++) {
				proj1[k] = 0;
			}//投影メモリ初期化

			//レイの方向を決定。角度theta0phi0で(0,0,1)。両方向に伸びてく。ボクセル長さがそのまま座標系の長さに対応しているため、そのままステップとして用いれる。			ray_step[0] = sin(ray_phi)*cos(ray_theta)*STEP;
			ray_step[1] = sin(ray_phi)*sin(ray_theta)*STEP;
			ray_step[2] = cos(ray_phi)*STEP;

			ray_start_setting(params, ray_position, ray_phi, ray_theta);
			//スタート位置設定、スタート位置と、パラメータを投げ込んで回転させる。



			//////////////////GPU転送および計算///////////////////////

			//GPUに投影キャンバスと投影位置設定データ転送
			cudaMemcpy(d_proj1, proj1, sizeof(unsigned short)*params.projection_sides*params.projection_sides, cudaMemcpyHostToDevice);
			cudaMemcpy(d_ray_position, ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3, cudaMemcpyHostToDevice);
		//	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);

			printf("ray_marching_start\n");

			//ブロック数設定とGPU計算呼び出し
			int BlockNum = (params.projection_sides*params.projection_sides + ThreadsPerBlock - 1) / ThreadsPerBlock;
			forward_marching_GPU << < BlockNum, ThreadsPerBlock >> > (d_input_volume, d_proj1, d_ray_position, ray_step[0], ray_step[1], ray_step[2], params);

			//投影像の回収
			cudaMemcpy(proj1, d_proj1, sizeof(unsigned short)*params.projection_sides*params.projection_sides, cudaMemcpyDeviceToHost);

			printf("ray_marching_end\n");

	


			//////////////////////評価値の計算////////////////////

			for (k = 0; k < params.projection_sides*params.projection_sides; k++) {
				valuemap[params.projection_div_theta*i + j] += proj1[k];
			}
			printf("value_%lf\n", valuemap[params.projection_div_theta*i + j]);
			



			//////////////////////書き出し(オプション)//////////////////////////
			if ( i % 10 == 0 && j % 10 == 0){
				printf("Writing projection\n");//ここから先は書き出し。書き出しファイルに形式の指定などはない。
				sprintf(name, "%s-uint16-%dx%d-(%f_%f).raw", params.out_name, params.projection_sides, params.projection_sides,ray_phi, ray_theta);
				printf("%s", name);
				FILE *out;
				out = fopen(name, "wb");
				if (out == NULL) {
					printf("\nFILE cannot open\n");
					exit(0);
				};
				fwrite(proj1, sizeof(unsigned short), params.projection_sides*params.projection_sides, out);
				fclose(out);
				printf("\nwriting_end\n\n");
			}
			//コマンドラインからの実行では出力ファイルはプログラムのフォルダ内にできる。直接テキストを投げ込むとテキストのあるフォルダにできる。
			



			//角度をtheta方向に更新
			ray_theta += PI / (params.projection_div_theta + 1);
		}
	}



	////////////////メモリ開放///////////////
	cudaFree(d_input_volume);
	cudaFree(d_proj1);
	cudaFree(d_ray_position);	
	delete[] proj1;
	delete[] ray_position;	
	delete[] input_volume;
	////////////////メモリ開放///////////////

	//////////////////////評価値の比較検討////////////////////
	/*
	float vphi, vtheta;
	float *value = new float[angle_div*angle_div];
	float max_value[3];
	for (i = 0; i < params.projection_div_theta*params.projection_div_phi; i++) {
		printf("value %f (phi %f theta %f)\n", temp[3*i], temp[3*i + 1], temp[3*i + 2]);

		for (j = 0; j < angle_div; j++) {
			for (k = 0; k < angle_div; k++) {
				value[j*angle_div + k] = 0;
			}
		}
		for (j = 0; j < angle_div; j++) {
			vphi = PI*2.0f*j/angle_div;
			for (k = 0; k < angle_div; k++) {
				vtheta = PI*2.0f*k / angle_div;
				for (l = 0; l < params.projection_div_phi * params.projection_div_theta ; l++) {
					value[j*angle_div + k] += temp[3*i]*(sin(temp[3 * l + 1])*cos(temp[3 * l + 2])*sin(vphi)*cos(vtheta) + sin(temp[3 * l + 1])*sin(temp[3 * l + 2])*sin(vphi)*sin(vtheta) + cos(temp[3 * l + 1])*cos(vphi));
				}
				//printf("value %f phi %f theta %f\n", value[j*angle_div + k], vphi, vtheta);
				if (max_value[0]<value[j*angle_div + k]) {
					max_value[0] = value[j*angle_div + k];
					max_value[1] = vphi;
					max_value[2] = vtheta;
				}
			}
		}
	}
	printf("max_value %f phi %f theta %f\n", max_value[0], max_value[1], max_value[2]);
	for (i = 0; i < params.projection_div_phi*params.projection_div_theta;i++) {
		printf("value %f \n", valuemap[i]);
	}
	delete[] value;
	*/
	



	///評価値の比較二つ目。
	printf("Writing value map\n");
	sprintf(name, "valuemap-%s-double-(phi%d_theta%d).raw", params.out_name, params.projection_div_phi, params.projection_div_theta);
	printf("%s", name);
	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(valuemap, sizeof(double), params.projection_div_phi*params.projection_div_theta, out);
	fclose(out);
	printf("\nwriting_end\n\n");//valuemapの書き出し
	
	printf("\n projection_trajectory_analysis_start\n");
	float v_phi_y = 0, v_theta_x = 0;//value_map上での座標
	int x, y;
	float axis_phi = 0, axis_theta = 0;//trajectoryの軸角度。tempに一応格納してある。
	float t = 0;//軌道の角度。開店する。
	double trajectory_value = 0;
	double *trajectory_index = new double[(params.projection_div_phi - 1)*params.projection_div_theta];//とりあえず、評価軌道は投影軸の数と一致させる。

	for (i = 0; i < params.projection_div_phi - 1; i++) {
		axis_theta = 0;
		axis_phi += PI / params.projection_div_phi;
		for (j = 0; j < params.projection_div_theta; j++ ) {
			trajectory_value = 0;
			for (t = 0; t < PI; t += 0.05) {//軸上で半回転
				v_phi_y = acos(-sin(axis_phi)*cos(t));
				v_theta_x = PI/2 + atan((-sin(axis_theta)*cos(axis_phi)*cos(t) + cos(axis_theta)*sin(t)) / (cos(axis_theta)*cos(axis_phi)*cos(t) - sin(axis_theta)*sin(t)));
				y = (int)(params.projection_div_phi - 1)*(v_phi_y / PI);
				x = (int)(params.projection_div_theta)*(v_theta_x / PI);
				trajectory_value += valuemap[y*params.projection_div_theta + x];
			}
			printf("trajectory value %lf phi %f theta %f\n", trajectory_value, axis_phi, axis_theta);
			trajectory_index[params.projection_div_theta*i + j] = trajectory_value;
			axis_theta += PI / params.projection_div_theta;
		}
	}//これで各軌道ごとの輝度値の合計が収納された。

	trajectory_value = trajectory_index[0];
	for (i = 0; i < params.projection_div_phi - 1; i++) {
		for (j = 0; j < params.projection_div_theta; j++) {
			if (trajectory_value > trajectory_index[params.projection_div_theta*i + j]) {
				trajectory_value = trajectory_index[params.projection_div_theta*i + j];
				x = j;
				y = i;
				axis_phi = x*PI / params.projection_div_phi;
				axis_theta = y*PI / params.projection_div_theta;
			}	
		}
	}
	printf("max_value_%lf axis phi_%f theta_%f\n", trajectory_value, axis_phi, axis_theta);


	////////////////メモリ開放///////////////
	delete[] trajectory_index;
	delete[] valuemap;
	
	////////////////メモリ開放///////////////

	printf("program_end\n");
	system("pause");
	return 0;
}