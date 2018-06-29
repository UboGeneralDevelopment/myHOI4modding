
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

#include <stdio.h>
#include<string>
#include<fstream>
#include<math.h>


int main(int argc, char** argv) {//argcとかには起動時に渡す変数がはいる。

	int i, j, k;//便利にカウントなどに使う数字
	Params params;//パラメータ構造体
	FILE *in, *out, *para;//入力ファイル用のファイル容器
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
	scanf_s("%f %f",&params.projection_phi,&params.projection_theta);

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n pixels %d %d\n angle %f %f\n", 
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.pixels_x,params.pixels_y,
		params.projection_phi,params.projection_theta);//パラメータ書き出し。

	int volume_size = params.voxels_x*params.voxels_y*params.voxels_z;
	printf("total voxels %d\n", volume_size);//あんまりいらないけどいちおうボクセル総数

	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//コマンドラインからの実行ではフルのパスを指定しなければ見つからない。あるいは設定テキストを直接プログラムに放り投げてもよい。
	}
	printf("load_success\n");

	unsigned short **input_volume = new unsigned short*[params.voxels_z];
	for (int i = 0; i < params.voxels_z; i++) { 
		input_volume[i] = new unsigned short[params.voxels_x*params.voxels_y];
	}//平面サイズ×厚みの2次元配列を確保。3次元配列に読み込もうとするやたら時間かかるので2次元で
	printf("memory_success\n");

	unsigned short *buff = new unsigned short[params.voxels_y*params.voxels_x];
	fseek(in, params.in_offset,SEEK_SET);
	for (i = 0; i<params.voxels_z; i++) {
		fread(buff, 2, params.voxels_x*params.voxels_y, in);
		for (j = 0; j<params.voxels_y*params.voxels_x; j++) {	
				input_volume[i][j] = buff[j];
		}
		printf("loading%d\n",i);
	}//input_volumeへのファイルの読み込み。平面ごとにbuffにいったん入れてからinput_volumeへ写す。
	delete[] buff;
	fclose(in);

	unsigned short *proj1 = new unsigned short[params.pixels_x*params.pixels_y];//proj1に投影制作
	int x, y, z;




	/*
	順投影の開発方針
	投影像は一枚のみ作るが、全方向から見れる。
	線源とディテクタ距離の指定がめんどいので、平衡投影を作る。
	まず、トモグラフをもとに、トモグラフボックスの中心点を中心とする友グラフを包み込む大きさの球形の投影範囲を指定する。
	球体の直径サイズの投影像キャンバスを作成する
	投影像キャンバスの各ピクセルにぶつかりに行く光線を設定する。今回は光線の出発座標のみでよい。
	ステップサイズを決定する。ボクセルサイズの半分としておく。ボクセルサイズはわからないため、自ら計算する必要がある。とりあえず１とでもおいておく。
	
	ここから先はGPU
	指定された角度に対して、トモグラフ中心を中心とする座標軸上での光点の出発点を決める。
	ステップごとに、xyz方向にどれだけ進むかをボクセル距離換算であらわす。
	点の出発点からステップごとに点の座標をボクセル座標で整数で出し、その位置でのトモグラフの値を調べるレイマーチングを行う。
	ただし、トモグラフボックスの存在する座標でのみ、値を取得し、線状の値を合計して投影像キャンバスにぶちこむ。
	以上
	*/



	printf("Writing\n");//ここから先は書き出し。書き出しファイルに形式の指定などはない。
	char name[1000];
	sprintf(name, "%s-uint16.raw",params.out_name);
	out = fopen(name, "wb");//outをnameとしてひらく。
	if (out == NULL) {          // オープンに失敗した場合
		printf("cannot open\n");         // エラーメッセージを出して
		exit(0);                         // 異常終了
	}

	for (i = 0; i < params.voxels_z; i++) {//ここではoutに直接inputを出力するファイルとなっている。出力内容に応じてこの部分は書き換える。
		fwrite(input_volume[i], 2, params.voxels_x*params.voxels_y, out);
	}//コマンドラインからの実行では出力ファイルはプログラムのフォルダ内にできる。直接テキストを投げ込むとテキストのあるフォルダにできる。
	fclose(out);


	for (int i = 0; i < params.voxels_z; i++) {
		delete[] input_volume[i];
	}
	delete[] input_volume;
	

	return 0;
}