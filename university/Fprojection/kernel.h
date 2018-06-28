#pragma once

#include<string>

struct Params
{
	char in_path_name[1000],out_name[1000];//入力ファイルと出力ファイル名
	
	int in_offset;//入力ファイルのオフセット。バイト

	float source_object_distance;//物体と線源距離
	float source_detector_distance;//物体と検出器距離

	int voxels_x, voxels_y, voxels_z;//入力ボリュームのボクセルサイズ
	
	int pixels_x, pixels_y;//出力画像のピクセル幅と高さ
	float pixel_pitch_x, pixel_pitch_y;//出力ピクセルサイズ、ピクセルピッチ、１ピクセルの距離換算。基本はx,yで変わらない
	
	float projection_phi, projection_theta;//最初の投影方向
	

	//再構成範囲
	//光軸中心の横ずれ、縦ずれ
};
