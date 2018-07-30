#pragma once

#include<string>

struct Params
{
	char in_path_name[1000],out_name[1000];//入力ファイルと出力ファイル名
	
	int in_offset;//入力ファイルのオフセット。バイト

	float source_object_distance;//物体と線源距離 使わない
	float source_detector_distance;//物体と検出器距離　使わない

	int voxels_x, voxels_y, voxels_z;//入力ボリュームのボクセルサイズ

	float projection_phi, projection_theta;//投影方向ベクトル
	
	//ここまでがテキストファイルによる入力
	//int step;//レイマーチングのステップ長。ボクセル長さを1とする相対値
	//int pixels_x, pixels_y;//出力画像のピクセル幅と高さ　使わない
	//float pixel_pitch_x, pixel_pitch_y;//出力ピクセルサイズ、ピクセルピッチ、１ピクセルの距離換算。基本はx,yで変わらない　使わない

	int projection_div_phi, projection_div_theta;//縦と横方向の分割総数それぞれ

	float projection_radius; //投影像の一片の長さ内部で計算する
	int projection_sides;//投影像の一片の長さを整数に丸めたもの　内部で計算する
	
	//再構成範囲
	//光軸中心の横ずれ、縦ずれ
};
