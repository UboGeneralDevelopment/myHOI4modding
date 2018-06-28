#pragma once

#include<string>

struct Params
{
	char in_path_name[1000],out_name[1000];//���̓t�@�C���Əo�̓t�@�C����
	
	int in_offset;//���̓t�@�C���̃I�t�Z�b�g�B�o�C�g

	float source_object_distance;//���̂Ɛ�������
	float source_detector_distance;//���̂ƌ��o�틗��

	int voxels_x, voxels_y, voxels_z;//���̓{�����[���̃{�N�Z���T�C�Y
	
	int pixels_x, pixels_y;//�o�͉摜�̃s�N�Z�����ƍ���
	float pixel_pitch_x, pixel_pitch_y;//�o�̓s�N�Z���T�C�Y�A�s�N�Z���s�b�`�A�P�s�N�Z���̋������Z�B��{��x,y�ŕς��Ȃ�
	
	float projection_phi, projection_theta;//�ŏ��̓��e����
	

	//�č\���͈�
	//�������S�̉�����A�c����
};
