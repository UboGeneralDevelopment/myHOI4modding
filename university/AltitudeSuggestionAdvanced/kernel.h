#pragma once

#include<string>

struct Params
{
	char in_path_name[1000],out_name[1000];//���̓t�@�C���Əo�̓t�@�C����
	
	int in_offset;//���̓t�@�C���̃I�t�Z�b�g�B�o�C�g

	float source_object_distance;//���̂Ɛ������� �g��Ȃ�
	float source_detector_distance;//���̂ƌ��o�틗���@�g��Ȃ�

	int voxels_x, voxels_y, voxels_z;//���̓{�����[���̃{�N�Z���T�C�Y

	float projection_phi, projection_theta;//���e�����x�N�g��
	
	//�����܂ł��e�L�X�g�t�@�C���ɂ�����
	//int step;//���C�}�[�`���O�̃X�e�b�v���B�{�N�Z��������1�Ƃ��鑊�Βl
	//int pixels_x, pixels_y;//�o�͉摜�̃s�N�Z�����ƍ����@�g��Ȃ�
	//float pixel_pitch_x, pixel_pitch_y;//�o�̓s�N�Z���T�C�Y�A�s�N�Z���s�b�`�A�P�s�N�Z���̋������Z�B��{��x,y�ŕς��Ȃ��@�g��Ȃ�

	int projection_div_phi, projection_div_theta;//�c�Ɖ������̕����������ꂼ��

	float projection_radius; //���e���̈�Ђ̒��������Ōv�Z����
	int projection_sides;//���e���̈�Ђ̒����𐮐��Ɋۂ߂����́@�����Ōv�Z����
	
	//�č\���͈�
	//�������S�̉�����A�c����
};
