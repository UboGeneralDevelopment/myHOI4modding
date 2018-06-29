
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

#include <stdio.h>
#include<string>
#include<fstream>
#include<math.h>


int main(int argc, char** argv) {//argc�Ƃ��ɂ͋N�����ɓn���ϐ����͂���B

	int i, j, k;//�֗��ɃJ�E���g�ȂǂɎg������
	Params params;//�p�����[�^�\����
	FILE *in, *out, *para;//���̓t�@�C���p�̃t�@�C���e��
						  //���̓t�@�C���Ɠ����ϐ��Ԃ̐����̂����̈ꎞ�I�ȃo�b�t�@

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	fscanf(para, "%s", params.in_path_name);//�ݒ�t�@�C������p�����[�^�ǂݍ��݁B�X�y�[�X���󂯂�Ǝ��ɍs���̂Ńt�@�C�����ɃX�y�[�X�͂��Ȃ��悤�ɁB
	fscanf(para, "%s", params.out_name);
	fscanf(para, "%d", &params.in_offset);
	fscanf(para, "%f %f", &params.source_object_distance, &params.source_detector_distance);
	fscanf(para, "%d %d %d", &params.voxels_x, &params.voxels_y, &params.voxels_z);
	fscanf(para, "%d %d", &params.pixels_x, &params.pixels_y);
	fscanf(para, "%f %f", &params.pixel_pitch_x, &params.pixel_pitch_y);
	fclose(para);

	printf("input projection angle in form of phi theta\n");//�����̓��e�p�x�ݒ�
	scanf_s("%f %f",&params.projection_phi,&params.projection_theta);

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n pixels %d %d\n angle %f %f\n", 
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.pixels_x,params.pixels_y,
		params.projection_phi,params.projection_theta);//�p�����[�^�����o���B

	int volume_size = params.voxels_x*params.voxels_y*params.voxels_z;
	printf("total voxels %d\n", volume_size);//����܂肢��Ȃ����ǂ��������{�N�Z������

	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//�R�}���h���C������̎��s�ł̓t���̃p�X���w�肵�Ȃ���Ό�����Ȃ��B���邢�͐ݒ�e�L�X�g�𒼐ڃv���O�����ɕ��蓊���Ă��悢�B
	}
	printf("load_success\n");

	unsigned short **input_volume = new unsigned short*[params.voxels_z];
	for (int i = 0; i < params.voxels_z; i++) { 
		input_volume[i] = new unsigned short[params.voxels_x*params.voxels_y];
	}//���ʃT�C�Y�~���݂�2�����z����m�ہB3�����z��ɓǂݍ������Ƃ���₽�玞�Ԃ�����̂�2������
	printf("memory_success\n");

	unsigned short *buff = new unsigned short[params.voxels_y*params.voxels_x];
	fseek(in, params.in_offset,SEEK_SET);
	for (i = 0; i<params.voxels_z; i++) {
		fread(buff, 2, params.voxels_x*params.voxels_y, in);
		for (j = 0; j<params.voxels_y*params.voxels_x; j++) {	
				input_volume[i][j] = buff[j];
		}
		printf("loading%d\n",i);
	}//input_volume�ւ̃t�@�C���̓ǂݍ��݁B���ʂ��Ƃ�buff�ɂ����������Ă���input_volume�֎ʂ��B
	delete[] buff;
	fclose(in);

	unsigned short *proj1 = new unsigned short[params.pixels_x*params.pixels_y];//proj1�ɓ��e����
	int x, y, z;




	/*
	�����e�̊J�����j
	���e���͈ꖇ�̂ݍ�邪�A�S�������猩���B
	�����ƃf�B�e�N�^�����̎w�肪�߂�ǂ��̂ŁA���t���e�����B
	�܂��A�g���O���t�����ƂɁA�g���O���t�{�b�N�X�̒��S�_�𒆐S�Ƃ���F�O���t���ݍ��ޑ傫���̋��`�̓��e�͈͂��w�肷��B
	���̂̒��a�T�C�Y�̓��e���L�����o�X���쐬����
	���e���L�����o�X�̊e�s�N�Z���ɂԂ���ɍs��������ݒ肷��B����͌����̏o�����W�݂̂ł悢�B
	�X�e�b�v�T�C�Y�����肷��B�{�N�Z���T�C�Y�̔����Ƃ��Ă����B�{�N�Z���T�C�Y�͂킩��Ȃ����߁A����v�Z����K�v������B�Ƃ肠�����P�Ƃł������Ă����B
	
	����������GPU
	�w�肳�ꂽ�p�x�ɑ΂��āA�g���O���t���S�𒆐S�Ƃ�����W����ł̌��_�̏o���_�����߂�B
	�X�e�b�v���ƂɁAxyz�����ɂǂꂾ���i�ނ����{�N�Z���������Z�ł���킷�B
	�_�̏o���_����X�e�b�v���Ƃɓ_�̍��W���{�N�Z�����W�Ő����ŏo���A���̈ʒu�ł̃g���O���t�̒l�𒲂ׂ郌�C�}�[�`���O���s���B
	�������A�g���O���t�{�b�N�X�̑��݂�����W�ł̂݁A�l���擾���A����̒l�����v���ē��e���L�����o�X�ɂԂ����ށB
	�ȏ�
	*/



	printf("Writing\n");//���������͏����o���B�����o���t�@�C���Ɍ`���̎w��Ȃǂ͂Ȃ��B
	char name[1000];
	sprintf(name, "%s-uint16.raw",params.out_name);
	out = fopen(name, "wb");//out��name�Ƃ��ĂЂ炭�B
	if (out == NULL) {          // �I�[�v���Ɏ��s�����ꍇ
		printf("cannot open\n");         // �G���[���b�Z�[�W���o����
		exit(0);                         // �ُ�I��
	}

	for (i = 0; i < params.voxels_z; i++) {//�����ł�out�ɒ���input���o�͂���t�@�C���ƂȂ��Ă���B�o�͓��e�ɉ����Ă��̕����͏���������B
		fwrite(input_volume[i], 2, params.voxels_x*params.voxels_y, out);
	}//�R�}���h���C������̎��s�ł͏o�̓t�@�C���̓v���O�����̃t�H���_���ɂł���B���ڃe�L�X�g�𓊂����ނƃe�L�X�g�̂���t�H���_�ɂł���B
	fclose(out);


	for (int i = 0; i < params.voxels_z; i++) {
		delete[] input_volume[i];
	}
	delete[] input_volume;
	

	return 0;
}