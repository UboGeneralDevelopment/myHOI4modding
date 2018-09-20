
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<string>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#define STEP 1
#define PI 3.1415926535897932384626433832795f //�~����
#define Filedivide 1�@//�ǂݍ��݃t�@�C���̕����B���ǎg��Ȃ������B
#define ThreadsPerBlock 512 //���݂� CUDA �ł� 512 �� MAX
#define angle_div 50 //�]�����̕�����

/*
���j
valuemap����荞��ŁA�����]�����邽�߂����̃v���O����
*/


int main(int argc, char** argv) {//argc�Ƃ��ɂ͋N�����ɓn���ϐ����͂���B

	int i, j, k, l;//�֗��ɃJ�E���g�ȂǂɎg������

	int in_offset;//�I�t�Z�b�g�������
	int valuemap_phi_div, valuemap_theta_div;//valuemap�̕������B���_���O�Ȃ̂ŁAphi�͖{����+1�ƂȂ��Ă���

	FILE *in_valuemap_file;//���̓t�@�C���p�̃t�@�C���e��

	printf("enter value map size in form of phi theta\n");//valuemap�̏c�Ɖ��̒�������
	scanf_s("%d %d", &valuemap_phi_div, &valuemap_theta_div);

	printf("enter offset if there is\n");//�I�t�Z�b�g������Γ��́B
	scanf_s("%d", &in_offset);

	if ((in_valuemap_file = fopen(argv[1], "rb")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};
	printf("file_open_success\n");

	double *valuemap = new double[valuemap_phi_div*valuemap_theta_div];
	double *trajectory_index = new double[valuemap_phi_div*valuemap_theta_div];//�]���O�����Ƃɕ]���l�����[����B�]���O�����̐��͓��e�p�x�ƂƂ肠������v������B
	for (i = 0; i < valuemap_phi_div*valuemap_theta_div;i++) {
		valuemap[i] = 0;
		trajectory_index[i] = 0;
	}
	printf("memory_success\n");

//	fseek(in_valuemap_file, in_offset, SEEK_SET);//�ǂݍ��ݎ��ɃI�t�Z�b�g�ړ�

	fread(valuemap, sizeof(double), valuemap_phi_div*valuemap_theta_div, in_valuemap_file);//�{�����[���ǂݍ���
	fclose(in_valuemap_file);
	printf("loading_success\n");
	/*
	for (i = 0; i < valuemap_phi_div*valuemap_theta_div; i++) {
		printf("%lf\n", valuemap[i]);
	}
	system("pause");
	*/
	printf("\n projection_trajectory_analysis_start\n");
	float v_phi_y = 0, v_theta_x = 0;//value_map��ł̊p�x���W
	int x, y;
	float axis_phi = 0, axis_theta = 0;//trajectory�̎��p�x�B
	float t = 0;//�O���̊p�x�B�J�X����B
	double trajectory_value = 0;
	int maxx, maxy;

	//�e�O�����Ƃɕ]���l���v�Z���Aindex�Ɏ��[
	for (i = 0; i < valuemap_phi_div; i++) {
		axis_theta = 0;
		axis_phi += PI / (valuemap_phi_div+1);

		for (j = 0; j < valuemap_theta_div; j++) {
			trajectory_value = 0;
			for (t = 0; t < PI; t += 0.05) {//����Ŕ���]
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
	}//����Ŋe�O�����Ƃ̋P�x�l�̍��v�����[���ꂽ�B

	trajectory_value = trajectory_index[0];


	//�O���̒��ł悳���Ȃ��̂�I��
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


	////////////////�������J��///////////////
	delete[] trajectory_index;
	delete[] valuemap;

	////////////////�������J��///////////////

	printf("program_end\n");
	system("pause");

	return 0;
}
