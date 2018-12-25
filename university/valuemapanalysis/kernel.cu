
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

	int in_offset = 0;//�I�t�Z�b�g�������
	int valuemap_phi_div, valuemap_theta_div;//valuemap�̕������B���_���O�Ȃ̂ŁAphi�͖{����+1�ƂȂ��Ă���

	FILE *in_valuemap_file;//���̓t�@�C���p�̃t�@�C���e��

	printf("enter value map size in form of phi theta\n");//valuemap�̏c�Ɖ��̒�������
	scanf_s("%d %d", &valuemap_phi_div, &valuemap_theta_div);

	printf("enter offset if there is\n");//�I�t�Z�b�g������Γ��́B
	scanf_s("%d", &in_offset);

	int limit = 10;
	printf("enter limit max 10\n");
	scanf_s("%d", &limit);
	float ban = 0.1;
	printf("enter threshold default 0.1\n");
	scanf_s("%f", &ban);

	if ((in_valuemap_file = fopen(argv[1], "rb")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};
	printf("file_open_success\n");

	float *valuemap = new float[valuemap_phi_div*valuemap_theta_div];
	float *trajectory_index = new float[valuemap_phi_div*valuemap_theta_div];//�]���O�����Ƃɕ]���l�����[����B�]���O�����̐��͓��e�p�x�̐��ƂƂ肠������v������B

	for (i = 0; i < valuemap_phi_div*valuemap_theta_div; i++) {
		
		valuemap[i] = 0;
		trajectory_index[i] = 0;
	}
	printf("memory_success\n");

	fseek(in_valuemap_file, in_offset, SEEK_SET);//�ǂݍ��ݎ��ɃI�t�Z�b�g�ړ�

	fread(valuemap, sizeof(float), valuemap_phi_div*valuemap_theta_div, in_valuemap_file);//�{�����[���ǂݍ���
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
	float trajectory_value = 0;

	//�e�O�����Ƃɕ]���l���v�Z���Aindex�Ɏ��[�B�]�����s���O���̐��͓��e���̐��ƈ�v�B
	for (i = 0; i < valuemap_phi_div; i++) {

		axis_phi = i*PI / valuemap_phi_div;

		for (j = 0; j < valuemap_theta_div; j++) {

			axis_theta = j*PI / valuemap_theta_div;
			trajectory_value = 0;

			for (t = 0; t < PI; t += 0.05) {//����Ŕ���]
				v_phi_y = acos(-sin(axis_phi)*cos(t));
				v_theta_x = PI/2 + atan((-sin(axis_theta)*cos(axis_phi)*cos(t) + cos(axis_theta)*sin(t)) / (cos(axis_theta)*cos(axis_phi)*cos(t) - sin(axis_theta)*sin(t)));
				y = (int)(0.5+(valuemap_phi_div*v_phi_y / PI));
				x = (int)(0.5+(valuemap_theta_div*v_theta_x / PI));
				trajectory_value += valuemap[y*valuemap_theta_div + x];
			}
			printf("trajectory value %f phi %f theta %f\n", trajectory_value, axis_phi, axis_theta);
			trajectory_index[valuemap_theta_div*i + j] = trajectory_value;


		}
	}//����Ŋe�O�����Ƃ̋P�x�l�̍��v�����[���ꂽ�B


	//�O���̒��ł悳���Ȃ��̂��������I�ԁB�ō��̂��̂��܂��I��ł��炻�̎��͊p�x�łȂ��O�������ɑI��ł����B�J��Ԃ����͂Ƃ肠�����S�D
	float trajectory_lanking[10][3];

	float axis_phi_max = 0;//�悢�O���̈ꎞ�ۊǌ�
	float axis_theta_max = 0;//�悢�O���̈ꎞ�ۊǌ�
	float trajectory_max = 0;//��Ԉ����O��
	for (k = 0; k < valuemap_theta_div*valuemap_phi_div;k++) {
		if (trajectory_max < trajectory_index[k]) {
			trajectory_max = trajectory_index[k];
		}
	}
	for (k = 0; k < 10; k++) {//�����L���O������
		trajectory_lanking[k][0] = trajectory_max;
		trajectory_lanking[k][1] = 0;
		trajectory_lanking[k][2] = 0;
	}

	int test = 0;

	for (k = 0; k < limit; k++) {
		trajectory_value =  trajectory_max;

		for (i = 0; i < valuemap_phi_div; i++) {
			axis_phi = i*PI / valuemap_phi_div;
			for (j = 0; j < valuemap_theta_div; j++) {
				axis_theta = j*PI / valuemap_theta_div;
				test = 0;
				//���؂���O���������̋O���Ƃ��Ԃ��Ă��Ȃ����̌�����s��
				for (l = 0; l < limit; l++) {
					if ((axis_phi >(trajectory_lanking[l][1] - ban) && axis_phi < (trajectory_lanking[l][1] + ban)) || (axis_theta >(trajectory_lanking[l][2] - ban) && axis_theta < (trajectory_lanking[l][2] + ban))) {
						test = 1;
					}
				}

				if (test < 0.5) {
					if (trajectory_value > trajectory_index[i*valuemap_theta_div + j]) {
						trajectory_value = trajectory_index[i*valuemap_theta_div + j];
						axis_phi_max = axis_phi;
						axis_theta_max = axis_theta;
					}
				}
			}
		}

		trajectory_lanking[k][0] = trajectory_value;
		trajectory_lanking[k][1] = axis_phi_max;
		trajectory_lanking[k][2] = axis_theta_max;
		printf("value_%d_%f axis phi_%f theta_%f\n", k,trajectory_lanking[k][0], trajectory_lanking[k][1], trajectory_lanking[k][2]);

	}




	printf("Writing trajectory\n");
	for (k = 0; k < limit; k++) {
		for (t = 0; t < PI; t += 0.05) {
			v_phi_y = acos(-sin(trajectory_lanking[k][1])*cos(t));
			v_theta_x = PI/2 + atan((-sin(trajectory_lanking[k][2])*cos(trajectory_lanking[k][1])*cos(t) + cos(trajectory_lanking[k][2])*sin(t)) / (cos(trajectory_lanking[k][2])*cos(trajectory_lanking[k][1])*cos(t) - sin(trajectory_lanking[k][2])*sin(t)));
			y = (int)(0.5+(valuemap_phi_div*v_phi_y / PI));
			x = (int)(0.5+(valuemap_theta_div*v_theta_x / PI));

			valuemap[y*valuemap_theta_div + x] = k*100;
	
		}
	}



	char name[1000];
	sprintf(name, "%s-trajectory-double-(axis_phi_%f_theta_%f_pdiv_%d_tdiv_%d).raw", argv[1],axis_phi, axis_theta, valuemap_phi_div, valuemap_theta_div);

	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(valuemap, sizeof(float), valuemap_phi_div*valuemap_theta_div, out);
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
