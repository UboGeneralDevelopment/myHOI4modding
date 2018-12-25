
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
���j

�ʒu���킹���̂��̂ƁA�ʒu���킹�ϊ��s���VGSTUDIO�ŏo�͂���B

�ϊ����s����2�{�����[���ƁA�ϊ��s��A���̂ق��ϊ��ɕK�v�ȃf�[�^��ǂݍ���

��1�{�����[���Ɠ��T�C�Y�̏o�̓{�����[�������A���̊e�{�N�Z������2�{�����[�����őΉ�����ʒu��ϊ��s��ɏ]���Čv�Z���A��2�{�����[���̒l���Ƃ��Ă��đ�����Ă����B
��2�{�����[����ɑΉ�����l���Ȃ���΂Ƃ肠�������u�B

�M���x�����l�ɕϊ��B

�ԊҌ�̃{�����[���ƐM���x�{�����[�����o��

*/

int main(int argc, char** argv){
	int i, j, k, l;
	FILE *in_volume, *in_reliability, *para;
	//�{�����[���A�M���x�A��]�s��Ƃ����̂ق��p�����[�^
	/*
	�ݒ�t�@�C���p�����[�^
	
	�{�����[���t�@�C���̃p�X
	�M���x�t�@�C���̃p�X
	�o�̓t�@�C����
	���̓{�����[���̃{�N�Z����XYZ�A�o�̓{�����[���̃{�N�Z����XYZ
	���̓{�����[���̃{�N�Z���T�C�Y�A�o�̓{�����[���̃{�N�Z���T�C�Y
	�ϊ��s��B�{�����[���P���{�����[��2�ȍ~�Ɉʒu���킹���Ă����B

	�K�v�ȕϊ��s��́A�ʒu���킹��̃{�����[������A�ʒu���킹����{�����[���ւ̕ϊ��ŁA�ʒu���킹����{�����[������ʒu���킹��ւ̕ϊ��ł͂Ȃ��̂Œ��ӁB
	��{�I�ɍ��W�͎��T�C�Y���W�n�Ōv�Z���Ă����B���{�����[���̂��ꂱ��ʒu�����{�����[���̂��������ʒu�ɑΉ�����Ƃ��������B
	��1�{�����[���ɑ�2�{�����[�����ʒu���킹���Ă����B
	�o�͂͑��{�����[���B

	*/

	char in_path_name_v[1000], in_path_name_r[1000], out_name[1000];
	//���̓t�@�C���Əo�̓t�@�C����
	int v_in[3], v_out[3];
	//v_in����2(����)�{�����[���ŁAv_out���o��
	double v_in_size, v_out_size;
	//���̓{�N�Z�����Ɠ��̓{�N�Z���T�C�Y
	double affin[4][4];
	//���͕ϊ��s��B�t�ϊ���VGSTUDIO����e�Ղɋ��܂�B

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	fscanf(para, "%s", in_path_name_v);
	fscanf(para, "%s", in_path_name_r);
	fscanf(para, "%s", out_name);
	fscanf(para, "%d %d %d %d %d %d", &v_in[0], &v_in[1], &v_in[2], &v_out[0], &v_out[1], &v_out[2]);
	fscanf(para, "%lf %lf", &v_in_size, &v_out_size);

	fscanf(para, "%lf %lf %lf %lf", &affin[0][0], &affin[1][0], &affin[2][0], &affin[3][0]);
	fscanf(para, "%lf %lf %lf %lf", &affin[0][1], &affin[1][1], &affin[2][1], &affin[3][1]);
	fscanf(para, "%lf %lf %lf %lf", &affin[0][2], &affin[1][2], &affin[2][2], &affin[3][2]);
	fscanf(para, "%lf %lf %lf %lf", &affin[0][3], &affin[1][3], &affin[2][3], &affin[3][3]);
	
	fclose(para);

	i = 0;
	j = 0;
	k = 0;
	double tx, ty, tz;
	double cx, cy, cz;

	float temp_1_1, temp_1_2, temp_1_3, temp_1_4, temp_2_1, temp_2_2;
	float y0, y1;

	//�{�����[���̊i�[�e��@xy���ʃT�C�Yvoxels1[0] * voxels1[1]�@x���T�C�Yvoxel1[0]
	float *ct_volume = new float[v_in[0]* v_in[1]* v_in[2]];	
	//�ϊ���̃{�����[���e��
	float *transformed_volume = new float[v_out[0] * v_out[1] * v_out[2]];

	//�ϊ��O�{�����[���ǂݍ���
	if ((in_volume = fopen(in_path_name_v, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}

	printf("volume load success\n");

	fread(ct_volume, 4, v_in[0] * v_in[1] * v_in[2], in_volume);//�{�����[���ǂݍ���

	fclose(in_volume);
	
	printf("transformation starts\n");


	for (i = 0; i < v_out[2]; i++) {
		printf("progress %d / %d\n", i, v_out[2]);
		for (j = 0; j < v_out[1]; j++) {
			for (k = 0; k < v_out[0]; k++) {
				//VGstudio�ł͍��W���S�̓{�����[�����S�ɒu����A���W���̕ϊ������߂邱�Ƃ��ł���B����Ɋ�Â��e�{�N�Z���̑Ή����v�Z
				//���W�n�ɂ́A��2�{�����[�������W�n�i���_�{�����[�����S�j�A��2�{�����[���{�N�Z�����W�n�i���_�{�����[���[�_�j�A�o�̓{�����[�������W�n�A�o�̓{�����[���{�N�Z�����W�n��4������B

				//�o�̓{�����[�������W�n�ɂ�����A�e�{�N�Z�������W
				tx = v_out_size*(k + 0.5) - (v_out_size*v_out[0] / 2);
				ty = v_out_size*(j + 0.5) - (v_out_size*v_out[1] / 2);
				tz = v_out_size*(i + 0.5) - (v_out_size*v_out[2] / 2);

				//�o�̓{�����[���̃{�N�Z�����W��ϊ����A��2�{�����[�������W�n�ł̍��W���v�Z�B
				cx = affin[0][0] * tx + affin[1][0] * ty + affin[2][0] * tz + affin[3][0];
				cy = affin[0][1] * tx + affin[1][1] * ty + affin[2][1] * tz + affin[3][1];
				cz = affin[0][2] * tx + affin[1][2] * ty + affin[2][2] * tz + affin[3][2];

				//��2�{�����[�������W�n�̌��_���{�����[���̒[�_�ֈړ��B
				cx = cx + (v_in_size*v_in[0] / 2);
				cy = cy + (v_in_size*v_in[1] / 2);
				cz = cz + (v_in_size*v_in[2] / 2);

				//���W�n�̋������{�N�Z���P�ʋ����ɕϊ��A�܂����W���_��(0,0,0)�̃{�N�Z���̒��S�Ƃ����Ƃ��̍��W�����߂�B
				cx = cx / v_in_size - 0.5;
				cy = cy / v_in_size - 0.5;
				cz = cz / v_in_size - 0.5;

				if (cx > 1 && cx < (v_in[0] - 2) && cy > 1 && cy < (v_in[1] - 2) && cz > 1 && cz < (v_in[2] - 2)) {
					y0 = ct_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_1 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = ct_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_2 = y0 + (cz - (int)cz)*(y1 - y0);
					
					y0 = ct_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_3 = y0 + (cz - (int)cz)*(y1 - y0);
					
					y0 = ct_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = ct_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_4 = y0 + (cz - (int)cz)*(y1 - y0);


					temp_2_1 = temp_1_1 + (cy - (int)cy)*(temp_1_2 - temp_1_1);

					temp_2_2 = temp_1_3 + (cy - (int)cy)*(temp_1_4 - temp_1_3);


					transformed_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = temp_2_1 + (cx - (int)cx)*(temp_2_2 - temp_2_1);
				
				}
				else {
					transformed_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = 0;
				}

				/*
				cxv = (int)(0.5 + (cx / vsize1));
				cyv = (int)(0.5 + (cy / vsize1));
				czv = (int)(0.5 + (cz / vsize1));

				//printf("%d %d %d :\n", cxv, cyv, czv);
				
				if ( cxv > 0 && cxv < voxels1[0]  && cyv > 0 && cxv < voxels1[1] && czv > 0 && czv < voxels1[2]) {
					transformed_volume[voxels2[0] * voxels2[1] * i + voxels2[0] * j + k] = ct_volume[cxv + voxels1[0] * cyv + voxels1[0] * voxels1[1] * czv];
				}
				else {
					transformed_volume[voxels2[0] * voxels2[1] * i + voxels2[0] * j + k] = 0;
				}

				//printf("%d %d %d : %d\n", k, j, i, transformed_volume[voxels2[0] * voxels2[1] * i + voxels2[0] * j + k]);
				*/

			}
		}
	}

	printf("Writing transformed volume\n");
	char name[1000];
	sprintf(name, "%s-(%dx%dx%d).raw", out_name, v_out[0], v_out[1], v_out[2]);
	printf("%s", name);

	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(transformed_volume, sizeof(float), v_out[0] * v_out[1] * v_out[2], out);
	
	printf("\nWriting transformed volume end\n");
	
	fclose(out);

	delete[] transformed_volume;
	delete[] ct_volume;
	

	//�M���x�̕ϊ�

	//�{�����[���̊i�[�e��@xy���ʃT�C�Yvoxels1[0] * voxels1[1]�@x���T�C�Yvoxel1[0]
	float *r_volume = new float[v_in[0] * v_in[1] * v_in[2]];
	//�ϊ���̃{�����[���e��
	float *transformed_r_volume = new float[v_out[0] * v_out[1] * v_out[2]];

	//�ϊ��O�{�����[���ǂݍ���
	if ((in_reliability = fopen(in_path_name_r, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);
	}

	printf("reliability load success\n");

	fread(r_volume, sizeof(float), v_in[0] * v_in[1] * v_in[2], in_reliability);//�{�����[���ǂݍ���

	fclose(in_reliability);

	printf("transformation starts\n");

	for (i = 0; i < v_out[2]; i++) {
		printf("progress %d / %d\n", i, v_out[2]);
		for (j = 0; j < v_out[1]; j++) {
			for (k = 0; k < v_out[0]; k++) {

				//�o�̓{�����[�������W�n�ɂ�����A�e�{�N�Z�������W
				tx = v_out_size*(k + 0.5) - (v_out_size*v_out[0] / 2);
				ty = v_out_size*(j + 0.5) - (v_out_size*v_out[1] / 2);
				tz = v_out_size*(i + 0.5) - (v_out_size*v_out[2] / 2);

				//�o�̓{�����[���̃{�N�Z�����W��ϊ����A��2�{�����[�������W�n�ł̍��W���v�Z�B
				cx = affin[0][0] * tx + affin[1][0] * ty + affin[2][0] * tz + affin[3][0];
				cy = affin[0][1] * tx + affin[1][1] * ty + affin[2][1] * tz + affin[3][1];
				cz = affin[0][2] * tx + affin[1][2] * ty + affin[2][2] * tz + affin[3][2];

				//��2�{�����[�������W�n�̌��_���{�����[���̒[�_�ֈړ��B
				cx = cx + (v_in_size*v_in[0] / 2);
				cy = cy + (v_in_size*v_in[1] / 2);
				cz = cz + (v_in_size*v_in[2] / 2);

				//���W�n�̋������{�N�Z���P�ʋ����ɕϊ��A�܂����W���_��(0,0,0)�̃{�N�Z���̒��S�Ƃ����Ƃ��̍��W�����߂�B
				cx = cx / v_in_size - 0.5;
				cy = cy / v_in_size - 0.5;
				cz = cz / v_in_size - 0.5;

				if (cx > 1 && cx < (v_in[0] - 2) && cy > 1 && cy < (v_in[1] - 2) && cz > 1 && cz < (v_in[2] - 2)) {
					y0 = r_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)cx + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_1 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = r_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)cx + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_2 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = r_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)(cx + 1) + v_in[0] * (int)cy + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_3 = y0 + (cz - (int)cz)*(y1 - y0);

					y0 = r_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)cz];
					y1 = r_volume[(int)(cx + 1) + v_in[0] * (int)(cy + 1) + v_in[0] * v_in[1] * (int)(cz + 1)];
					temp_1_4 = y0 + (cz - (int)cz)*(y1 - y0);


					temp_2_1 = temp_1_1 + (cy - (int)cy)*(temp_1_2 - temp_1_1);

					temp_2_2 = temp_1_3 + (cy - (int)cy)*(temp_1_4 - temp_1_3);


					transformed_r_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = temp_2_1 + (cx - (int)cx)*(temp_2_2 - temp_2_1);

				}
				else {
					transformed_r_volume[v_out[0] * v_out[1] * i + v_out[0] * j + k] = 0;
				}

			}
		}
	}

	printf("Writing transformed reliability volume\n");
	
	sprintf(name, "%s-reliability-(%dx%dx%d).raw", out_name, v_out[0], v_out[1], v_out[2]);
	printf("%s", name);

	
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(transformed_r_volume, sizeof(float), v_out[0] * v_out[1] * v_out[2], out);

	printf("\nWriting transformed reliability end\n");

	fclose(out);

	delete[] transformed_r_volume;
	delete[] r_volume;
	system("pause");
	return 0;
}
