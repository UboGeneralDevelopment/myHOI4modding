
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

#include<stdio.h>
#include<string>
#include<fstream>
#include<math.h>

#define STEP 1
#define PI 3.1415926535897932384626433832795f //�~����
#define Filedivide 1�@//�ǂݍ��݃t�@�C���̕����B���ǎg��Ȃ������B
#define ThreadsPerBlock 512 //���݂� CUDA �ł� 512 �� MAX


/*
���j

�Ƃ肠�����A�c�Ɖ��̕����������肵�ĕ����������摜�𐶐�������B���������͕��������Ƃ͈ˑ�����K�v���͂Ȃ����A�Ƃ肠�����͌��������͕��������ƈ�v������B

�܂��A����Ă���͓̂��ߗ��̍��v�ł͂Ȃ��A���̍ő�l�݂̂𗘗p�B��蓧�ߑ�"�炵��"�Ȃ�

�Ƃ肠�����A�摜�ꖇ�ɂ���������̕]���l���Z�肵�A������e���e�p�x���Z���s���č��v�������̂����̊p�x�ɂ����鍇�v�̕]���l�Ƃ��Čv�Z����B
�摜�ꖇ�̕]���l�́A�P�x�l*�s�N�Z�����̍��v�Ƃ���B�]���l���Z�肷�镔���𕪊����ĉ��ǂ����₷���悤�ɂ���B

�t�@�C���ǂݍ��ݏI�����_�ŁA���e�p�x�����Ǝ��s�p�x�񐔐ݒ���s���B
���e�L�����o�X�쐬�����C�}�[�`���O�v�Z���]���l�Z�o���������J���@�܂ł���A�̃v���Z�X�Ƃ��ă��W���[�����B

�]���l�̍Ĕ�r�������s���ŏI��]���p�x�̎Z�o�y�т��̊p�x�ɂ�����f�����e���̍쐬�B

���e�o���t�@�C���Ɠ��e���t�@�C����main���ō쐬���āA��]�ݒ肾���O���֐��ɓ�����B


����A���������̐��Y���ł��Ȃ��B�ӂ�����

*/

__global__ void forward_marching_GPU(unsigned short *d_input_volume, float *d_proj1, float *d_ray_position, float step_x, float step_y, float step_z, Params params) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int j;
	float v = 0;//�P�x�̊i�[��
	d_proj1[idx] = 0;
	//���C���ɂ��ēƗ��v�Z���Ă��邽�߁Aid�͈�ł����B
		for (j = 0; j < params.projection_radius / 2.0 / STEP; j++) {//���v���X�e�b�v�𓥂ނ���j�ŕ\���Ă���
			int x = (int)(d_ray_position[idx * 3] + j*step_x);//�����̈�X�e�b�v���ƂɁAxyz�����ɉ��{�N�Z�����i�ނ����v�Z���Aint�Ŋۂ߂��čŏI�I�ȓ_���W�𓾂Ă���B
			int y = (int)(d_ray_position[idx * 3 + 1] + j*step_y);
			int z = (int)(d_ray_position[idx * 3 + 2] + j*step_z);
			if (x > 0 && y > 0 && z > 0 && x < params.voxels_x && y < params.voxels_y && z < params.voxels_z) {
				//������̓_���A�{�����[���̒u���Ă���{�b�N�X��Ԃɂ��鎞�A
				//v += d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];//���ɂ��̍��\��ɂ���g���O���t�̃{�N�Z���̒l�����Z���Ă����B
				if (v < d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y]) {//�ő�l���Ƃ�B
					v = d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];
				}
			}
			int x2 = (int)(d_ray_position[idx * 3] - j*step_x);//�t�����ɂ��}�[�`���O����B
			int y2 = (int)(d_ray_position[idx * 3 + 1] - j*step_y);
			int z2 = (int)(d_ray_position[idx * 3 + 2] - j*step_z);
			if (x2 > 0 && y2 > 0 && z2 > 0 && x2 < params.voxels_x && y2 < params.voxels_y && z2 < params.voxels_z) {
				//������̓_���A�{�����[���̒u���Ă���{�b�N�X��Ԃɂ��鎞�A
				//v += d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];//���ɂ��̍��\�ɂ���g���O���t�̃{�N�Z���̒l�����Z���Ă����B
				if (v < d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y]) {//�ő�l���Ƃ�B
					v = d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];
				}
			}
		}
		d_proj1[idx] = v;
}


void ray_start_setting(Params params, float *ray_position) {
	int i, j;
	//�������B���W���S��ɂ����ă��C�̃X�^�[�g�ʒu��ݒ�Bi���L�����o�X��ł�y���W,j���L�����o�X��ł�x���W�ɑΉ�
	for (i = 0; i < params.projection_sides; i++) {
		for (j = 0; j < params.projection_sides; j++) {
			ray_position[(i*params.projection_sides + j) * 3] = j - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 1] = i - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 2] = 0;
		}
	}

	//�X�^�[�g���W��x,y���ɑ΂���theta,phi������]������
	float a, b, c;
	for (i = 0; i < params.projection_sides*params.projection_sides; i++) {
		//�܂�x���ɑ΂��ĉ�]
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = a;
		ray_position[i * 3 + 1] = cos(params.projection_theta)*b;//-sin(params.projection_theta)*c
		ray_position[i * 3 + 2] = sin(params.projection_theta)*b;//+cos(params.projection_theta)*c
	 //����y���ɑ΂��ĉ�]
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = cos(params.projection_phi)*a + sin(params.projection_phi)*c;
		ray_position[i * 3 + 1] = b;
		ray_position[i * 3 + 2] = -sin(params.projection_phi)*a + cos(params.projection_phi)*c;
		//�{�����[���̒��S���W�ɕ��s�ړ�
		ray_position[i * 3] = ray_position[i * 3] + params.voxels_x / 2.0 - 0.5;
		ray_position[i * 3 + 1] = ray_position[i * 3 + 1] + params.voxels_y / 2.0 - 0.5;
		ray_position[i * 3 + 2] = ray_position[i * 3 + 2] + params.voxels_z / 2.0 - 0.5;
		//printf("ray_2_position(%f,%f,%f)\n", ray_position[i * 3], ray_position[i * 3 + 1], ray_position[i * 3 + 1]);
	}
}





int main(int argc, char** argv) {//argc�Ƃ��ɂ͋N�����ɓn���ϐ����͂���B

	int i, j, k;//�֗��ɃJ�E���g�ȂǂɎg������
	Params params;//�p�����[�^�\����
	FILE *in, *para;//���̓t�@�C���p�̃t�@�C���e��
						  //���̓t�@�C���Ɠ����ϐ��Ԃ̐����̂����̈ꎞ�I�ȃo�b�t�@

	//////////////////////�p�����[�^�ǂݍ���//////////////////////////

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};
	
	//�ݒ�t�@�C������p�����[�^�ǂݍ��݁B�X�y�[�X���󂯂�Ǝ��ɍs���̂Ńt�@�C�����ɃX�y�[�X�͂��Ȃ��悤�ɁB
	fscanf(para, "%s", params.in_path_name);
	fscanf(para, "%s", params.out_name);
	fscanf(para, "%d", &params.in_offset);
	fscanf(para, "%f %f", &params.source_object_distance, &params.source_detector_distance);//�����g��Ȃ�
	fscanf(para, "%d %d %d", &params.voxels_x, &params.voxels_y, &params.voxels_z);
	fscanf(para, "%f %f", &params.projection_phi, &params.projection_theta);//���܂����Ȃ�
	fclose(para);

	printf("projection angle division number in form of phi theta\n");//���e�p�x�̐��̐ݒ�
	scanf_s("%d %d", &params.projection_div_phi, &params.projection_div_theta);

	params.projection_radius = sqrt(params.voxels_x * params.voxels_x + params.voxels_y *params.voxels_y + params.voxels_z * params.voxels_z);//���e���̈�Ђ̒���
	params.projection_sides = (int)params.projection_radius;//�{�N�Z�������A�܂萮���Ɋۂ߂��ꍇ�B

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n angle division phi %f theta %f\n projection radius %f/n projection sides %d",
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.projection_div_phi, params.projection_div_theta,
		params.projection_radius,params.projection_sides);//�p�����[�^�����o���B
	//////////////////////�p�����[�^�ǂݍ��ݏI��//////////////////////////

	//////////////////////�t�@�C���ǂݍ���//////////////////////////
	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//�R�}���h���C������̎��s�ł̓t���̃p�X���w�肵�Ȃ���Ό�����Ȃ��B������\��2�񂩂��B���邢�͐ݒ�e�L�X�g�𒼐ڃv���O�����ɕ��蓊���Ă��悢�B
	}
	printf("load_success\n");
	
	//���ʃT�C�Y�~���݂�1�����z��쐬
	unsigned short *input_volume = new unsigned short[params.voxels_y*params.voxels_x*params.voxels_z];
	printf("memory_success\n");

	fseek(in, params.in_offset, SEEK_SET);//�ǂݍ��ݎ��ɃI�t�Z�b�g�ړ�
	fread(input_volume, 2, params.voxels_x*params.voxels_y*params.voxels_z, in);//�ǂݍ��݁`�`
	fclose(in);
	printf("loading_success\n");
	//////////////////////�t�@�C���ǂݍ��ݏI��//////////////////////////
	
	float ray_step[3];//���C�̊i�[�e��쐬
	char name[1000];

	//////////////////////�J��Ԃ��Ώ�////////////////////////
	//////////////////////���e�����̐ݒ�////////////////////////

	for (i = 0; i < params.projection_div_phi ; i++ ) {
		params.projection_phi = PI*(1.0f*i / params.projection_div_phi);
		//for (j = 0; j < params.projection_div_theta; j++) {
			//params.projection_theta = PI*(1.0f*j / params.projection_div_theta);
			//�����œ��e�L�����o�X�쐬�AGPU�ł̓��e���쐬�A�]���l�̌v�Z�܂Ōv�Z���J��Ԃ����B

	//�����̕����̌J��Ԃ��ݒ���������Ȃ���΂��Ȃ���Ύ����I�ɏ����ݒ�ꖇ�݂̂��͂���B

	//////////////////////���e�����̐ݒ�I��////////////////////////


	//////////////////////���e�L�����o�X�������A���e�X�^�[�g�ʒu�A���C�ݒ�L�����o�X�쐬//////////////////////////
	printf("projection_setting_start\n");


	float *proj1 = new float[params.projection_sides*params.projection_sides];//���e���̃L�����o�X�쐬

	//���e���C�̃X�^�[�g�ʒu���i�[����e����쐬�Bxyz���W�Őݒ�
	float *ray_position = new float[params.projection_sides*params.projection_sides * 3];//�L�����o�X�T�C�Y��3�{
	//��������A�X�^�[�g�ʒu�A���C�A�p�����[�^�𓊂�����ŊJ�X�����X�^�[�g�ʒu�ƃ��C���󂯎��֐������B
	
	ray_start_setting(params, ray_position);
	
	ray_step[0] = { cos(params.projection_theta)*sin(params.projection_phi)*STEP};
	ray_step[1] = { -sin(params.projection_theta)*STEP};
	ray_step[2] = { cos(params.projection_theta)*cos(params.projection_phi)*STEP };
	//�����Ń��C�̕���������B���łɁA�{�N�Z�����������̂܂܍��W�n�̒����ɑΉ����Ă��邽�߁A���̂܂܃X�e�b�v�Ƃ��ėp�����B
	
	printf("ray,ray_canvas,ray_start_positions are successfully_created\n");
	//////////////////////���e�L�����o�X�쐬�I��//////////////////////////


	//////////////////GPU�]������у��C�}�[�`���O�v�Z///////////////////////

	float* d_proj1;
	float* d_ray_position;
	unsigned short* d_input_volume;
	//GPU�������m��
	cudaMalloc(&d_proj1, sizeof(float)*params.projection_sides*params.projection_sides);
	cudaMalloc(&d_ray_position,sizeof(float)*params.projection_sides*params.projection_sides * 3);
	cudaMalloc(&d_input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z);	
	//GPU�Ƀf�[�^�]��
	cudaMemcpy(d_proj1, proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ray_position, ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);
	
	printf("ray_marching_start\n");
	
	//�u���b�N���ݒ��GPU�v�Z�Ăяo��
	int BlockNum = (params.projection_sides*params.projection_sides + ThreadsPerBlock - 1) / ThreadsPerBlock;
	forward_marching_GPU <<< BlockNum, ThreadsPerBlock >>> ( d_input_volume,  d_proj1, d_ray_position, ray_step[0], ray_step[1], ray_step[2], params);

	//���e���̉����CUDA�������J��
	cudaMemcpy(proj1, d_proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyDeviceToHost);

	cudaFree(d_input_volume);
	cudaFree(d_proj1);
	cudaFree(d_ray_position);
	printf("ray_marching_end\n");

	//////////////////////GPU�]�������//////////////////////////


	//////////////////////�]���l�̌v�Z////////////////////

	//////////////////////�]���l�̌v�Z�I��/////////////////////////


	//////////////////////�����o��(�I�v�V����)//////////////////////////

	printf("Writing\n");//���������͏����o���B�����o���t�@�C���Ɍ`���̎w��Ȃǂ͂Ȃ��B

	sprintf(name, "%s-float-%dx%d-(%f-%f-%f).raw", params.out_name, params.projection_sides, params.projection_sides,ray_step[0], ray_step[1], ray_step[2]);
	printf("%s", name);
	
	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	
	//out�ɓ��e�L�����o�X���Ԃ����ށB�Ƃ肠�����̓t���[�g��
	fwrite(proj1, sizeof(float), params.projection_sides*params.projection_sides, out);
	
	fclose(out);
	printf("\nwriting_end\n");

//�R�}���h���C������̎��s�ł͏o�̓t�@�C���̓v���O�����̃t�H���_���ɂł���B���ڃe�L�X�g�𓊂����ނƃe�L�X�g�̂���t�H���_�ɂł���B

	delete[] proj1;
	delete[] ray_position;	

		//}
	}
	//////////////////////�����o���I��//////////////////////////
	//////////////////////�J��Ԃ��ΏۏI��////////////////////////


	//////////////////////�]���l�̔�r����////////////////////

	//////////////////////�]���l�̔�r�����I��////////////////




	delete[] input_volume;




	printf("program_end\n");

	return 0;
}