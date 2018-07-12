
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

#include<stdio.h>
#include<string>
#include<fstream>
#include<math.h>

#define STEP 1
#define Filedivide 1
#define ThreadsPerBlock 512 //���݂� CUDA �ł� 512 �� MAX


//��GB�͊m�ۉ\�炵�����Afloat�łԂ�����ł�̂ł��ӂ�Ă�\���͂���B�Ƃ肠�����͂��̂܂܍��B

__global__ void forward_marching_GPU(unsigned short *d_input_volume, float *d_proj1, float *d_ray_position, float step_x, float step_y, float step_z, Params params) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int j;
	float v = 0;//�P�x�̊i�[��
	//���C���ɂ��ēƗ��v�Z���Ă��邽�߁Aid�͈�ł����B
		for (j = 0; j < params.projection_radius / 2.0 / STEP; j++) {//���v���X�e�b�v�𓥂ނ���j�ŕ\���Ă���
			int x = (int)(d_ray_position[idx * 3] + j*step_x);//�����̈�X�e�b�v���ƂɁAxyz�����ɉ��{�N�Z�����i�ނ����v�Z���Aint�Ŋۂ߂��čŏI�I�ȓ_���W�𓾂Ă���B
			int y = (int)(d_ray_position[idx * 3 + 1] + j*step_y);
			int z = (int)(d_ray_position[idx * 3 + 2] + j*step_z);
			if (x > 0 && y > 0 && z > 0 && x < params.voxels_x && y < params.voxels_y && z < params.voxels_z) {
				//������̓_���A�{�����[���̒u���Ă���{�b�N�X��Ԃɂ��鎞�A
				v += d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];//���ɂ��̍��\��ɂ���g���O���t�̃{�N�Z���̒l�����Z���Ă����B
			}
			int x2 = (int)(d_ray_position[idx * 3] - j*step_x);//�t�����ɂ��}�[�`���O����B
			int y2 = (int)(d_ray_position[idx * 3 + 1] - j*step_y);
			int z2 = (int)(d_ray_position[idx * 3 + 2] - j*step_z);
			if (x2 > 0 && y2 > 0 && z2 > 0 && x2 < params.voxels_x && y2 < params.voxels_y && z2 < params.voxels_z) {
				//������̓_���A�{�����[���̒u���Ă���{�b�N�X��Ԃɂ��鎞�A
				v += d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];//���ɂ��̍��\�ɂ���g���O���t�̃{�N�Z���̒l�����Z���Ă����B
			}
		}
		d_proj1[idx] = v;
}


int main(int argc, char** argv) {//argc�Ƃ��ɂ͋N�����ɓn���ϐ����͂���B

	int i, j, k;//�֗��ɃJ�E���g�ȂǂɎg������
	Params params;//�p�����[�^�\����
	FILE *in, *para;//���̓t�@�C���p�̃t�@�C���e��
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
	scanf_s("%f %f", &params.projection_phi, &params.projection_theta);

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n pixels %d %d\n angle %f %f\n",
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.pixels_x, params.pixels_y,
		params.projection_phi, params.projection_theta);//�p�����[�^�����o���B



	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//�R�}���h���C������̎��s�ł̓t���̃p�X���w�肵�Ȃ���Ό�����Ȃ��B������\��2�񂩂��B���邢�͐ݒ�e�L�X�g�𒼐ڃv���O�����ɕ��蓊���Ă��悢�B
	}
	printf("load_success\n");
	
	//���ʃT�C�Y�~���݂�1�����z��쐬
	unsigned short *input_volume = new unsigned short[params.voxels_y*params.voxels_x*params.voxels_z];
	printf("memory_success\n");

	fseek(in, params.in_offset, SEEK_SET);
	fread(input_volume, 2, params.voxels_x*params.voxels_y*params.voxels_z, in);
	fclose(in);
	printf("loading_success\n");


	//���e���̃L�����o�X�쐬
	params.projection_radius = params.voxels_x * params.voxels_x + params.voxels_y *params.voxels_y + params.voxels_z * params.voxels_z;
	params.projection_radius = sqrt(params.projection_radius);//���e���̈�Ђ̒���
	params.projection_sides = (int)params.projection_radius;//�{�N�Z�������A�܂萮���Ɋۂ߂��ꍇ�B

	float *proj1 = new float[params.projection_sides*params.projection_sides];
	printf("projection_canvas_successfully_created\n");

	//���e���C�̃X�^�[�g�ʒu���i�[����e����쐬�Bxyz���W�Őݒ�
	float *ray_position = new float[params.projection_sides*params.projection_sides * 3];//�L�����o�X�T�C�Y��3�{

	printf("ray_successfully_created\n");

	//���W��ł̃��C�̃X�^�[�g�ʒu��ݒ�Bi���L�����o�X��ł�y���W,j���L�����o�X��ł�x���W�ɑΉ�
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
		ray_position[i * 3] = cos(params.projection_phi)*a+sin(params.projection_phi)*c;
		ray_position[i * 3 + 1] = b;
		ray_position[i * 3 + 2] = -sin(params.projection_phi)*a + cos(params.projection_phi)*c;
		//�{�����[���̒��S���W�ɕ��s�ړ�
		ray_position[i * 3] = ray_position[i * 3] + params.voxels_x / 2.0 - 0.5;
		ray_position[i * 3 + 1] = ray_position[i * 3 + 1] + params.voxels_y / 2.0 - 0.5;
		ray_position[i * 3 + 2] = ray_position[i * 3 + 2] + params.voxels_z / 2.0 - 0.5;
		//printf("ray_2_position(%f,%f,%f)\n", ray_position[i * 3], ray_position[i * 3 + 1], ray_position[i * 3 + 1]);
	}
	
	float ray_step[3] = { cos(params.projection_theta)*sin(params.projection_phi)*STEP ,-sin(params.projection_theta)*STEP ,cos(params.projection_theta)*cos(params.projection_phi)*STEP };
	//�����Ń��C�̕���������B���łɁA�{�N�Z�����������̂܂܍��W�n�̒����ɑΉ����Ă��邽�߁A���̂܂܃X�e�b�v�Ƃ��ėp�����B

	printf("projection_sides_%d ray_marching_sum_%d\n", params.projection_sides,params.projection_sides*params.projection_sides);



	//��������GPU�ɓ]�����Čv�Z���Ă����B

	float* d_proj1;
	float* d_ray_position;
	unsigned short* d_input_volume;

	cudaMalloc(&d_proj1, sizeof(float)*params.projection_sides*params.projection_sides);
	cudaMalloc(&d_ray_position,sizeof(float)*params.projection_sides*params.projection_sides * 3);
	cudaMalloc(&d_input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z);	

	cudaMemcpy(d_proj1, proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ray_position, ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);
	
	printf("ray_marching_start\n");
	
	//�u���b�N���ݒ��GPU�Ăяo��
	int BlockNum = (params.projection_sides*params.projection_sides + ThreadsPerBlock - 1) / ThreadsPerBlock;
	forward_marching_GPU <<< BlockNum, ThreadsPerBlock >>> ( d_input_volume,  d_proj1, d_ray_position, ray_step[0], ray_step[1], ray_step[2], params);

	//���e���̉����CUDA�������J��
	cudaMemcpy(proj1, d_proj1, sizeof(float)*params.projection_sides*params.projection_sides, cudaMemcpyDeviceToHost);

	cudaFree(d_input_volume);
	cudaFree(d_proj1);
	cudaFree(d_ray_position);



	printf("ray_marching_end\n");

	printf("Writing\n");//���������͏����o���B�����o���t�@�C���Ɍ`���̎w��Ȃǂ͂Ȃ��B
	char name[1000];
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
	

	/*
	�����e�̉��Ǖ��j
	�S�������猩���B�����ƃf�B�e�N�^�����̎w�肪�߂�ǂ��̂ŁA���s���e�����B
	��{�I�ɁA���̓g���O���t��1�{�N�Z����P�ʋ����Ƃ��A�g���O���t�̂͂��̃{�N�Z�������W���S�Ƃ�����W�ő�����s���B
	(0,0,0)�Ƀ{�N�Z���̒[������B((�ӂ̒���-1)/2)�Ń{�b�N�X�^�񒆍��W�ƂȂ�B���W�̓t���[�g
	���e�̊p�x�ݒ�͈ܓx�ƌo�x��p����Bz���W�����ɊJ�X��Axy���ʏ�̂��鎲��ŉ�]������B
	���e�p�x�̎w��ɂ��ƂÂ��A�L�����o�X��̊e�����̏o�����W�ʒu��ϊ�����B����̓t���[�g�ł悢�B
	�����̃X�e�b�v�T�C�Y�ƃ}�[�`���O���������肷��B�X�e�b�v�����͕ς����邪�Ƃ肠�����{�N�Z���T�C�Y�A�����͋��̂̒��a�B
	
	��������GPU�ōs�������B
	�{�����[���Ɠ��e���A���e�o���_�̃������𓮓I��CUDA�Ɋm�ۂ��A�R�s�[�A���Ƃ̓��C�̕�����X�e�b�v�񐔁A���C�̌��ȂǕK�v�Ȑ������Ԃ�����Ōv�Z����B
	���������A���̃f�[�^�e���CUDA����R�s�[���āACUDA�̃��������J���B
	����

	�I��������o�͂���
	*/



	delete[] input_volume;
	delete[] proj1;
	delete[] ray_position;
	printf("program_end\n");

	return 0;
}