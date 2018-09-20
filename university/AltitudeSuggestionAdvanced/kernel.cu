
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
#define angle_div 50 //�]�����̕�����

/*
���j

�c�Ɖ��̕������ɂ��ẮA�ő啪�����ݒ��A�i�ܓx���ƂɎ�����������B���������͕��������Ƃ͕K�������ˑ����Ȃ��B
���_�ɂ�����B����]�͈Ӗ����Ȃ����߁A������x�ɑ΂��Ď��ۂɂ�x+1�������A���̍ŏ��ƍŌ�̒��_�ʒu�ł̎B�����s��Ȃ��B

����Ă���͓̂��ߗ��̍��v�ł͂Ȃ��A���̍ő�l�݂̂𗘗p�B��蓧�ߑ�"�炵��"�Ȃ�B
�ő�l�݂̂��Ƃ�̂ŁA���f����uint16,unsigne int�t�@�C���B���ɓ������̍��v���o��������Ε��ς���B

�����o�����ߗ��Ɗp�x�̃}�b�v��double�ŏo�́B

�t�@�C���ǂݍ��ݏI�����_�ŁA���e�p�x�����Ǝ��s�p�x�񐔐ݒ���s���B
���e�L�����o�X�쐬�����C�}�[�`���O�v�Z���]���l�Z�o���������J���@�܂ł���A�̃v���Z�X�Ƃ��ă��W���[�����B
���e�o���t�@�C���Ɠ��e���t�@�C����main���ō쐬���āA��]�ݒ肾���O���֐��ɓ�����B

�t���O�����g������̂ŁAmalloc�Ƃ�free��for���̊O�B
�{�����[����GPU�ւ̓ǂݍ��݂͈��̂݁B

*/

__global__ void forward_marching_GPU(unsigned short *d_input_volume, unsigned short *d_proj1, float *d_ray_position, float step_x, float step_y, float step_z, Params params) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int j;
	int x, y, z, x2, y2, z2;
	float vf = 0.0;
	unsigned short v = 0;//�P�x�̊i�[��
	d_proj1[idx] = 0;
	//���C���ɂ��ēƗ��v�Z���Ă��邽�߁Aid�͈�ł����B
		for (j = 0; j < params.projection_radius / 2.0 / STEP; j++) {//���v���X�e�b�v�𓥂ނ���j�ŕ\���Ă���
			x = (int)(d_ray_position[idx * 3] + j*step_x);//�����̈�X�e�b�v���ƂɁAxyz�����ɉ��{�N�Z�����i�ނ����v�Z���Aint�Ŋۂ߂��čŏI�I�ȓ_���W�𓾂Ă���B
			y = (int)(d_ray_position[idx * 3 + 1] + j*step_y);
			z = (int)(d_ray_position[idx * 3 + 2] + j*step_z);
			if (x > 0 && y > 0 && z > 0 && x < params.voxels_x && y < params.voxels_y && z < params.voxels_z) {
				//������̓_���A�{�����[���̒u���Ă���{�b�N�X��Ԃɂ��鎞�A
				//v += d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];//���ɂ��̍��\��ɂ���g���O���t�̃{�N�Z���̒l�����Z���Ă����B
				if (vf < d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y]) {//�ő�l���Ƃ�B
					vf = d_input_volume[x + y * params.voxels_x + z * params.voxels_x * params.voxels_y];
				}
			}
			x2 = (int)(d_ray_position[idx * 3] - j*step_x);//�t�����ɂ��}�[�`���O����B
			y2 = (int)(d_ray_position[idx * 3 + 1] - j*step_y);
			z2 = (int)(d_ray_position[idx * 3 + 2] - j*step_z);
			if (x2 > 0 && y2 > 0 && z2 > 0 && x2 < params.voxels_x && y2 < params.voxels_y && z2 < params.voxels_z) {
				//������̓_���A�{�����[���̒u���Ă���{�b�N�X��Ԃɂ��鎞�A
				//v += d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];//���ɂ��̍��\�ɂ���g���O���t�̃{�N�Z���̒l�����Z���Ă����B
				if (vf < d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y]) {//�ő�l���Ƃ�B
					vf = d_input_volume[x2 + y2 * params.voxels_x + z2 * params.voxels_x * params.voxels_y];
				}
			}
		}
		v = (int)vf;
		d_proj1[idx] = v;
}


void ray_start_setting(Params params, float *ray_position, float ray_phi, float ray_theta) {
	int i, j;
	//�������B���W���S��ɂ����ă��C�̃X�^�[�g�ʒu��ݒ�Bi���L�����o�X��ł�y���W,j���L�����o�X��ł�x���W�ɑΉ��B
	for (i = 0; i < params.projection_sides; i++) {
		for (j = 0; j < params.projection_sides; j++) {
			ray_position[(i*params.projection_sides + j) * 3] = j - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 1] = i - params.projection_radius / 2;
			ray_position[(i*params.projection_sides + j) * 3 + 2] = 0;
		}
	}

	//�X�^�[�g���W��y,z���ɑ΂���theta,phi������]�������̂��A���s�ړ�������
	float a, b, c;
	for (i = 0; i < params.projection_sides*params.projection_sides; i++) {
		//�܂�y���ɂ����ĉ�]
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = cos(ray_phi)*a + sin(ray_phi)*c;
		ray_position[i * 3 + 1] = b;
		ray_position[i * 3 + 2] = -sin(ray_phi)*a + cos(ray_phi)*c;
	 //����z���ɂ����ĉ�]
		a = ray_position[i * 3];
		b = ray_position[i * 3 + 1];
		c = ray_position[i * 3 + 2];
		ray_position[i * 3] = cos(ray_theta)*a - sin(ray_theta)*b;
		ray_position[i * 3 + 1] = sin(ray_theta)*a + cos(ray_theta)*b;
		ray_position[i * 3 + 2] = c;

		//�{�����[���̒��S���W�ɕ��s�ړ�
		ray_position[i * 3] = ray_position[i * 3] + params.voxels_x / 2.0 - 0.5;
		ray_position[i * 3 + 1] = ray_position[i * 3 + 1] + params.voxels_y / 2.0 - 0.5;
		ray_position[i * 3 + 2] = ray_position[i * 3 + 2] + params.voxels_z / 2.0 - 0.5;
		//printf("ray_2_position(%f,%f,%f)\n", ray_position[i * 3], ray_position[i * 3 + 1], ray_position[i * 3 + 1]);
	}

}



int main(int argc, char** argv) {//argc�Ƃ��ɂ͋N�����ɓn���ϐ����͂���B

	int i, j, k, l;//�֗��ɃJ�E���g�ȂǂɎg������
	Params params;//�p�����[�^�\����
	FILE *in, *para;//���̓t�@�C���p�̃t�@�C���e��
						  //���̓t�@�C���Ɠ����ϐ��Ԃ̐����̂����̈ꎞ�I�ȃo�b�t�@

	//////////////////////�p�����[�^�ǂݍ��݂ƃp�����[�^�ݒ�//////////////////////////

	if ((para = fopen(argv[1], "r")) == NULL) {
		printf("FILE do not exist\n");
		exit(0);
	};

	//�ݒ�t�@�C������p�����[�^�ǂݍ��݁B�X�y�[�X���󂯂�Ǝ��ɍs���̂Ńt�@�C�����ɃX�y�[�X�͂��Ȃ��B
	fscanf(para, "%s", params.in_path_name);
	fscanf(para, "%s", params.out_name);
	fscanf(para, "%d", &params.in_offset);
	fscanf(para, "%f %f", &params.source_object_distance, &params.source_detector_distance);//�����g��Ȃ�
	fscanf(para, "%d %d %d", &params.voxels_x, &params.voxels_y, &params.voxels_z);
	fscanf(para, "%f %f", &params.projection_phi, &params.projection_theta);//�����̏����ݒ�B���܂����Ȃ�
	fclose(para);

	params.projection_div_phi = 4;//�Ƃ肠������`�B������scan�ł�݂��܂��Ă��悢�B
	params.projection_div_theta = 4;

	printf("projection angle division number in form of phi theta\n");//���e�p�x�̖����̐ݒ�B�����œǂݍ��ޓ��e�͂悭�ύX����B
	scanf_s("%d %d", &params.projection_div_phi, &params.projection_div_theta);

	params.projection_radius = sqrt(params.voxels_x * params.voxels_x + params.voxels_y *params.voxels_y + params.voxels_z * params.voxels_z);//���e���̈�Ђ̒���
	params.projection_sides = (int)params.projection_radius;//�{�N�Z�������A�܂萮���Ɋۂ߂��ꍇ�B

	printf(" input %s\n output %s\n distance %f %f\n voxels %d %d %d\n start angle phi %f theta %f\n angle division phi %d theta %d\n projection radius %f projection sides %d",
		params.in_path_name, params.out_name,
		params.source_object_distance, params.source_detector_distance,
		params.voxels_x, params.voxels_y, params.voxels_z,
		params.projection_phi, params.projection_theta,
		params.projection_div_phi, params.projection_div_theta,
		params.projection_radius, params.projection_sides);//�p�����[�^�����o���B




	//////////////////////�ǂݍ��݂Ɗe�탁�����m��,GPU�]��//////////////////////////
	if ((in = fopen(params.in_path_name, "rb")) == NULL) {
		printf("FILE do not exist_1\n");
		exit(0);//�R�}���h���C������̎��s�ł̓t���̃p�X���w�肵�Ȃ���Ό�����Ȃ��B������\��2�񂩂��B���邢�͐ݒ�e�L�X�g�𒼐ڃv���O�����ɕ��Ă��悢�B
	}
	printf("load_success\n");

	unsigned short *input_volume = new unsigned short[params.voxels_y*params.voxels_x*params.voxels_z];
	//���ʃT�C�Y�~���݂�1�����z��쐬
	printf("memory_success\n");

	fseek(in, params.in_offset, SEEK_SET);//�ǂݍ��ݎ��ɃI�t�Z�b�g�ړ�
	fread(input_volume, 2, params.voxels_x*params.voxels_y*params.voxels_z, in);//�{�����[���ǂݍ���
	fclose(in);
	printf("loading_success\n");

	unsigned short *proj1 = new unsigned short[params.projection_sides*params.projection_sides];
	//���e���̃������m�ہBCPU��
	float *ray_position = new float[params.projection_sides*params.projection_sides * 3];
	//�L�����o�X�T�C�Y��3�{�̓��e���C�̃X�^�[�g�ʒu���i�[���郁�����BCPU���Bxyz���W�Őݒ�
	float ray_step[3];
	//���C�̊i�[�e��쐬
	float ray_phi = 0.0;
	float ray_theta = 0.0;
	//���C�̕��������ݒ�
	char name[1000];
	//�����o���p�̖��O
	double *valuemap = new double[params.projection_div_phi*params.projection_div_theta];
	for (k = 0; k < params.projection_div_phi*params.projection_div_theta; k++) {
		valuemap[k] = 0;
	}
	//�]���l�̃�����

	printf("projection_setting_start\n");
	//////////////////////�ȉ�GPU�ł̃������m��
	unsigned short* d_proj1;
	float* d_ray_position;
	unsigned short* d_input_volume;
	//GPU�������m��
	cudaMalloc(&d_proj1, sizeof(unsigned short)*params.projection_sides*params.projection_sides);
	cudaMalloc(&d_ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3);
	cudaMalloc(&d_input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z);

	//�{�����[������GPU�ւƓ]��
	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);




	/////////////////////////�J��Ԃ��ΏہB���e�����A�]�����A�]���}�b�v���쐻////////////////////////////////////

	for (i = 0; i < params.projection_div_phi; i++) {//�J��Ԃ��񐔂̐ݒ�Bphi����,theta������]�œ�d���[�v�ɂȂ��Ă���B�B
		//theta������]���̏����ݒ�Bphi�ɏ������ړ�������B
		ray_theta = 0;
		ray_phi += PI / (params.projection_div_phi + 1);

		for (j = 0; j < params.projection_div_theta; j++) {

			printf("phi(%f)_theta(%f)\n",ray_phi, ray_theta);
			//////////////////////���e�������A���e�X�^�[�g�ʒu�ݒ�A���C�����ݒ�̏�����//////////////////////////

			for (k = 0; k < params.projection_sides*params.projection_sides; k++) {
				proj1[k] = 0;
			}//���e������������

			//���C�̕���������B�p�xtheta0phi0��(0,0,1)�B�������ɐL�тĂ��B�{�N�Z�����������̂܂܍��W�n�̒����ɑΉ����Ă��邽�߁A���̂܂܃X�e�b�v�Ƃ��ėp�����B			ray_step[0] = sin(ray_phi)*cos(ray_theta)*STEP;
			ray_step[1] = sin(ray_phi)*sin(ray_theta)*STEP;
			ray_step[2] = cos(ray_phi)*STEP;

			ray_start_setting(params, ray_position, ray_phi, ray_theta);
			//�X�^�[�g�ʒu�ݒ�A�X�^�[�g�ʒu�ƁA�p�����[�^�𓊂�����ŉ�]������B



			//////////////////GPU�]������ьv�Z///////////////////////

			//GPU�ɓ��e�L�����o�X�Ɠ��e�ʒu�ݒ�f�[�^�]��
			cudaMemcpy(d_proj1, proj1, sizeof(unsigned short)*params.projection_sides*params.projection_sides, cudaMemcpyHostToDevice);
			cudaMemcpy(d_ray_position, ray_position, sizeof(float)*params.projection_sides*params.projection_sides * 3, cudaMemcpyHostToDevice);
		//	cudaMemcpy(d_input_volume, input_volume, sizeof(unsigned short)*params.voxels_x*params.voxels_y*params.voxels_z, cudaMemcpyHostToDevice);

			printf("ray_marching_start\n");

			//�u���b�N���ݒ��GPU�v�Z�Ăяo��
			int BlockNum = (params.projection_sides*params.projection_sides + ThreadsPerBlock - 1) / ThreadsPerBlock;
			forward_marching_GPU << < BlockNum, ThreadsPerBlock >> > (d_input_volume, d_proj1, d_ray_position, ray_step[0], ray_step[1], ray_step[2], params);

			//���e���̉��
			cudaMemcpy(proj1, d_proj1, sizeof(unsigned short)*params.projection_sides*params.projection_sides, cudaMemcpyDeviceToHost);

			printf("ray_marching_end\n");

	


			//////////////////////�]���l�̌v�Z////////////////////

			for (k = 0; k < params.projection_sides*params.projection_sides; k++) {
				valuemap[params.projection_div_theta*i + j] += proj1[k];
			}
			printf("value_%lf\n", valuemap[params.projection_div_theta*i + j]);
			



			//////////////////////�����o��(�I�v�V����)//////////////////////////
			if ( i % 10 == 0 && j % 10 == 0){
				printf("Writing projection\n");//���������͏����o���B�����o���t�@�C���Ɍ`���̎w��Ȃǂ͂Ȃ��B
				sprintf(name, "%s-uint16-%dx%d-(%f_%f).raw", params.out_name, params.projection_sides, params.projection_sides,ray_phi, ray_theta);
				printf("%s", name);
				FILE *out;
				out = fopen(name, "wb");
				if (out == NULL) {
					printf("\nFILE cannot open\n");
					exit(0);
				};
				fwrite(proj1, sizeof(unsigned short), params.projection_sides*params.projection_sides, out);
				fclose(out);
				printf("\nwriting_end\n\n");
			}
			//�R�}���h���C������̎��s�ł͏o�̓t�@�C���̓v���O�����̃t�H���_���ɂł���B���ڃe�L�X�g�𓊂����ނƃe�L�X�g�̂���t�H���_�ɂł���B
			



			//�p�x��theta�����ɍX�V
			ray_theta += PI / (params.projection_div_theta + 1);
		}
	}



	////////////////�������J��///////////////
	cudaFree(d_input_volume);
	cudaFree(d_proj1);
	cudaFree(d_ray_position);	
	delete[] proj1;
	delete[] ray_position;	
	delete[] input_volume;
	////////////////�������J��///////////////

	//////////////////////�]���l�̔�r����////////////////////
	/*
	float vphi, vtheta;
	float *value = new float[angle_div*angle_div];
	float max_value[3];
	for (i = 0; i < params.projection_div_theta*params.projection_div_phi; i++) {
		printf("value %f (phi %f theta %f)\n", temp[3*i], temp[3*i + 1], temp[3*i + 2]);

		for (j = 0; j < angle_div; j++) {
			for (k = 0; k < angle_div; k++) {
				value[j*angle_div + k] = 0;
			}
		}
		for (j = 0; j < angle_div; j++) {
			vphi = PI*2.0f*j/angle_div;
			for (k = 0; k < angle_div; k++) {
				vtheta = PI*2.0f*k / angle_div;
				for (l = 0; l < params.projection_div_phi * params.projection_div_theta ; l++) {
					value[j*angle_div + k] += temp[3*i]*(sin(temp[3 * l + 1])*cos(temp[3 * l + 2])*sin(vphi)*cos(vtheta) + sin(temp[3 * l + 1])*sin(temp[3 * l + 2])*sin(vphi)*sin(vtheta) + cos(temp[3 * l + 1])*cos(vphi));
				}
				//printf("value %f phi %f theta %f\n", value[j*angle_div + k], vphi, vtheta);
				if (max_value[0]<value[j*angle_div + k]) {
					max_value[0] = value[j*angle_div + k];
					max_value[1] = vphi;
					max_value[2] = vtheta;
				}
			}
		}
	}
	printf("max_value %f phi %f theta %f\n", max_value[0], max_value[1], max_value[2]);
	for (i = 0; i < params.projection_div_phi*params.projection_div_theta;i++) {
		printf("value %f \n", valuemap[i]);
	}
	delete[] value;
	*/
	



	///�]���l�̔�r��ځB
	printf("Writing value map\n");
	sprintf(name, "valuemap-%s-double-(phi%d_theta%d).raw", params.out_name, params.projection_div_phi, params.projection_div_theta);
	printf("%s", name);
	FILE *out;
	out = fopen(name, "wb");
	if (out == NULL) {
		printf("\nFILE cannot open\n");
		exit(0);
	};
	fwrite(valuemap, sizeof(double), params.projection_div_phi*params.projection_div_theta, out);
	fclose(out);
	printf("\nwriting_end\n\n");//valuemap�̏����o��
	
	printf("\n projection_trajectory_analysis_start\n");
	float v_phi_y = 0, v_theta_x = 0;//value_map��ł̍��W
	int x, y;
	float axis_phi = 0, axis_theta = 0;//trajectory�̎��p�x�Btemp�Ɉꉞ�i�[���Ă���B
	float t = 0;//�O���̊p�x�B�J�X����B
	double trajectory_value = 0;
	double *trajectory_index = new double[(params.projection_div_phi - 1)*params.projection_div_theta];//�Ƃ肠�����A�]���O���͓��e���̐��ƈ�v������B

	for (i = 0; i < params.projection_div_phi - 1; i++) {
		axis_theta = 0;
		axis_phi += PI / params.projection_div_phi;
		for (j = 0; j < params.projection_div_theta; j++ ) {
			trajectory_value = 0;
			for (t = 0; t < PI; t += 0.05) {//����Ŕ���]
				v_phi_y = acos(-sin(axis_phi)*cos(t));
				v_theta_x = PI/2 + atan((-sin(axis_theta)*cos(axis_phi)*cos(t) + cos(axis_theta)*sin(t)) / (cos(axis_theta)*cos(axis_phi)*cos(t) - sin(axis_theta)*sin(t)));
				y = (int)(params.projection_div_phi - 1)*(v_phi_y / PI);
				x = (int)(params.projection_div_theta)*(v_theta_x / PI);
				trajectory_value += valuemap[y*params.projection_div_theta + x];
			}
			printf("trajectory value %lf phi %f theta %f\n", trajectory_value, axis_phi, axis_theta);
			trajectory_index[params.projection_div_theta*i + j] = trajectory_value;
			axis_theta += PI / params.projection_div_theta;
		}
	}//����Ŋe�O�����Ƃ̋P�x�l�̍��v�����[���ꂽ�B

	trajectory_value = trajectory_index[0];
	for (i = 0; i < params.projection_div_phi - 1; i++) {
		for (j = 0; j < params.projection_div_theta; j++) {
			if (trajectory_value > trajectory_index[params.projection_div_theta*i + j]) {
				trajectory_value = trajectory_index[params.projection_div_theta*i + j];
				x = j;
				y = i;
				axis_phi = x*PI / params.projection_div_phi;
				axis_theta = y*PI / params.projection_div_theta;
			}	
		}
	}
	printf("max_value_%lf axis phi_%f theta_%f\n", trajectory_value, axis_phi, axis_theta);


	////////////////�������J��///////////////
	delete[] trajectory_index;
	delete[] valuemap;
	
	////////////////�������J��///////////////

	printf("program_end\n");
	system("pause");
	return 0;
}