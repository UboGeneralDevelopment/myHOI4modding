
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <math.h>
#include <time.h>
#include <float.h>


#define PI 3.1415926535897932384626433832795f
#define EPSILON 0.000001

#define LAMBDA 1.0 //OSConvex�@�̃p�����^

#define threadN 512 //���݂� CUDA �ł� 512 �� MAX
#define tomoBlock 10 //foward-projection �̂Ƃ��Ɉ�x�ɓ]������f�ʐ�
#define ITER 50 //10 //�J��Ԃ���
#define SUB 20 //OS�@�̕�����
#define DT 0.5 //ray-marching �̃X�e�b�v�T�C�Y (�{�N�Z���T�C�Y�Ƃ̑��Βl)//�p��̓���B�s�N�Z���T�C�Y�A�{�N�Z���T�C�Y�Ƃ����̂́A�����ł̃T�C�Y�̂���

__global__ void forward_GPU(int W, int H, float dW, float dH, float D,//���e�̉��Əc�̃s�N�Z�����ƃT�C�Y�A���̂Ƃ��񂰂񋗗�
                            int subP, float *sum, //ray-sum �摜�B�����̒l�̎��W���u (�T�C�Y�� W x H x subP)
                            float sXY, float sZ, //�č\���~���̔��a�ƁA�����̔����B
                            int N, int zS, int zE, float *F, float pitch, //CT�摜�̒f�ʁA����xy�s�N�Z����N�A���� (zS <= z < zE)�B�t�@�C����F�B���ʂ̒f�ʂ�tomoblock�������Ă���Bpitch�̓{�N�Z���T�C�Y
                            float *rotS, float *rotC, //��]�ϊ��̂��߂̃e�[�u�� (������ subP ��)
                            float *ray, float dt //�����̃f�[�^�B�����͊p�x�Ǝn�_�A�I�_������Bdt�̓X�e�b�v���ŁA�{�N�Z���T�C�Y�̔����̒���
                            ){
  
  int id = blockIdx.x*blockDim.x + threadIdx.x;//�������ƂɃX���b�h�����蓖��
  if(id < W*H){
    int id5 = 5*id;//�X���b�h�Ŏ��s�������������������Ă��Ă���Bid��0�X�^�[�g���ۂ��B
    float dx1 = ray[id5++];
    float dy1 = ray[id5++];
    float dz = ray[id5++];
    float ts = ray[id5++];
    float te = ray[id5];
    
    //�Ώۂ� volume �̈�̂� marching ����悤�ɊJ�n�E�I���ʒu�𒲐��B�g���O���t���ׂĂ������Ă���킯�ł͂Ȃ����߁B
    if(dz != 0 && ts > 0){
      float t1 = ((zS)*pitch - sZ)/dz;
      float t2 = ((zE)*pitch - sZ)/dz;
      if(t1 < t2){
        if(t1 > ts)
          ts = t1;
        if(t2 < te)
          te = t2;
      }
      else{
        if(t2 > ts)
          ts = t2;
        if(t1 < te)
          te = t1;
      }
    }
    else if(ts > 0){ //dz = 0 �̂Ƃ�
      if((zS)*pitch > sZ || (zE)*pitch < sZ)
        ts = -1;
    }
    
    if(ts > 0 && ts < te){
      //�X�^�[�g�ʒu���A��������dt�̐����{�̋����ɂȂ�悤�ɂ���B�X�^�[�g�����ɂ����Ƌ߂��X�e�b�v�������X�^�[�g�Ƃ���B
      ts = ((int)(ts/dt))*dt;
      
      //�{�N�Z���T�C�Y���Z�̃X�e�b�v�� z�@�{�N�Z����ɂ��Az���ǂꂾ���i�ނ��B
      dz = dz/pitch;
      //X�����̃{�N�Z���ʒu z�@���W�����S�͂��Ԃ�č\���{�b�N�X�̒[����
      float sz = sZ/pitch;
      
      for(int i=0; i<subP/4; i++){//�����Ă���CT�f�ʂɑ΂��āA�쐬���鏇���e�̐���������]������B
        float rS = -rotS[i];
        float rC = rotC[i];
        
        //X�����̃{�N�Z���ʒu x y
        float sx = (D*rS + sXY)/pitch; 
        float sy = (-D*rC + sXY)/pitch;
        
        //�{�N�Z���T�C�Y���Z�̃X�e�b�v�� x y
        float dx = (dx1*rC - dy1*rS)/pitch;
        float dy = (dx1*rS + dy1*rC)/pitch;
        
        float v = 0;//�l�̊i�[��
        
        //���C�}�[�`���O
        for(float t=ts; t<te; t+=dt){
          int x = (int)(sx + t*dx);//�����̈�X�e�b�v���ƂɁAxyz�����ɉ��{�N�Z�����i�ނ����v�Z���Aint�Ŋۂ߂��čŏI�I�ȓ_���W�𓾂Ă���B
          int y = (int)(sy + t*dy);
          int z = (int)(sz + t*dz);
          if(x >= 0 && y >= 0 && z >= zS &&
             x < N && y < N && z < zE)//������̓_���A�č\���~���̒u���Ă���{�b�N�X��ԁi�c���{�N�Z����N�AzS�������{�N�Z���͈́�zE�j�ɂ��鎞�A
            v += F[((z-zS)*N+y)*N+x];//���ɂ��̍��\��ɂ���g���O���t�̃{�N�Z���̒l�����Z���Ă����B
        }
        sum[i*W*H+id] += v;
      }
    }
  }//���̈�A�̏����e�ɂ��A�쐬���鏇���e���ׂĂ�zS<����<zE�͈̔͂̏����e��������B���Ƃ͂�������������ɌJ��Ԃ��B
}

__global__ void backpro_GPU(int W, int H, float dW, float dH, float D, float d, float shift,//�f�B�e�N�^�̃s�N�Z�����ƃT�C�Y
                                     int subP,// float *sum, //ray-sum �摜 (�T�C�Y�� W x H x subP)
                                     float* diff, //OSConvex�@�̕��q�̈ꕔ�Bexp(-dt*sum)-S
                                     float* bunbo, //OSConvex�@�̕���Bdt*sum*exp(-dt*sum)
                                     //float dt,
                                     float sXY, float sZ, //���u��CT�č\���̈�̑傫��
                                     int N, int z_id, float *F, //CT�摜��1�f��
                                     float *rotS, float *rotC, //��]�ϊ��̂��߂̃e�[�u�� (������ subP ��)
                                     float *pxy, float *pz //���W�̃f�[�^
                                     ){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if(id < N*N){
    float x1 = pxy[2*id];
    if(x1 < sXY){
      float y1 = pxy[2*id+1];
      float z = pz[z_id];
      float f = 0;
      float w = 0;
      for(int i=0; i<subP/4; i++){
        float x = x1*rotC[i] - y1*rotS[i];
        float y = x1*rotS[i] + y1*rotC[i];
        
        float U = D/(D+y);
        
        float s = x*U + dW;
        float ds = s/d - 0.5f + shift;
        int n = ds;  //detector ID
        
        float v = z*U + dH;
        float dv = v/d - 0.5f;
        int m = dv;  //detector ID
        
        if(n >= 0 && n <= W-2 && m >= 0 && m <= H-2){
          //if(n > W/2+10)
            //continue;
          ds -= n; //for linear interpolation on detector along X
          dv -= m; //for linear interpolation on detector along Z
          int idx = (i*H+m)*W+n;
          f += (1.0f-dv)*((1.0f-ds)*diff[idx] + ds*diff[idx+1]) +
            dv*((1.0f-ds)*diff[idx+W]+ds*diff[idx+W+1]);
          
          w += (1.0f-dv)*((1.0f-ds)*bunbo[idx] + ds*bunbo[idx+1]) +
            dv*((1.0f-ds)*bunbo[idx+W]+ds*bunbo[idx+W+1]);
        }
      }
      
      //OSConvex�@�ŕ␳
      float lambda = 1.0f; //0.5;//0.00005;
      if(w != 0)
        F[id] *= (1+lambda*f/w);
      
      if( F[id]<0 ) F[id] = 0.0;
    }
  }
}

void convex(int XY, int Z, float **F, float sXY, float sZ, //tomogram�A���ʉ��c�{�N�Z�����A�����{�N�Z�����A�g���O�����{�����[���A�č\���f�ʉ��T�C�Y�̔����i�~�����a�j�A�f�ʏc�T�C�Y�̔���
            int W, int H, int P, float **S, //sinogram�A�c�s�N�Z�����A���s�N�Z�����A�����A�{�����[��
            float d, float D, float shift,//d�͓��e���I�u�W�F�N�g���W�ɒu�������̃s�N�Z���T�C�Y,D�͕��̂Ɛ�������,shit�͂���
            float *rotS, float *rotC){
  //(XY, XY, Z): CT�{�����[���̃{�N�Z�����D(sXY, sXY, sZ): �{�����[���̃T�C�Y�̔���
  
  int i, j, k;
  
  float s2 = sXY*sXY;//�č\�����a�̓��
  float dW = 0.5f*d*W;//���e�̉��T�C�Y/2
  float dH = 0.5f*d*H;//���e�̏c�T�C�Y/2
  
  float pitch = 2.0f*sXY/XY; //�{�N�Z���̃T�C�Y
  
  int subN = P/SUB;//��x�̌v�Z�ō쐬���鏇���e�̖���
  
  float *rot_tmp = (float*)malloc(sizeof(float)*subN);
  
  //ray-sum �̒l. backprojection ���ɂ͕␳�ʂɏ��������
  float *sum = (float*)malloc(sizeof(float)*W*H*subN);//���e�l�̗e��
  
  //�����̏�� �������f�e�N�^�[�ւ̒P�ʕ����x�N�g���i���ʂ�xy�A����z�Ƃ���j�Ǝn�_�E�I�_�̃p�����[�^
  float *ray = (float*)malloc(sizeof(float)*W*H*5);//�x�N�g��3�v�f�Ǝn�_�ƏI�_�ʒu���i�[���Aray��ɂ�5�v�f�A���ꂪ�V�m�O�����̕��ʌ�����B
  for(int i=0; i<H; i++){
    for(int j=0; j<W; j++){
      float dx = (j+0.5f-shift)*d-dW;//�������猩�����e��̓_��x���W�B�f�e�N�^�Ɛ����̒��S�_�̉�����␳�t���B
      float dy = D;//�������猩�����e��̓_��y���W�A���s�B
      float dz = (i+0.5f)*d-dH;//�錾���猩�����e��̓_��z���W
      float l = sqrt(dx*dx+dy*dy+dz*dz);//�x�N�g������
      dx /= l;//���K��
      dy /= l;
      dz /= l;
      
      ray[5*(i*W+j)  ] = dx;
      ray[5*(i*W+j)+1] = dy;
      ray[5*(i*W+j)+2] = dz;
      
      float A = dx*dx + dy*dy;//�������炳���Ŏn�_�ƏI�_�̌v�Z���s���Ă��邪�A�����炭�͉~������ђʂ���ŏ��̒����ƏI���̒��������ꂼ��ts��te�Ƃ��Ă���
      float B = -dy*D;
      float C = D*D - s2;
      float Det = B*B - A*C;
      //float v = 0;
      if(Det > 0){
        Det = sqrt(Det);
        float ts = (-B-Det)/A;
        float te = (-B+Det)/A;
        ray[5*(i*W+j)+3] = ts;
        ray[5*(i*W+j)+4] = te;
        if(dz > 0){
          float tz = sZ/dz;
          if(tz < ts)
            ray[5*(i*W+j)+3] = -1;
          else if(tz < te)
            ray[5*(i*W+j)+4] = tz;
        }
        else if(dz < 0){
          float tz = -sZ/dz;
          if(tz < ts)
            ray[5*(i*W+j)+3] = -1;
          else if(tz < te)
            ray[5*(i*W+j)+4] = tz;
        }
      }
      else
        ray[5*(i*W+j)+3] = -1;
    }
  }
  
  float *pxy = (float*)malloc(sizeof(float)*XY*XY*2);
  for(i=0; i<XY; i++){
    float y = (i+0.5f)*pitch - sXY; //�ύX
    for(j=0; j<XY; j++){
      float x = (j+0.5f)*pitch - sXY; //�ύX
      if(x*x + y*y > s2)
        x = sXY;
      pxy[2*(i*XY+j)  ] = x;
      pxy[2*(i*XY+j)+1] = y;
    }
  }
  
  float* pz = (float*)malloc(sizeof(float)*Z);
  for(i=0; i<Z; i++)
    pz[i] = (i+0.5f)*pitch - sZ; //�ύX
  
  for(i=0; i<Z; i++)
    for(j=0; j<XY; j++)
      for(k=0; k<XY; k++)
        if(pxy[2*(j*XY+k)] != sXY)
          F[i][j*XY+k] = 0.00005/sXY;//1;//0; //OSConvex�@�͂珉���l��1/(2*���a*10000), ��Z�␳�Ȃ珉���l��1, ���Z�␳�Ȃ珉���l��0
        else
          F[i][j*XY+k] = 0;
  
  float dt = DT*pitch;//�{�N�Z���T�C�Y�̔����B���̕��ŃX�e�b�v���i��ł����B
  
  float* d_sum;
  float* d_F;
  float* d_rotS;
  float* d_rotC;
  float* d_ray;
  float* d_pxy;
  float* d_pz;
  cudaMalloc((void**)&d_sum, sizeof(float)*subN*W*H); //forward�̒��O
  cudaMalloc((void**)&d_F, sizeof(float)*tomoBlock*XY*XY);
  cudaMalloc((void**)&d_rotS, sizeof(float)*subN);
  cudaMalloc((void**)&d_rotC, sizeof(float)*subN);
  cudaMalloc((void**)&d_ray, sizeof(float)*5*W*H); //forward �̒��O�Aray���R�s�[���ċL��
  cudaMemcpy(d_ray, ray, sizeof(float)*5*W*H, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_pxy, sizeof(float)*XY*XY*2); //back�̒��O
  cudaMemcpy(d_pxy, pxy, sizeof(float)*XY*XY*2, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_pz, sizeof(float)*Z); //back�̒��O
  cudaMemcpy(d_pz, pz, sizeof(float)*Z, cudaMemcpyHostToDevice);
  
  //OSConvex�@�ŕ␳ //back�̒��O
  float* tmp_OSConvex = (float*)malloc(sizeof(float)*W*H*subN);
  float* d_bunbo;
  cudaMalloc((void**)&d_bunbo, sizeof(float)*subN*W*H);
  
  float* f_tmp = (float*)malloc(sizeof(float)*XY*XY*tomoBlock);
  
  int iter, sub;
  for(iter=0; iter<ITER; iter++){
    printf("Iteration %d / %d:\n  Sub-iteration: ", iter+1, ITER);
    for(sub=0; sub<SUB; sub++){
      printf("%d ", sub+1);
      
      //0�ŏ��������� ray-sum �摜�𑗂�
      for(i=0; i<subN*W*H; i++)
        sum[i] = 0;
      cudaMemcpy(d_sum, sum, sizeof(float)*subN*W*H, cudaMemcpyHostToDevice);
      
      //��]�̂��߂̃e�[�u���𑗂�
      for(i=0; i<subN; i++)
        rot_tmp[i] = rotS[sub+i*SUB];//�쐬���鏇���e�̐��ɂR�U�O�x�𓙕�������B
      cudaMemcpy(d_rotS, rot_tmp, sizeof(float)*subN, cudaMemcpyHostToDevice);
      for(i=0; i<subN; i++)
        rot_tmp[i] = rotC[sub+i*SUB];
      cudaMemcpy(d_rotC, rot_tmp, sizeof(float)*subN, cudaMemcpyHostToDevice);
      
      //tomogram �𕪂��� ray-sum �����Z���Ă���
      for(i=0; i<Z; i+=tomoBlock){
        for(j=0; j<tomoBlock; j++){
          if(i+j==Z)
            break;
          for(k=0; k<XY*XY; k++)
            f_tmp[j*XY*XY+k] = F[i+j][k];//f_temp�ɂ̓g���O�����̒f�ʂ̑w��tomoblock�������Ă���
        }
        cudaMemcpy(d_F, f_tmp, sizeof(float)*XY*XY*j, cudaMemcpyHostToDevice);//d_F��f_temp�𑗂�
        int e = i + tomoBlock;
        if(e > Z)
          e = Z;
        int blockN = (H*W+threadN-1)/threadN;//�u���b�N���𓊉e�̃s�N�Z�������u���b�N������X���b�h���Ŋ��������������������߂ɐݒ�B
        
        forward_GPU<<< blockN, threadN >>>(W, H, dW, dH, D,
                                           subN, d_sum,
                                           sXY, sZ,
                                           XY, i, e, d_F, pitch,
                                           d_rotS, d_rotC,
                                           d_ray, dt);
        cudaThreadSynchronize();
      }
      
      //ray-sum�̒l���擾
      cudaMemcpy(sum, d_sum, sizeof(float)*subN*W*H, cudaMemcpyDeviceToHost);
      
      for(i=0; i<subN; i++)
        for(j=0; j<H; j++)
          for(k=0; k<W; k++){
            
            double t = (double)sum[(i*H+j)*W+k]*dt;
            double exp_mt = exp( -t );
            
            sum[(i*H+j)*W+k] = (float)(exp_mt - S[SUB*i+sub][j*W+k]);//�z�� sum[] ���g���܂킷
            
            tmp_OSConvex[(i*H+j)*W+k] = (float)(t*exp_mt);
          }
      
      cudaMemcpy( d_sum, sum, sizeof(float)*subN*W*H, cudaMemcpyHostToDevice);//�z�� d_sum[] ���g���܂킷
      cudaMemcpy( d_bunbo, tmp_OSConvex, sizeof(float)*subN*W*H, cudaMemcpyHostToDevice);
      
      //�␳�ʂ� backprojection ����
      
      for(i=0; i<Z; i++){
        cudaMemcpy(d_F, F[i], sizeof(float)*XY*XY, cudaMemcpyHostToDevice);
        int blockN = (XY*XY+threadN-1)/threadN;
        
        backpro_GPU<<< blockN, threadN >>>(W, H, dW, dH, D, d, shift,
                                           subN,
                                           d_sum,
                                           d_bunbo,
                                           sXY, sZ,
                                           XY, i, d_F,
                                           d_rotS, d_rotC,
                                           d_pxy, d_pz);
        
        cudaThreadSynchronize();
        cudaMemcpy(F[i], d_F, sizeof(float)*XY*XY, cudaMemcpyDeviceToHost);
        for(j=0; j<XY*XY; j++)
          if(isfinite (F[i][j]) == 0)
            F[i][j] = 0;
      }
    }
    printf("\n");
    //�r���̒l���o��
    
    if( iter%5==0 ){
      char name[255];
      sprintf(name, "temp%d-%dx%dx%d-float.raw", iter, XY, XY, Z );
      FILE* outF = fopen(name, "w+b");
      for(i=0; i<Z; i++)
        fwrite(F[i], XY*XY, 4, outF);
      fclose(outF);
    }
  }
  
  cudaFree(d_sum);
  cudaFree(d_F);
  cudaFree(d_rotS);
  cudaFree(d_rotC);
  cudaFree(d_ray);
  cudaFree(d_pxy);
  cudaFree(d_pz);
  
  cudaFree(d_bunbo);
  free(tmp_OSConvex);
  free(rot_tmp);
  free(sum);
  free(ray);
  free(pxy);
  free(pz);
}

int main(int argc, char** argv){
  int i, j;
  
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("#GPU = %d\n", deviceCount);
  int device;
  int useGPU = 0;
  for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d (%s) has compute capability %d.%d.\nsharedMem = %dKB\nglobalMem = %dMB\n",
           device, deviceProp.name,
           deviceProp.major, deviceProp.minor,
           deviceProp.sharedMemPerBlock/1024,
           deviceProp.totalGlobalMem/1024/1024);
  }
  printf("Using GPU %d\n", useGPU);
  cudaSetDevice(useGPU);
  
  char in_path[1000], out_name[1000], format[100];
  float s2o, s2d;
  int W, H, P;
  float detP;
  
  int N;
  float scaleW, scaleH;
  float shift;
  
  float dW, dH;
  float sizeR, sizeH;
  int zN;
  float **ct, **sino;
  FILE *in, *out, *para;
  float delta;
  
  char name[1000];
  
  time_t timeS, timeE;
  time(&timeS);
  
  //�p�����[�^���t�@�C������Ǎ�
  para = fopen(argv[1], "r");
  
  fscanf(para, "%s", in_path);
  fscanf(para, "%s", out_name);
  fscanf(para, "%f %f", &s2d, &s2o);//�f�e�N�^�A���́A�����̋����Ȃ�
  fscanf(para, "%d %d %d %f", &W, &H, &P, &detP);//���e���̉��s�N�Z���A�c�s�N�Z�����A�����A�s�N�Z���T�C�Y
  fscanf(para, "%d %f %f", &N, &scaleW, &scaleH);//�{�����[���̂����̃{�N�Z�����A���Əc�̍č\���͈́i�č\�����̂�����������邩���߂�j
  fscanf(para, "%f", &shift);//�f�e�N�^�[���S�Ɛ����̉�����␳
  if(fscanf(para, "%s", format) == EOF) //uint16, float, log
    sprintf(format, "uint16");
  
  fclose(para);
  
  //detector size
  dW = detP*W;
  dH = detP*H;
  
  //image size (cylinder)//�č\�����s���{�����[���̃T�C�Y�B�~���`
  sizeR = (float)(0.5f*s2o*dW/sqrt(s2d*s2d + 0.25f*dW*dW)); //radius�A�č\�����s���͈͂̉~���̔��a
  sizeH = 0.5f*(s2o)/s2d*dH; //height/2�A�~�������̔���
  sizeR *= scaleW;
  sizeH *= scaleH;
  
  //#volxels in z-axis
  zN = (int)(sizeH*N/sizeR);//�~���̒f�ʂ̍����ƕ��̔�Ń{�N�Z����������B
  
  printf("Volume Size = %f x %f x %f\n", 2*sizeR, 2*sizeR, 2*sizeH);
  printf("#voxels = %d x %d x %d\n", N, N, zN);
  printf("Voxel size = %f\n", 2*sizeR/N);
  printf( "Sinogram format: %s\n", format );
  
  //tomogram
  ct = (float**)malloc(zN*sizeof(float*));
  for(i=0; i<zN; i++)
    ct[i] = (float*)malloc(N*N*sizeof(float));
  
  //sinogram
  sino = (float**)malloc(P*sizeof(float*));
  for(i=0; i<P; i++)
    sino[i] = (float*)malloc(W*H*sizeof(float));
  
  in = fopen(in_path, "rb");
  
  if(strcmp(format, "float") == 0){ // I / I_0 in [0,1]
    for(i=0; i<P; i++)
      for(j=0; j<H; j++)
        fread(&(sino[i][W*(H-j-1)]), sizeof(float), W, in);
  }
  else if(strcmp(format, "log") == 0){ //projection values
    for(i=0; i<P; i++)
      for(j=0; j<H; j++)
        fread(&(sino[i][W*(H-j-1)]), sizeof(float), W, in);
    for(i=0; i<P; i++)
      for(j=0; j<H; j++)
        for(int k=0; k<W; k++)
          sino[i][j*W+k] = (float)exp(-sino[i][j*W+k]);
  }
  else{
    unsigned short maxV = 0;
    unsigned short *tmp = (unsigned short*)malloc(W*sizeof(unsigned short));
    for(i=0; i<P; i++)
      for(j=0; j<H; j++){
        fread(tmp, sizeof(unsigned short), W, in);
        for(int k=0; k<W; k++)
          if(tmp[k] > maxV)
            maxV = tmp[k];
      }
    rewind( in );
    printf("Air-value: %d\n", maxV);
    for(i=0; i<P; i++)
      for(j=0; j<H; j++){
        fread(tmp, sizeof(unsigned short), W, in);
        for(int k=0; k<W; k++)
          sino[i][W*(H-j-1)+k] = (float)tmp[k]/maxV;
      }
    delete[] tmp;
  }
  
  fclose(in);
  
  delta = (float)(detP*s2o/s2d); //scaled detector pitch���e�s�N�Z���T�C�Y���I�u�W�F�N�g����W�ł̃T�C�Y�ɕϊ������B
  
  /* ���e�p�x�ݒ� */
  float *rotS = (float*)malloc(sizeof(float)*P);
  float *rotC = (float*)malloc(sizeof(float)*P);
  for(i=0; i<P; i++){
    float a = PI*(2.0f*i/P);
    rotS[i] = (float)sin(a);
    rotC[i] = (float)cos(a);
  }
  
  convex(N, zN, ct, sizeR, sizeH,
         W, H, P, sino, delta, s2o, shift,
         rotS, rotC);
  
  free(rotS);
  free(rotC);
  
  for(i=0; i<P; i++)
    free(sino[i]);
  free(sino);
  
  printf("Writing\n");
  
  sprintf(name, "%s-float-%dx%dx%d-%.6gmm.raw",
          out_name, N, N, zN, 2*sizeR/N);
  
  out = fopen(name, "wb");
  for(i=0; i<zN; i++)
    fwrite(ct[i], 4, N*N, out);
  fclose(out);
  
  for(i=0; i<zN; i++)
    free(ct[i]);
  free(ct);
  
  time(&timeE);
  printf("Timing = %f sec.\n", difftime(timeE, timeS));
  
  return 0;
}
