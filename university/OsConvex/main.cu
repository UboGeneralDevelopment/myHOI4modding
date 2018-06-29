
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <math.h>
#include <time.h>
#include <float.h>


#define PI 3.1415926535897932384626433832795f
#define EPSILON 0.000001

#define LAMBDA 1.0 //OSConvex法のパラメタ

#define threadN 512 //現在の CUDA では 512 が MAX
#define tomoBlock 10 //foward-projection のときに一度に転送する断面数
#define ITER 50 //10 //繰り返し回数
#define SUB 20 //OS法の分割数
#define DT 0.5 //ray-marching のステップサイズ (ボクセルサイズとの相対値)//用語の統一。ピクセルサイズ、ボクセルサイズというのは、現実でのサイズのこと

__global__ void forward_GPU(int W, int H, float dW, float dH, float D,//投影の横と縦のピクセル数とサイズ、物体とせんげん距離
                            int subP, float *sum, //ray-sum 画像。光線の値の収集装置 (サイズは W x H x subP)
                            float sXY, float sZ, //再構成円柱の半径と、高さの半分。
                            int N, int zS, int zE, float *F, float pitch, //CT画像の断面、平面xyピクセル数N、高さ (zS <= z < zE)。ファイルはF。普通の断面がtomoblock枚入っている。pitchはボクセルサイズ
                            float *rotS, float *rotC, //回転変換のためのテーブル (長さは subP 個)
                            float *ray, float dt //光線のデータ。光線は角度と始点、終点がある。dtはステップ幅で、ボクセルサイズの半分の長さ
                            ){
  
  int id = blockIdx.x*blockDim.x + threadIdx.x;//光線ごとにスレッドを割り当て
  if(id < W*H){
    int id5 = 5*id;//スレッドで実行される光線を引っ張ってきている。idは0スタートっぽい。
    float dx1 = ray[id5++];
    float dy1 = ray[id5++];
    float dz = ray[id5++];
    float ts = ray[id5++];
    float te = ray[id5];
    
    //対象の volume 領域のみ marching するように開始・終了位置を調整。トモグラフすべてが入ってくるわけではないため。
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
    else if(ts > 0){ //dz = 0 のとき
      if((zS)*pitch > sZ || (zE)*pitch < sZ)
        ts = -1;
    }
    
    if(ts > 0 && ts < te){
      //スタート位置が、線源からdtの整数倍の距離になるようにする。スタート距離にもっと近いステップ距離をスタートとする。
      ts = ((int)(ts/dt))*dt;
      
      //ボクセルサイズ換算のステップ量 z　ボクセル一つにつき、zがどれだけ進むか。
      dz = dz/pitch;
      //X線源のボクセル位置 z　座標軸中心はたぶん再構成ボックスの端っこ
      float sz = sZ/pitch;
      
      for(int i=0; i<subP/4; i++){//送られてきたCT断面に対して、作成する順投影の数分だけ回転させる。
        float rS = -rotS[i];
        float rC = rotC[i];
        
        //X線源のボクセル位置 x y
        float sx = (D*rS + sXY)/pitch; 
        float sy = (-D*rC + sXY)/pitch;
        
        //ボクセルサイズ換算のステップ量 x y
        float dx = (dx1*rC - dy1*rS)/pitch;
        float dy = (dx1*rS + dy1*rC)/pitch;
        
        float v = 0;//値の格納庫
        
        //レイマーチング
        for(float t=ts; t<te; t+=dt){
          int x = (int)(sx + t*dx);//ここの一ステップごとに、xyz方向に何ボクセル分進むかを計算し、intで丸められて最終的な点座標を得ている。
          int y = (int)(sy + t*dy);
          int z = (int)(sz + t*dz);
          if(x >= 0 && y >= 0 && z >= zS &&
             x < N && y < N && z < zE)//光線上の点が、再構成円柱の置いてあるボックス空間（縦横ボクセル数N、zS＜高さボクセル範囲＜zE）にある時、
            v += F[((z-zS)*N+y)*N+x];//ｖにその座表情にあるトモグラフのボクセルの値を加算していく。
        }
        sum[i*W*H+id] += v;
      }
    }
  }//この一連の順投影により、作成する順投影すべてのzS<高さ<zEの範囲の順投影がつくられる。あとはこれを高さ方向に繰り返す。
}

__global__ void backpro_GPU(int W, int H, float dW, float dH, float D, float d, float shift,//ディテクタのピクセル数とサイズ
                                     int subP,// float *sum, //ray-sum 画像 (サイズは W x H x subP)
                                     float* diff, //OSConvex法の分子の一部。exp(-dt*sum)-S
                                     float* bunbo, //OSConvex法の分母。dt*sum*exp(-dt*sum)
                                     //float dt,
                                     float sXY, float sZ, //装置とCT再構成領域の大きさ
                                     int N, int z_id, float *F, //CT画像の1断面
                                     float *rotS, float *rotC, //回転変換のためのテーブル (長さは subP 個)
                                     float *pxy, float *pz //座標のデータ
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
      
      //OSConvex法で補正
      float lambda = 1.0f; //0.5;//0.00005;
      if(w != 0)
        F[id] *= (1+lambda*f/w);
      
      if( F[id]<0 ) F[id] = 0.0;
    }
  }
}

void convex(int XY, int Z, float **F, float sXY, float sZ, //tomogram、平面横縦ボクセル数、高さボクセル数、トモグラムボリューム、再構成断面横サイズの半分（円柱半径）、断面縦サイズの半分
            int W, int H, int P, float **S, //sinogram、縦ピクセル数、横ピクセル数、枚数、ボリューム
            float d, float D, float shift,//dは投影をオブジェクト座標に置いた時のピクセルサイズ,Dは物体と線源距離,shitはずれ
            float *rotS, float *rotC){
  //(XY, XY, Z): CTボリュームのボクセル数．(sXY, sXY, sZ): ボリュームのサイズの半分
  
  int i, j, k;
  
  float s2 = sXY*sXY;//再構成半径の二乗
  float dW = 0.5f*d*W;//投影の横サイズ/2
  float dH = 0.5f*d*H;//投影の縦サイズ/2
  
  float pitch = 2.0f*sXY/XY; //ボクセルのサイズ
  
  int subN = P/SUB;//一度の計算で作成する順投影の枚数
  
  float *rot_tmp = (float*)malloc(sizeof(float)*subN);
  
  //ray-sum の値. backprojection 時には補正量に書き換わる
  float *sum = (float*)malloc(sizeof(float)*W*H*subN);//投影値の容器
  
  //光線の情報 線源よりデテクターへの単位方向ベクトル（平面をxy、高さzとする）と始点・終点のパラメータ
  float *ray = (float*)malloc(sizeof(float)*W*H*5);//ベクトル3要素と始点と終点位置を格納し、ray一つにつき5要素、それがシノグラムの平面個分ある。
  for(int i=0; i<H; i++){
    for(int j=0; j<W; j++){
      float dx = (j+0.5f-shift)*d-dW;//線源から見た投影上の点のx座標。デテクタと線源の中心点の横ずれ補正付き。
      float dy = D;//線源から見た投影上の点のy座標、奥行。
      float dz = (i+0.5f)*d-dH;//宣言から見た投影上の点のz座標
      float l = sqrt(dx*dx+dy*dy+dz*dz);//ベクトル長さ
      dx /= l;//正規化
      dy /= l;
      dz /= l;
      
      ray[5*(i*W+j)  ] = dx;
      ray[5*(i*W+j)+1] = dy;
      ray[5*(i*W+j)+2] = dz;
      
      float A = dx*dx + dy*dy;//ここからさきで始点と終点の計算を行っているが、おそらくは円柱状を貫通する最初の長さと終わりの長さをそれぞれtsとteとしている
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
    float y = (i+0.5f)*pitch - sXY; //変更
    for(j=0; j<XY; j++){
      float x = (j+0.5f)*pitch - sXY; //変更
      if(x*x + y*y > s2)
        x = sXY;
      pxy[2*(i*XY+j)  ] = x;
      pxy[2*(i*XY+j)+1] = y;
    }
  }
  
  float* pz = (float*)malloc(sizeof(float)*Z);
  for(i=0; i<Z; i++)
    pz[i] = (i+0.5f)*pitch - sZ; //変更
  
  for(i=0; i<Z; i++)
    for(j=0; j<XY; j++)
      for(k=0; k<XY; k++)
        if(pxy[2*(j*XY+k)] != sXY)
          F[i][j*XY+k] = 0.00005/sXY;//1;//0; //OSConvex法はら初期値は1/(2*半径*10000), 乗算補正なら初期値は1, 加算補正なら初期値は0
        else
          F[i][j*XY+k] = 0;
  
  float dt = DT*pitch;//ボクセルサイズの半分。この幅でステップが進んでいく。
  
  float* d_sum;
  float* d_F;
  float* d_rotS;
  float* d_rotC;
  float* d_ray;
  float* d_pxy;
  float* d_pz;
  cudaMalloc((void**)&d_sum, sizeof(float)*subN*W*H); //forwardの直前
  cudaMalloc((void**)&d_F, sizeof(float)*tomoBlock*XY*XY);
  cudaMalloc((void**)&d_rotS, sizeof(float)*subN);
  cudaMalloc((void**)&d_rotC, sizeof(float)*subN);
  cudaMalloc((void**)&d_ray, sizeof(float)*5*W*H); //forward の直前、rayをコピーして記憶
  cudaMemcpy(d_ray, ray, sizeof(float)*5*W*H, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_pxy, sizeof(float)*XY*XY*2); //backの直前
  cudaMemcpy(d_pxy, pxy, sizeof(float)*XY*XY*2, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_pz, sizeof(float)*Z); //backの直前
  cudaMemcpy(d_pz, pz, sizeof(float)*Z, cudaMemcpyHostToDevice);
  
  //OSConvex法で補正 //backの直前
  float* tmp_OSConvex = (float*)malloc(sizeof(float)*W*H*subN);
  float* d_bunbo;
  cudaMalloc((void**)&d_bunbo, sizeof(float)*subN*W*H);
  
  float* f_tmp = (float*)malloc(sizeof(float)*XY*XY*tomoBlock);
  
  int iter, sub;
  for(iter=0; iter<ITER; iter++){
    printf("Iteration %d / %d:\n  Sub-iteration: ", iter+1, ITER);
    for(sub=0; sub<SUB; sub++){
      printf("%d ", sub+1);
      
      //0で初期化した ray-sum 画像を送る
      for(i=0; i<subN*W*H; i++)
        sum[i] = 0;
      cudaMemcpy(d_sum, sum, sizeof(float)*subN*W*H, cudaMemcpyHostToDevice);
      
      //回転のためのテーブルを送る
      for(i=0; i<subN; i++)
        rot_tmp[i] = rotS[sub+i*SUB];//作成する順投影の数に３６０度を等分割する。
      cudaMemcpy(d_rotS, rot_tmp, sizeof(float)*subN, cudaMemcpyHostToDevice);
      for(i=0; i<subN; i++)
        rot_tmp[i] = rotC[sub+i*SUB];
      cudaMemcpy(d_rotC, rot_tmp, sizeof(float)*subN, cudaMemcpyHostToDevice);
      
      //tomogram を分けて ray-sum を加算していく
      for(i=0; i<Z; i+=tomoBlock){
        for(j=0; j<tomoBlock; j++){
          if(i+j==Z)
            break;
          for(k=0; k<XY*XY; k++)
            f_tmp[j*XY*XY+k] = F[i+j][k];//f_tempにはトモグラムの断面の層がtomoblock枚入っている
        }
        cudaMemcpy(d_F, f_tmp, sizeof(float)*XY*XY*j, cudaMemcpyHostToDevice);//d_Fにf_tempを送る
        int e = i + tomoBlock;
        if(e > Z)
          e = Z;
        int blockN = (H*W+threadN-1)/threadN;//ブロック数を投影のピクセル数をブロック当たりスレッド数で割った数よりもすこし多めに設定。
        
        forward_GPU<<< blockN, threadN >>>(W, H, dW, dH, D,
                                           subN, d_sum,
                                           sXY, sZ,
                                           XY, i, e, d_F, pitch,
                                           d_rotS, d_rotC,
                                           d_ray, dt);
        cudaThreadSynchronize();
      }
      
      //ray-sumの値を取得
      cudaMemcpy(sum, d_sum, sizeof(float)*subN*W*H, cudaMemcpyDeviceToHost);
      
      for(i=0; i<subN; i++)
        for(j=0; j<H; j++)
          for(k=0; k<W; k++){
            
            double t = (double)sum[(i*H+j)*W+k]*dt;
            double exp_mt = exp( -t );
            
            sum[(i*H+j)*W+k] = (float)(exp_mt - S[SUB*i+sub][j*W+k]);//配列 sum[] を使いまわす
            
            tmp_OSConvex[(i*H+j)*W+k] = (float)(t*exp_mt);
          }
      
      cudaMemcpy( d_sum, sum, sizeof(float)*subN*W*H, cudaMemcpyHostToDevice);//配列 d_sum[] を使いまわす
      cudaMemcpy( d_bunbo, tmp_OSConvex, sizeof(float)*subN*W*H, cudaMemcpyHostToDevice);
      
      //補正量を backprojection する
      
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
    //途中の値を出力
    
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
  
  //パラメータをファイルから読込
  para = fopen(argv[1], "r");
  
  fscanf(para, "%s", in_path);
  fscanf(para, "%s", out_name);
  fscanf(para, "%f %f", &s2d, &s2o);//デテクタ、物体、線源の距離など
  fscanf(para, "%d %d %d %f", &W, &H, &P, &detP);//投影像の横ピクセル、縦ピクセル数、枚数、ピクセルサイズ
  fscanf(para, "%d %f %f", &N, &scaleW, &scaleH);//ボリュームのｘｙのボクセル数、横と縦の再構成範囲（再構成像のうち何割を作るか決める）
  fscanf(para, "%f", &shift);//デテクター中心と線源の横ずれ補正
  if(fscanf(para, "%s", format) == EOF) //uint16, float, log
    sprintf(format, "uint16");
  
  fclose(para);
  
  //detector size
  dW = detP*W;
  dH = detP*H;
  
  //image size (cylinder)//再構成を行うボリュームのサイズ。円柱形
  sizeR = (float)(0.5f*s2o*dW/sqrt(s2d*s2d + 0.25f*dW*dW)); //radius、再構成を行う範囲の円柱の半径
  sizeH = 0.5f*(s2o)/s2d*dH; //height/2、円柱高さの半分
  sizeR *= scaleW;
  sizeH *= scaleH;
  
  //#volxels in z-axis
  zN = (int)(sizeH*N/sizeR);//円柱の断面の高さと幅の比でボクセル数を決定。
  
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
  
  delta = (float)(detP*s2o/s2d); //scaled detector pitch投影ピクセルサイズをオブジェクト上座標でのサイズに変換した。
  
  /* 投影角度設定 */
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
