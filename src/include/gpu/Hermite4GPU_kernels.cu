/*
 * Copyright (c) 2016
 *
 * Cristián Maureira-Fredes <cmaureirafredes@gmail.com>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#undef _GLIBCXX_ATOMIC_BUILTINS
#include "Hermite4GPU.cuh"

/*
 * @fn k_init_acc_jr
 *
 *
 * @desc GPU Kernel which calculates the initial acceleration and jerk
 * of all the particles of the system.
 *
 */
__global__ void k_init_acc_jrk (Predictor *p,
                                Forces *f,
                                int n,
                                double e2,
                                int dev,
                                int dev_size)
{

    extern __shared__ Predictor sh[];

    Forces ff;
    ff.a[0]  = 0.0;
    ff.a[1]  = 0.0;
    ff.a[2]  = 0.0;
    ff.a1[0] = 0.0;
    ff.a1[1] = 0.0;
    ff.a1[2] = 0.0;

    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int tx = threadIdx.x;

    if (id < dev_size)
    {
      Predictor pred = p[id+(dev*dev_size)];
      //Predictor pred = p[id];
      int tile = 0;
      for (int i = 0; i < n; i += BSIZE)
      {
          int idx = tile * BSIZE + tx;
          sh[tx]   = p[idx];
          __syncthreads();
          for (int k = 0; k < BSIZE; k++)
          {
              k_force_calculation(pred, sh[k], ff, e2);
          }
          __syncthreads();
          tile++;
      }
      f[id] = ff;
    }
}

__device__ void k_force_calculation(const Predictor &i_p,
                                    const Predictor &j_p,
                                    Forces &f,
                                    const double &e2)
{
    double rx = j_p.r[0] - i_p.r[0];
    double ry = j_p.r[1] - i_p.r[1];
    double rz = j_p.r[2] - i_p.r[2];

    double vx = j_p.v[0] - i_p.v[0];
    double vy = j_p.v[1] - i_p.v[1];
    double vz = j_p.v[2] - i_p.v[2];

    double r2     = rx*rx + ry*ry + rz*rz + e2;
    double rinv   = rsqrt(r2);
    double r2inv  = rinv  * rinv;
    double r3inv  = r2inv * rinv;
    double r5inv  = r2inv * r3inv;
    double mr3inv = r3inv * j_p.m;
    double mr5inv = r5inv * j_p.m;

    double rv = rx*vx + ry*vy + rz*vz;

    f.a[0] += (rx * mr3inv);
    f.a[1] += (ry * mr3inv);
    f.a[2] += (rz * mr3inv);

    f.a1[0] += (vx * mr3inv - (3 * rv) * rx * mr5inv);
    f.a1[1] += (vy * mr3inv - (3 * rv) * ry * mr5inv);
    f.a1[2] += (vz * mr3inv - (3 * rv) * rz * mr5inv);
}
/*
 * @fn k_prediction
 *
 *
 * @desc GPU Kernel which calculates the predictors
 *
 */
__global__ void k_prediction(Forces *f,
                             double4 *r,
                             double4 *v,
                             double *t,
                             Predictor *p,
                             int dev_size,
                             double ITIME)
{

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < dev_size)
    {
        double dt  = ITIME - t[i];
        double dt2 = 0.5 * (dt * dt);
        double dt3 = 0.166666666666666 * (dt * dt * dt);

        Forces ff = f[i];
        double4 rr = r[i];
        double4 vv = v[i];

        p[i].r[0] = (dt3 * ff.a1[0]) + (dt2 * ff.a[0]) + (dt * vv.x) + rr.x;
        p[i].r[1] = (dt3 * ff.a1[1]) + (dt2 * ff.a[1]) + (dt * vv.y) + rr.y;
        p[i].r[2] = (dt3 * ff.a1[2]) + (dt2 * ff.a[2]) + (dt * vv.z) + rr.z;

        p[i].v[0] = (dt2 * ff.a1[0]) + (dt * ff.a[0]) + vv.x;
        p[i].v[1] = (dt2 * ff.a1[1]) + (dt * ff.a[1]) + vv.y;
        p[i].v[2] = (dt2 * ff.a1[2]) + (dt * ff.a[2]) + vv.z;

        p[i].m = rr.w;
    }
}

/*
 * @fn k_update()
 *
 * @brief Gravitational interaction kernel.
 */
__global__ void k_update(Predictor *i_p,
                         Predictor *j_p,
                         Forces *fout,
                         int n,
                         int total,
                         double e2)
{
    int ibid = blockIdx.x;
    int jbid = blockIdx.y;
    int tid  = threadIdx.x;
    int iaddr  = tid + blockDim.x * ibid;
    int jstart = (n * (jbid  )) / NJBLOCK;
    int jend   = (n * (jbid+1)) / NJBLOCK;

    Predictor ip = i_p[iaddr];
    Forces fo;
    fo.a[0] = 0.0;
    fo.a[1] = 0.0;
    fo.a[2] = 0.0;
    fo.a1[0] = 0.0;
    fo.a1[1] = 0.0;
    fo.a1[2] = 0.0;

        for(int j=jstart; j<jend; j+=BSIZE)
        {
            __shared__ Predictor jpshare[BSIZE];
            __syncthreads();
            Predictor *src = (Predictor *)&j_p[j];
            Predictor *dst = (Predictor *)jpshare;
            dst[      tid] = src[      tid];
            dst[BSIZE+tid] = src[BSIZE+tid];
            __syncthreads();

            // If the total amount of particles is not a multiple of BSIZE
            if(jend-j < BSIZE)
            {
                #pragma unroll 4
                for(int jj=0; jj<jend-j; jj++)
                {
                    Predictor jp = jpshare[jj];
                    k_force_calculation(ip, jp, fo, e2);
                }
            }
            else
            {
                #pragma unroll 4
                for(int jj=0; jj<BSIZE; jj++)
                {
                    Predictor jp = jpshare[jj];
                    k_force_calculation(ip, jp, fo, e2);
                }
            }
        }
        fout[iaddr*NJBLOCK + jbid] = fo;
}

/*
 * @fn k_reduce()
 *
 * @brief Forces reduction kernel
 */
__global__ void k_reduce(Forces *in,
                       Forces *out,
                       int shift_id,
                       int shift)
{
    extern __shared__ Forces sdata[];

    const int xid   = threadIdx.x;
    const int bid   = blockIdx.x;
    const int iaddr = xid + blockDim.x * bid;

    sdata[xid] = in[iaddr+shift*NJBLOCK];
    __syncthreads();

    if(xid < 8) sdata[xid] += sdata[xid + 8];
    if(xid < 4) sdata[xid] += sdata[xid + 4];
    if(xid < 2) sdata[xid] += sdata[xid + 2];
    if(xid < 1) sdata[xid] += sdata[xid + 1];

    if(xid == 0){
        out[bid] = sdata[0];
    }
}

__global__ void k_energy(double4 *r,
                         double4 *v,
                         double *ekin,
                         double *epot,
                         int n,
                         int dev_size,
                         int dev)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j;
    double ekin_tmp = 0.0;
    int id = i+dev*dev_size;

    if (i < dev_size)
    {
        double epot_tmp = 0.0;
        double4 ri = r[id];
        double4 vi = v[id];
        for (j = id+1; j < n; j++)
        {
            double rx = r[j].x - ri.x;
            double ry = r[j].y - ri.y;
            double rz = r[j].z - ri.z;
            double r2 = rx*rx + ry*ry + rz*rz;

            epot_tmp -= (ri.w * r[j].w) * rsqrt(r2);
        }

        double vx = vi.x * vi.x;
        double vy = vi.y * vi.y;
        double vz = vi.z * vi.z;
        double v2 = vx + vy + vz;

        ekin_tmp = 0.5 * ri.w * v2;

        ekin[i] = ekin_tmp;
        epot[i] = epot_tmp;
    }
}





__global__ void k_correction(unsigned int *move,
                             Forces *f,
                             Forces *old,
                             Predictor *p,
                             double4 *r,
                             double4 *v,
                             double *t,
                             double *dt,
                             double3 *a2,
                             double3 *a3,
                             unsigned int dev_size,
                             double ITIME,
                             double ETA)
{
  // thread index
  int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (thread_idx < dev_size)
  {
      // i is the particle to move (gets taken from (ns->h_)move)
      // i is an index into all the other arrays
      int i = move[thread_idx];

      Forces ff = f[i];
      Forces oo = old[i];
      Predictor pp = p[i];

      double dt1 = dt[i];
      double dt2 = dt1 * dt1;
      double dt3 = dt2 * dt1;
      double dt4 = dt2 * dt2;
      double dt5 = dt4 * dt1;

      double dt2inv = 1.0/dt2;
      double dt3inv = 1.0/dt3;

      double dt3_6 = 0.166666666666666*dt3;
      double dt4_24 = 0.041666666666666*dt4;
      double dt5_120 = 0.008333333333333*dt5;

      // Acceleration 2nd derivate
      a2[i].x = (-6 * (oo.a[0] - ff.a[0] ) - dt1 * (4 * oo.a1[0] + 2 * ff.a1[0]) ) * dt2inv;
      a2[i].y = (-6 * (oo.a[1] - ff.a[1] ) - dt1 * (4 * oo.a1[1] + 2 * ff.a1[1]) ) * dt2inv;
      a2[i].z = (-6 * (oo.a[2] - ff.a[2] ) - dt1 * (4 * oo.a1[2] + 2 * ff.a1[2]) ) * dt2inv;

      // Acceleration 3rd derivate
      a3[i].x = (12 * (oo.a[0] - ff.a[0] ) + 6 * dt1 * (oo.a1[0] + ff.a1[0]) ) * dt3inv;
      a3[i].y = (12 * (oo.a[1] - ff.a[1] ) + 6 * dt1 * (oo.a1[1] + ff.a1[1]) ) * dt3inv;
      a3[i].z = (12 * (oo.a[2] - ff.a[2] ) + 6 * dt1 * (oo.a1[2] + ff.a1[2]) ) * dt3inv;


      // Correcting position
      r[i].x = pp.r[0] + (dt4_24)*a2[i].x + (dt5_120)*a3[i].x;
      r[i].y = pp.r[1] + (dt4_24)*a2[i].y + (dt5_120)*a3[i].y;
      r[i].z = pp.r[2] + (dt4_24)*a2[i].z + (dt5_120)*a3[i].z;

      // Correcting velocity
      v[i].x = pp.v[0] + (dt3_6)*a2[i].x +   (dt4_24)*a3[i].x;
      v[i].y = pp.v[1] + (dt3_6)*a2[i].y +   (dt4_24)*a3[i].y;
      v[i].z = pp.v[2] + (dt3_6)*a2[i].z +   (dt4_24)*a3[i].z;

      t[i] = ITIME;
      double normal_dt = k_get_timestep_normal(ETA, a2[i], a3[i], dt[i], ff);
      dt[i] = k_normalize_dt(normal_dt, dt[i], t[i]);

  }

}

/** Vector magnitude calculation; copied from the one in NbodyUtils **/
__device__ double k_get_magnitude(const double &x, const double &y, const double &z)
{
  return sqrt(x*x + y*y + z*z);
}

/** Time step calculation; copied from the one in NbodyUtils.
Used to take an unsigned int i argument but I got rid if it.
**/
__device__ double k_get_timestep_normal(const float &ETA,
                                 const double3 &a2,
                                 const double3 &a3,
                                 const double &dt,
                                 const Forces &f)
{
  // Calculating a_{1,i}^{(2)} = a_{0,i}^{(2)} + dt * a_{0,i}^{(3)}
  double ax1_2 = a2.x + dt * a3.x;
  double ay1_2 = a2.y + dt * a3.y;
  double az1_2 = a2.z + dt * a3.z;

  // |a_{1,i}|
  double abs_a1 = k_get_magnitude(f.a[0],
                                f.a[1],
                                f.a[2]);
  // |j_{1,i}|
  double abs_j1 = k_get_magnitude(f.a1[0],
                                f.a1[1],
                                f.a1[2]);
  // |j_{1,i}|^{2}
  double abs_j12  = abs_j1 * abs_j1;
  // a_{1,i}^{(3)} = a_{0,i}^{(3)} because the 3rd-order interpolation
  double abs_a1_3 = k_get_magnitude(a3.x,
                                  a3.y,
                                  a3.z);
  // |a_{1,i}^{(2)}|
  double abs_a1_2 = k_get_magnitude(ax1_2, ay1_2, az1_2);
  // |a_{1,i}^{(2)}|^{2}
  double abs_a1_22  = abs_a1_2 * abs_a1_2;

  // variable used to be called "normal_dt" and was returned (just skipping the new variable declaration)
  return sqrt(ETA * ((abs_a1 * abs_a1_2 + abs_j12) / (abs_j1 * abs_a1_3 + abs_a1_22)));
}

/** Normalization of the timestep.
 * This method take care of the limits conditions to avoid large jumps between
 * the timestep distribution
 Copied from the version in NbodyUtils; that version takes an argument "unsigned int i"
 but does not use it, so I dropped that argument.

 For this, the local copy of new_dt is rewritten a lot, so I'm letting that one
 be non-constant. In k_correction, a newly declared variable (not used after this)
 is passed in as new_dt, so I will let it be pass-by-reference since we don't need
 to keep that data safe past this function.
 old_dt and t are still pass by reference and constant.
 */
__device__ double k_normalize_dt(double &new_dt,
                          const double &old_dt,
                          const double &t)
{
  if (new_dt <= old_dt/8)
  {
      new_dt = D_TIME_MIN;
  }
  else if ( old_dt/8 < new_dt && new_dt <= old_dt/4)
  {
      new_dt = old_dt / 8;
  }
  else if ( old_dt/4 < new_dt && new_dt <= old_dt/2)
  {
      new_dt = old_dt / 4;
  }
  else if ( old_dt/2 < new_dt && new_dt <= old_dt)
  {
      new_dt = old_dt / 2;
  }
  else if ( old_dt < new_dt && new_dt <= old_dt * 2)
  {
      new_dt = old_dt;
  }
  else if (2 * old_dt < new_dt)
  {
      double val = t/(2 * old_dt);
      //float val = t/(2 * old_dt);
      if(std::ceil(val) == val)
      {
          new_dt = 2.0 * old_dt;
      }
      else
      {
          new_dt = old_dt;
      }
  }
  else
  {
      //std::cerr << "this will never happen...I promise" << std::endl;
      new_dt = old_dt;
  }

  //if (new_dt <= D_TIME_MIN)
  if (new_dt < D_TIME_MIN)
  {
      new_dt = D_TIME_MIN;
  }
  //else if (new_dt >= D_TIME_MAX)
  else if (new_dt > D_TIME_MAX)
  {
      new_dt = D_TIME_MAX;
  }

  return new_dt;
}
