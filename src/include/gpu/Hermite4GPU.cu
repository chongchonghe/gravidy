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
#include "nvToolsExt.h"

/** Constructor that uses its parent one.
 * Additionally handles the split of the particles of the system among the available
 * GPUs, allocation of the variables, and defining widely use sizes for arrays.
 */
Hermite4GPU::Hermite4GPU(NbodySystem *ns, Logger *logger, NbodyUtils *nu)
            : Hermite4(ns, logger, nu)
{
    smem = sizeof(Predictor) * BSIZE;
    smem_reduce = sizeof(Forces) * NJBLOCK + 1;

    int detected_gpus;
    CSC(cudaGetDeviceCount(&detected_gpus));

    if (ns->gpus > 0)
    {
        gpus = ns->gpus;
    }
    else
    {
        gpus = detected_gpus;
    }

    if (detected_gpus > gpus)
    {
        std::string s = "";
        s += std::string("Not using all the available GPUs: ");
        s += std::string(SSTR(gpus));
        s += std::string(" of ");
        s += std::string(SSTR(detected_gpus));
        logger->log_warning(s);
    }

    logger->log_info(std::string("GPUs: ")+std::string(SSTR(gpus)));

    std::string ss = "";
    ss += std::string("Splitting ");
    ss += std::string(SSTR(ns->n));
    ss += std::string(" particles in ");
    ss += std::string(SSTR(gpus));
    ss += std::string(" GPUs");
    logger->log_info(ss);

    if (ns->n % gpus == 0)
    {
        size_t size = ns->n/gpus;
        for ( int g = 0; g < gpus; g++)
            n_part[g] = size;
    }
    else
    {
        size_t size = std::ceil(ns->n/(float)gpus);
        for ( int g = 0; g < gpus; g++)
        {
            if (ns->n - size*(g+1) > 0)
                n_part[g] = size;
            else
                n_part[g] = ns->n - size*g;
        }
    }

    for(int g = 0; g < gpus; g++)
    {
        std::string sss = "";
        sss += std::string("GPU ");
        sss += std::string(SSTR(g));
        sss += std::string(" particles: ");
        sss += std::string(SSTR(n_part[g]));
        logger->log_info(sss);
    }

    save_log2n(); // assign ns->log2n and ns->nblocks_reduce

    i1_size = ns->n * sizeof(int);
    d1_size = ns->n * sizeof(double);
    d3_size = ns->n * sizeof(double3);
    d4_size = ns->n * sizeof(double4);
    ff_size = ns->n * sizeof(Forces);
    pp_size = ns->n * sizeof(Predictor);

    alloc_arrays_device();
}

/** Destructor in charge of memory deallocation */
Hermite4GPU::~Hermite4GPU()
{
    free_arrays_device();
}

/** Method in charge of allocating the data structures on the available GPUs,
 * also initializing all the arrays to zero
 */
void Hermite4GPU::alloc_arrays_device()
{
    for(int g = 0; g < gpus; g++)
    {
        // Setting GPU
        CSC(cudaSetDevice(g));

        CSC(cudaMalloc((void**)&ns->d_r[g], d4_size));
        CSC(cudaMalloc((void**)&ns->d_v[g], d4_size));
        CSC(cudaMalloc((void**)&ns->d_f[g], ff_size));
        CSC(cudaMalloc((void**)&ns->d_p[g], pp_size));
        CSC(cudaMalloc((void**)&ns->d_t[g], d1_size));
        CSC(cudaMalloc((void**)&ns->d_i[g], pp_size));
        CSC(cudaMalloc((void**)&ns->d_dt[g], d1_size));

        CSC(cudaMalloc((void**)&ns->d_a2[g], d3_size));
        CSC(cudaMalloc((void**)&ns->d_a3[g], d3_size));
        CSC(cudaMalloc((void**)&ns->d_old[g], ff_size));

        CSC(cudaMalloc((void**)&ns->d_ekin[g], d1_size));
        CSC(cudaMalloc((void**)&ns->d_epot[g], d1_size));
        CSC(cudaMalloc((void**)&ns->d_move[g], i1_size));
        CSC(cudaMalloc((void**)&ns->d_fout[g], ff_size * NJBLOCK));
        CSC(cudaMalloc((void**)&ns->d_fout_tmp[g], ff_size * NJBLOCK));

        CSC(cudaMalloc((void**)&ns->d_nact[g], sizeof(unsigned int))); // single uint
        CSC(cudaMalloc((void**)&ns->d_max_mass[g], sizeof(float))); // single float
        CSC(cudaMalloc((void**)&ns->d_time_tmp[g], sizeof(double)*ns->nblocks_reduce));

        CSC(cudaMemset(ns->d_r[g], 0, d4_size));
        CSC(cudaMemset(ns->d_v[g], 0, d4_size));
        CSC(cudaMemset(ns->d_f[g], 0, ff_size));
        CSC(cudaMemset(ns->d_p[g], 0, pp_size));
        CSC(cudaMemset(ns->d_t[g], 0, d1_size));
        CSC(cudaMemset(ns->d_i[g], 0, pp_size));
        CSC(cudaMemset(ns->d_dt[g], 0, d1_size));

        CSC(cudaMemset(ns->d_a2[g], 0, d3_size));
        CSC(cudaMemset(ns->d_a3[g], 0, d3_size));
        CSC(cudaMemset(ns->d_old[g], 0, ff_size));

        CSC(cudaMemset(ns->d_ekin[g], 0, d1_size));
        CSC(cudaMemset(ns->d_epot[g], 0, d1_size));
        CSC(cudaMemset(ns->d_move[g], 0, i1_size));
        CSC(cudaMemset(ns->d_fout[g], 0, ff_size * NJBLOCK));
        CSC(cudaMemset(ns->d_fout_tmp[g], 0, ff_size * NJBLOCK));

        CSC(cudaMemset(ns->d_nact[g], 0, sizeof(unsigned int)));
        CSC(cudaMemset(ns->d_max_mass[g], 0, sizeof(float)));
        CSC(cudaMemset(ns->d_time_tmp[g], 0, sizeof(double)*ns->nblocks_reduce));

        ns->h_fout_gpu[g] = new Forces[ns->n*NJBLOCK];
    }

    // Extra CPU array
    ns->h_fout_tmp = new Forces[ns->n*NJBLOCK];
}

/** Method in charge of deallocating the data structures on the available GPUs.
 */
void Hermite4GPU::free_arrays_device()
{

    for(int g = 0; g < gpus; g++)
    {
        // Setting GPU
        CSC(cudaSetDevice(g));

        CSC(cudaFree(ns->d_r[g]));
        CSC(cudaFree(ns->d_v[g]));
        CSC(cudaFree(ns->d_f[g]));
        CSC(cudaFree(ns->d_p[g]));
        CSC(cudaFree(ns->d_t[g]));
        CSC(cudaFree(ns->d_i[g]));
        CSC(cudaFree(ns->d_dt[g]));

        CSC(cudaFree(ns->d_a2[g]));
        CSC(cudaFree(ns->d_a3[g]));
        CSC(cudaFree(ns->d_old[g]));

        CSC(cudaFree(ns->d_ekin[g]));
        CSC(cudaFree(ns->d_epot[g]));
        CSC(cudaFree(ns->d_move[g]));
        CSC(cudaFree(ns->d_fout[g]));
        CSC(cudaFree(ns->d_fout_tmp[g]));

        CSC(cudaFree(ns->d_nact[g]));
        CSC(cudaFree(ns->d_max_mass[g]));
        CSC(cudaFree(ns->d_time_tmp[g]));

        delete ns->h_fout_gpu[g];
    }

    delete ns->h_fout_tmp;
    //delete ns->h_fout_gpu;
}

/** Method in charge of the prediction step.
 * This can be use on the CPU (commented section) or on the GPUs.
 * The reason of having both reasons, is the improvement is not much for small
 * amount of particles.
 */
void Hermite4GPU::predicted_pos_vel(double ITIME)
{
    ns->gtime.prediction_ini = omp_get_wtime();

    // for(int g = 0; g < gpus; g++)
    // {
    //     CSC(cudaSetDevice(g));
    //     int shift = g*n_part[g-1];
    //     size_t ff_size = n_part[g] * sizeof(Forces);
    //     size_t d4_size = n_part[g] * sizeof(double4);
    //     size_t d1_size = n_part[g] * sizeof(double);
    //
    //     // all already there (make sure they make it there the first time)
    //     CSC(cudaMemcpyAsync(ns->d_f[g], ns->h_f + shift, ff_size, cudaMemcpyHostToDevice, 0));
    //     CSC(cudaMemcpyAsync(ns->d_r[g], ns->h_r + shift, d4_size, cudaMemcpyHostToDevice, 0));
    //     CSC(cudaMemcpyAsync(ns->d_v[g], ns->h_v + shift, d4_size, cudaMemcpyHostToDevice, 0));
    //     CSC(cudaMemcpyAsync(ns->d_t[g], ns->h_t + shift, d1_size, cudaMemcpyHostToDevice, 0));
    // }

    // Executing kernels
    for(int g = 0; g < gpus; g++)
    {
        CSC(cudaSetDevice(g));

        nthreads = BSIZE;
        nblocks = std::ceil(n_part[g]/(float)nthreads);

        k_prediction <<< nblocks, nthreads >>> (ns->d_f[g],
                                                ns->d_r[g],
                                                ns->d_v[g],
                                                ns->d_t[g],
                                                ns->d_p[g],
                                                n_part[g],
                                                ITIME);
        get_kernel_error();
    }

    // for(int g = 0; g < gpus; g++)
    // {
    //     CSC(cudaSetDevice(g));
    //     size_t slice = g*n_part[g-1];
    //     size_t pp_size = n_part[g] * sizeof(Predictor);
    //
    //     CSC(cudaMemcpyAsync(&ns->h_p[slice], ns->d_p[g], pp_size, cudaMemcpyDeviceToHost, 0));
    // }

    ns->gtime.prediction_end += omp_get_wtime() - ns->gtime.prediction_ini;
}

/** Method in charge of the corrector step.
 * This is not implemented on the GPU because the benefit was not much
 * for small amount of particles.
 */
void Hermite4GPU::correction_pos_vel(double ITIME, unsigned int nact)
{
    // Timer
    ns->gtime.correction_ini = omp_get_wtime();

    // Executing kernels
    for (int g = 0; g < gpus; g++)
    {
      CSC(cudaSetDevice(g));

      nthreads = BSIZE;
      // nblocks = std::ceil(n_part[g]/(float)nthreads);
      nblocks = std::ceil(nact/(float)nthreads); // nact, since this is only doing that many iterations

      k_correction <<< nblocks, nthreads >>> (ns->d_move[g],
                                              ns->d_f[g],
                                              ns->d_old[g],
                                              ns->d_p[g],
                                              ns->d_r[g],
                                              ns->d_v[g],
                                              ns->d_t[g],
                                              ns->d_dt[g],
                                              ns->d_a2[g],
                                              ns->d_a3[g],
                                              nact,
                                              ITIME,
                                              ns->eta);
      get_kernel_error();
    }

    for(int g = 0; g < gpus; g++)
    {
        // t, dt, a2, a3
        CSC(cudaSetDevice(g));
        size_t slice = g*n_part[g-1];
        size_t d1_size = n_part[g] * sizeof(double);
        size_t d4_size = n_part[g] * sizeof(double4);

        CSC(cudaMemcpy(&ns->h_t[slice], ns->d_t[g], d1_size, cudaMemcpyDeviceToHost));
        CSC(cudaMemcpy(&ns->h_dt[slice], ns->d_dt[g], d1_size, cudaMemcpyDeviceToHost));
        CSC(cudaMemcpy(&ns->h_r[slice], ns->d_r[g], d4_size, cudaMemcpyDeviceToHost));
        // CSC(cudaMemcpyAsync(&ns->h_v[slice], ns->d_v[g], d4_size, cudaMemcpyDeviceToHost, 0)); // not needed by every iteration
    }

    ns->gtime.correction_end += omp_get_wtime() - ns->gtime.correction_ini;
}

/** Method in charge of the initialization of all the particle's acceleration
 * and first derivative of the system, at the begining of the simulation.
 */
void Hermite4GPU::init_acc_jrk()
{
    size_t pp_size = ns->n * sizeof(Predictor);

    // Copying arrays to device
    #pragma omp parallel for num_threads(gpus)
    for(int g = 0; g < gpus; g++)
    {
        CSC(cudaSetDevice(g));

        // All this information from the predictors is needed by each device
        CSC(cudaMemcpy(ns->d_p[g], ns->h_p, pp_size, cudaMemcpyHostToDevice));
        // CSC(cudaMemcpyAsync(ns->d_p[g], ns->h_p, pp_size, cudaMemcpyHostToDevice, 0));
    }

    // Executing kernels
    for(int g = 0; g < gpus; g++)
    {
        CSC(cudaSetDevice(g));

        nthreads = BSIZE;
        nblocks = std::ceil(n_part[g]/(float)nthreads);

        k_init_acc_jrk <<< nblocks, nthreads, smem >>> (ns->d_p[g],
                                                        ns->d_f[g],
                                                        ns->n,
                                                        ns->e2,
                                                        g,
                                                        n_part[g]);
        get_kernel_error();
    }

    for(int g = 0; g < gpus; g++)
    {
        CSC(cudaSetDevice(g));

        size_t chunk = n_part[g]*sizeof(Forces);
        size_t slice = g*n_part[g-1];

        CSC(cudaMemcpy(&ns->h_f[slice], ns->d_f[g], chunk, cudaMemcpyDeviceToHost));
        //CSC(cudaMemcpyAsync(&ns->h_f[slice], ns->d_f[g], chunk, cudaMemcpyDeviceToHost, 0));
    }
}

/** Method in charge of the force interaction between \f$N_{act}\f$ and the whole
 * system.
 *  First there is a tmp construction of predictors to be send to the GPUs.
 *  Then the data is copied to the devices.
 *  The first kernel perform the preliminary calculation of the forces in JPBLOCKS.
 *  The second kernel, reduction, is in charge of summing all the preliminary forces
 *  to the final value for all the active particles.
 */
void Hermite4GPU::update_acc_jrk(unsigned int nact)
{
    // Timer begin
    ns->gtime.update_ini = omp_get_wtime();

    ns->gtime.grav_ini = omp_get_wtime();
    for(int g = 0; g < gpus; g++)
    {
        if (n_part[g] > 0)
        {
          CSC(cudaSetDevice(g));

          /**
          As best I can tell:
          smem is a size_t argument to instruct the kernel how much shared memory to give each block.
          Shared memory only works for a single block, so no sense making it larger.
          Not sure what's up with smem_reduce and +1, haven't gotten that far yet.
            smem = sizeof(Predictor) * BSIZE;
            smem_reduce = sizeof(Forces) * NJBLOCK + 1;
          So the smem is for a Predictor array, one block's worth of Predictors.

          The kernel grid/block setup is
          nact_blocks is the number of blocks needed for nact, so that one move particle is on each thread.
          (nact + BSIZE - 1) / BSIZE = 1 + (nact-1)/BSIZE
          so basically rounded up number of blocks to fit nact on the threads.
          Block size BSIZE is already set to 32 in common.hpp, and that's good because there are still 32 threads per warp nowadays.

          Then the J dimension (.y in grid/block parlance) is NJBLOCK, which I think is artificially set to 16 but could be set larger depending on how many total blocks (threads) are available.
          Basically, you have 1 thread per move particle (nact threads) in the x direction, and then 16 particles in the y direction (J).
          The blocks are basically 32-element long rows arranged mostly side by size (unless nact is short) and then stacked up 16 times, so that no matter the x length, the y length is 16 blocks.
          y length in threads is also 16, because y length of block is 1 thread, but those 16 y threads are each in a different block and so don't have shared memory.

          Each particle is given 16 threads.

          Diving into the k_update kernel,
          int ibid = blockIdx.x;
          int jbid = blockIdx.y;
          int tid  = threadIdx.x;
          int iaddr  = tid + blockDim.x * ibid;
          int jstart = (n * (jbid  )) / NJBLOCK;
          int jend   = (n * (jbid+1)) / NJBLOCK;
          so iaddr is the x address and jbid is the y address (since only 1 thread in y per block).
          iaddr is used to grab the i particle which is "moved" this iteration.

          I need to spend more time thinking about jstart and jend and that for loop.
          If we divide n into NJBLOCK sections, jstart and jend would delineate the jth section.
          Like if n were 1600 and NJBLOCK 16, then there's steps of 100, so jstart and jend
          are 0,100 for j=0, 100,200 for j=1, 1500,1600 for j=15, etc.

          Each iaddr is a "move" particle.
          Each j block row does a different section of other particles (100-200, etc)
          Each BSIZE step loop makes shared memory and stores the next BSIZE particles.
          Then we iterate thru that BSIZE within the BSIZE step loop and each
          j block row uses the same j particle for that step, hence the shared memory.
          Every 32 (BSIZE) steps we dump the shared memory and refill it for the next step.

          So that outer BSIZE is only for memory management and the inner one actually runs
          thru every particle. Each j row will deal with a subet of n and each thread will
          go thru that n. It all checks out!

          **/

          int  nact_blocks;

          // if (nact < BSIZE) { // actual command
          if (nact > 0) { // dummy condition, only run k_update_smallnact
            // For a number of particles < BSIZE (32), use threads more optimally
            // There are still a lot of calculations so using threads efficiently can save time.
            // It is not uncommon for entire calls to update_acc_jrk to have nact < 10
            nvtxRangePushA("block per particle (new method)");
            nact_blocks = nact;
            dim3 nblocks(nact_blocks, NJBLOCK, 1);
            dim3 nthreads(BSIZE, 1, 1);
            // size_t smem_smallnact = sizeof(Forces) * BSIZE;
            // k_update_smallnact <<< nblocks, nthreads, smem_smallnact >>> (ns->d_move[g],  // we don't need/use the dynamically allocated shared memory
            k_update_smallnact <<< nblocks, nthreads >>> (ns->d_move[g],
                                                          ns->d_p[g],
                                                          ns->d_fout[g],
                                                          n_part[g],
                                                          nact,
                                                          ns->e2);
            nvtxRangePop();

          } else {
            // Use the regular way where each thread x is a particle
            // Single kernel launch is cheaper even if a few blockIdx.x > 0 blocks are almost empty
            // Blocks, threads and shared memory configuration
            nvtxRangePushA("thread per particle (original method)");
            nact_blocks = 1 + (nact-1)/BSIZE;
            dim3 nblocks(nact_blocks, NJBLOCK, 1);
            dim3 nthreads(BSIZE, 1, 1);

            // Kernel to update the forces for the particles in d_i
            // k_update <<< nblocks, nthreads, smem >>> (ns->d_move[g], // we don't need/use the dynamically allocated shared memory
            k_update <<< nblocks, nthreads >>> (ns->d_move[g],
                                                ns->d_p[g], // now full predictor array; got rid of second predictor arg bc it would be a duplicate now
                                                ns->d_fout[g], // size is ff_size * NJBLOCK
                                                n_part[g], // former N
                                                nact,
                                                ns->e2);


            nvtxRangePop();
          }

        }
    }

    ns->gtime.grav_end += omp_get_wtime() - ns->gtime.grav_ini;

    ns->gtime.reduce_ini = omp_get_wtime();
    for(int g = 0; g < gpus; g++)
    {
        size_t chunk = 2<<14;
        if (n_part[g] > 0)
        {
            CSC(cudaSetDevice(g));
            // Blocks, threads and shared memory configuration for the reduction.
            if (nact <= chunk) // limit 32768
            {
                dim3 rgrid   (nact,   1, 1);
                dim3 rthreads(NJBLOCK, 1, 1);

                // Kernel to reduce que temp array with the forces
                k_reduce <<< rgrid, rthreads, smem_reduce >>>(ns->d_fout[g],
                                                            ns->d_fout_tmp[g],
                                                            0,
                                                            0);
            }
            else
            {

                int smax = std::ceil(nact/(float)chunk);
                unsigned int shift = 0;
                size_t size_launch = 0;

                for(unsigned int s = 0; shift < nact; s++)
                {
                    // shift_id : s
                    // shift: moving pointer
                    // size_launch: chunk to multipy by Forces size
                    if (nact < shift + chunk)
                        size_launch = nact-shift;
                    else
                        size_launch = chunk;

                    dim3 rgrid   (size_launch,   1, 1);
                    dim3 rthreads(NJBLOCK, 1, 1);
                    k_reduce <<< rgrid, rthreads, smem_reduce >>>(ns->d_fout[g],
                                                                  ns->d_fout_tmp[g]+shift,
                                                                  s,
                                                                  shift);


                    shift += chunk;
                }
            }
        }
    }
    ns->gtime.reduce_end += omp_get_wtime() - ns->gtime.reduce_ini;

    // Update forces in the host
    ns->gtime.reduce_forces_ini = omp_get_wtime();

    int g = 0; // g can only be 0 here. Only use 1 GPU for this.
    if (gpus > 1) {
      char gpu_msg[128];
      sprintf(gpu_msg, "GPUS (%d) GTR THAN 1!!!!!", gpus);
      logger->log(1, gpu_msg);
    }
    CSC(cudaSetDevice(g));

    nthreads = BSIZE;
    nblocks = std::ceil(nact/(float)nthreads);

    k_assign_forces <<< nblocks, nthreads >>> (ns->d_move[g],
                                               ns->d_fout_tmp[g],
                                               ns->d_f[g],
                                               nact);
    get_kernel_error();


    ///// CPU reduce forces across GPUs. Updates big F array with small subset F array
    ///// The zero initialization is because it assumes multiple GPUs and so must add.
    //// We must assume 1 GPU for now.

    // #pragma omp parallel for
    // for (int i = 0; i < nact; i++)
    // {
    //     int id = ns->h_move[i];
    //     ns->h_f[id].a[0] = 0.0;
    //     ns->h_f[id].a[1] = 0.0;
    //     ns->h_f[id].a[2] = 0.0;
    //     ns->h_f[id].a1[0] = 0.0;
    //     ns->h_f[id].a1[1] = 0.0;
    //     ns->h_f[id].a1[2] = 0.0;
    //
    //     for(int g = 0; g < gpus; g++)
    //     {
    //         if (n_part[g] > 0)
    //         {
    //             ns->h_f[id] += ns->h_fout_gpu[g][i];
    //         }
    //     }
    // }

    // for(int g = 0; g < gpus; g++)
    // {
    //     if (n_part[g] > 0)
    //     {
    //         CSC(cudaSetDevice(g));
    //
    //         size_t slice = g*n_part[g-1];
    //         size_t ff_size = n_part[g] * sizeof(Forces);
    //
    //         // Copy from the GPU all forces, which have been updated
    //         // old copy:
    //         //CSC(cudaMemcpy(ns->h_fout_gpu[g], ns->d_fout_tmp[g], chunk, cudaMemcpyDeviceToHost));
    //     }
    // }


    ns->gtime.reduce_forces_end += omp_get_wtime() - ns->gtime.reduce_forces_ini;

    // Timer end
    ns->gtime.update_end += (omp_get_wtime() - ns->gtime.update_ini);
}


/** Method in charge of saving the old values of the acceleration and
 * its first derivative to be use in the Corrector integration step
 */
void Hermite4GPU::save_old_acc_jrk_gpu(unsigned int nact)
{

  /// Send over the move array and let the GPU pull out that data.
  nvtxRangePushA("save_old memcpy");
  for(int g = 0; g < gpus; g++)
  {
      if (n_part[g] > 0)
      {
          CSC(cudaSetDevice(g));
          // Copy to the GPU (d_i) the preddictor host array (h_i)
          size_t chunk = nact * sizeof(unsigned int);
          // CSC(cudaMemcpyAsync(ns->d_i[g], ns->h_i, chunk, cudaMemcpyHostToDevice, 0));
          // CSC(cudaMemcpyAsync(ns->d_move[g], ns->h_move, chunk, cudaMemcpyHostToDevice, 0));
          CSC(cudaMemcpy(ns->d_move[g], ns->h_move, chunk, cudaMemcpyHostToDevice));
      }
  }
  nvtxRangePop();

  nvtxRangePushA("save_old k launch");
  // Executing kernels
  for(int g = 0; g < gpus; g++)
  {
      CSC(cudaSetDevice(g));

      nthreads = BSIZE;
      nblocks = std::ceil(nact/(float)nthreads); // only safe on 1 GPU

      k_save_old_acc_jrk <<< nblocks, nthreads >>> (ns->d_move[g],
                                                    ns->d_f[g], // input
                                                    ns->d_old[g], // output (written by this function)
                                                    nact);
      get_kernel_error();
  }
  nvtxRangePop();
  // No memory copying necessary in either direction
}

/** Method in charge of finding all the particles that need to be
 * updated on the following integration step.
 */
unsigned int Hermite4GPU::find_particles_to_move_gpu(double ITIME)
{
  // Pass a whole bunch of things into the kernel launch.
  // Launch on 1 block with BSIZE (32) threads
  // Skip the gpu loop, this is designed for 1 gpu
  int g = 0;
  nblocks = 1;
  nthreads = BSIZE;
  // size_t smem_find = ns->n * sizeof(unsigned int);
  CSC(cudaSetDevice(g));

  k_find_particles_to_move <<< nblocks, nthreads >>> (ns->d_move[g],
                                                      ns->d_r[g],
                                                      ns->d_t[g],
                                                      ns->d_dt[g],
                                                      ITIME,
                                                      2*std::numeric_limits<double>::epsilon(),
                                                      ns->n,
                                                      ns->max_mass,
                                                      ns->d_nact[g],
                                                      ns->d_max_mass[g]);

  get_kernel_error();
  unsigned int nact_result;

  CSC(cudaMemcpy(&nact_result, ns->d_nact[g], sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CSC(cudaMemcpy(&ns->max_mass, ns->d_max_mass[g], sizeof(float), cudaMemcpyDeviceToHost));
  size_t chunk = nact_result * sizeof(unsigned int);
  CSC(cudaMemcpyAsync(ns->h_move, ns->d_move[g], chunk, cudaMemcpyDeviceToHost, 0));

  return nact_result;

}

/** Find the next integration time.
Minimum of the (current + next) times of all particles.
This is a big reduce algorithm, so runs fastest on powers of 2.
**/
void Hermite4GPU::next_integration_time_gpu(double &CTIME)
{
  // ns->log2n stores the precalculated log2(n)
  nthreads = BSIZE_LARGE;
  nblocks = ns->nblocks_reduce;

  int g = 0; // Only 1 GPU
  CSC(cudaSetDevice(g));

  k_time_min_reduce <<< nblocks, nthreads >>> (ns->d_t[g], ns->d_dt[g], ns->d_time_tmp[g]);

  if (nblocks > 1) {
    // final reduce, since n is larger than 1024
    // For now, assume that the number of elements remaining is a power of 2 between 2 and 512, inclusive
    nthreads = ns->nblocks_reduce/2; // do one thread per two previous blocks (double load)
    nblocks = 1;
    size_t smem_reduce_time = sizeof(double) * nthreads; // shared memory to make fast reduce
    k_time_min_reduce_final <<< nblocks, nthreads, smem_reduce_time >>> (ns->d_time_tmp[g]);
  }
  // if nblocks == 1, then that was the final round of reduction.

  CSC(cudaMemcpy(&CTIME, ns->d_time_tmp[g], sizeof(double), cudaMemcpyDeviceToHost));

}


/**
We have to transfer some data into the GPU once, at the beginning of the simulation.
Necessary now because we are removing many of the unnecessary data movements.
**/
void Hermite4GPU::initial_data_transfer()
{
  for(int g = 0; g < gpus; g++)
  {
      CSC(cudaSetDevice(g));
      int shift = g*n_part[g-1];
      size_t d1_size = n_part[g] * sizeof(double);
      size_t d4_size = n_part[g] * sizeof(double4);
      size_t d3_size = n_part[g] * sizeof(double3);
      size_t ff_size = n_part[g] * sizeof(Forces);
      size_t pp_size = n_part[g] * sizeof(Predictor);
      size_t i1_size = n_part[g] * sizeof(unsigned int);

      CSC(cudaMemcpyAsync(ns->d_t[g], ns->h_t + shift, d1_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_dt[g], ns->h_dt + shift, d1_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_r[g], ns->h_r + shift, d4_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_v[g], ns->h_v + shift, d4_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_f[g], ns->h_f + shift, ff_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_old[g], ns->h_old + shift, ff_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_p[g], ns->h_p + shift, pp_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_a2[g], ns->h_a2 + shift, d3_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_a3[g], ns->h_a3 + shift, d3_size, cudaMemcpyHostToDevice, 0));
      CSC(cudaMemcpyAsync(ns->d_move[g], ns->h_move + shift, i1_size, cudaMemcpyHostToDevice, 0));


  }
}

/**
Data only needed on writeout snapshots
**/
void Hermite4GPU::snapshot_data_transfer()
{
  for(int g = 0; g < gpus; g++)
  {
      CSC(cudaSetDevice(g));
      size_t slice = g*n_part[g-1];
      size_t ff_size = n_part[g] * sizeof(Forces);
      size_t d4_size = n_part[g] * sizeof(double4);
      // size_t ff_size = n_part[g] * sizeof(Forces);

      CSC(cudaMemcpyAsync(&ns->h_f[slice], ns->d_f[g], ff_size, cudaMemcpyDeviceToHost, 0)); // only needed on writeout
      // CSC(cudaMemcpyAsync(&ns->h_r[slice], ns->d_r[g], d4_size, cudaMemcpyDeviceToHost, 0)); // only needed on writeout
      CSC(cudaMemcpyAsync(&ns->h_v[slice], ns->d_v[g], d4_size, cudaMemcpyDeviceToHost, 0)); // only needed on writeout
  }

}

/** Method in charge of calculating the potential and kinetic energy
 * on the GPU devices
 */
double Hermite4GPU::get_energy_gpu()
{
    double time_energy_ini = omp_get_wtime();

    //// This was loading in a bad version of v from the CPU while the GPU had the good one
    // size_t d4_size = ns->n * sizeof(double4);
    //
    // for(int g = 0; g < gpus; g++)
    // {
    //     CSC(cudaSetDevice(g));
    //
    //     // CSC(cudaMemcpyAsync(ns->d_r[g], ns->h_r, d4_size, cudaMemcpyHostToDevice, 0)); // correction already copies this
    //     CSC(cudaMemcpyAsync(ns->d_v[g], ns->h_v, d4_size, cudaMemcpyHostToDevice, 0));
    // }

    int nthreads = BSIZE;
    for(int g = 0; g < gpus; g++)
    {
        CSC(cudaSetDevice(g));

        int nblocks = std::ceil(n_part[g]/(float)nthreads);
        k_energy <<< nblocks, nthreads >>> (ns->d_r[g],
                                            ns->d_v[g],
                                            ns->d_ekin[g],
                                            ns->d_epot[g],
                                            ns->n,
                                            n_part[g],
                                            g);
    }

    for(int g = 0; g < gpus; g++)
    {
        CSC(cudaSetDevice(g));

        size_t chunk = n_part[g]*sizeof(double);
        size_t slice = g*n_part[g-1];

        CSC(cudaMemcpyAsync(&ns->h_ekin[slice], ns->d_ekin[g], chunk, cudaMemcpyDeviceToHost, 0));
        CSC(cudaMemcpyAsync(&ns->h_epot[slice], ns->d_epot[g], chunk, cudaMemcpyDeviceToHost, 0));
    }

    // Reduction on CPU
    ns->en.kinetic = 0.0;
    ns->en.potential = 0.0;
    for (int i = 0; i < ns->n; i++)
    {
        ns->en.kinetic   += ns->h_ekin[i];
        ns->en.potential += ns->h_epot[i];
    }

    double time_energy_end = omp_get_wtime() - time_energy_ini;

    return ns->en.kinetic + ns->en.potential;
}

/** Method that get the last kernel error if the code is running with the DEBUG
 * flag
 */
void Hermite4GPU::get_kernel_error()
{
    #ifdef KERNEL_ERROR_DEBUG
    logger->log_error(std::string(cudaGetErrorString(cudaGetLastError())));
    #endif
}

/** Method to start the device timer
 */
void Hermite4GPU::gpu_timer_start(){
    cudaEventRecord(start);
}

/** Method that ends the device timer
 */
float Hermite4GPU::gpu_timer_stop(std::string f){
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    #if KERNEL_TIME
    if (f != "")
    {
        std::string s = "";
        s += std::string("Kernel ");
        s += std::string(SSTR(f));
        s += std::string(" : ");
        s += std::string(SSTR(msec));
        logger->log_info(s)
    }
    #endif
    return msec;
}

/** This method is not implemented becasue we use a CUDA kernel
 * to perfom the force calculation, not a host method.
 */
void Hermite4GPU::force_calculation(const Predictor &pi, const Predictor &pj, Forces &fi) {}


/** Find log2(n) **/
void Hermite4GPU::save_log2n()
{
  // unsigned int n = ns->n;
  unsigned int s = 0;
  while (((ns->n)>>s) > 1) {
    // n >>= 1;
    s++;
  }
  ns->log2n = s;
  // Also save the nblocks_reduce number, which is ns->n/1024 (assuming BSIZE_LARGE = 512)
  ns->nblocks_reduce = 1<<(ns->log2n - LOG2_BSIZE_LARGE - 1);
}
