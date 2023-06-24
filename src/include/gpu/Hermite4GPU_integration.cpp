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
#include "Hermite4GPU.cuh"

void Hermite4GPU::integration()
{

    ns->gtime.integration_ini = omp_get_wtime();

    double ATIME = 1.0e+10; // Actual integration time
    double ITIME = ns->snapshot_time;     // Integration time
    unsigned int nact     = 0;       // Active particles
    unsigned int nsteps   = 0;       // Amount of steps per particles on the system
    static long long interactions = 0;
    unsigned int output_factor = 1;

    // Setting maximum number of threads for OpenMP sections
    omp_set_num_threads(omp_get_max_threads());

    init_acc_jrk();
    init_dt(ATIME, ETA_S, ITIME);

    nvtxRangePushA("initial_data_transfer");
    initial_data_transfer();
    nvtxRangePop();

    // Initial energy calculation
    ns->en.ini = get_energy_gpu();   // Initial calculation of the energy of the system
    ns->en.tmp = ns->en.ini;


    // Getting system information:
    nu->nbody_attributes();

    logger->print_info();
    logger->write_info();
    logger->print_energy_log(ITIME, ns->iterations, interactions, nsteps, ns->en.ini);

    unsigned int snap_number = ns->snapshot_number;
    logger->write_snapshot(snap_number, ITIME);
    snap_number++;

    if (ns->ops.print_all)
    {
        logger->print_all(ITIME, snap_number);
    }
    if (ns->ops.print_lagrange)
    {
        nu->lagrange_radii();
        logger->print_lagrange_radii(ITIME, nu->layers_radii);
    }

    while (ITIME < ns->integration_time)
    {
        // Pushing the epoch (iteration) number to profiler
        // the, convert string to char* to use it in nvtxRangePushA
        std::string epoch = "Epoch " + std::to_string(ns->iterations);
        char const *epoch_char = epoch.c_str();
        nvtxRangePushA(epoch_char);

        ITIME = ATIME;

        nvtxRangePushA("find_particles_to_move");
        nact = find_particles_to_move(ITIME);
        nvtxRangePop();

        save_old_acc_jrk_gpu(nact);

        nvtxRangePushA("predicted_pos_vel");
        predicted_pos_vel(ITIME);
        nvtxRangePop();

        nvtxRangePushA("update_acc_jrk");
        update_acc_jrk(nact);
        nvtxRangePop();

        nvtxRangePushA("correction_pos_vel");
        correction_pos_vel(ITIME, nact);
        nvtxRangePop();

        // Update the amount of interactions counter
        interactions += nact * ns->n;

        // Find the next integration time
        next_integration_time(ATIME);


        if (ITIME >= ns->interval_time * output_factor)
        {
            // Transfer over whatever it needs (get_energy_gpu does complementary transfers)
            snapshot_data_transfer();
            logger->print_energy_log(ITIME, ns->iterations, interactions, nsteps, get_energy_gpu());
            if (ns->ops.print_all)
            {
                logger->print_all(ITIME, snap_number);
            }
            if (ns->ops.print_lagrange)
            {
                nu->lagrange_radii();
                logger->print_lagrange_radii(ITIME, nu->layers_radii);
            }
            if (ITIME < ns->integration_time)
            {
                logger->write_snapshot(snap_number, ITIME);
                snap_number++;
            }
            output_factor += 1;
        }

        // Update nsteps with nact
        nsteps += nact;

        // Increase iteration counter
        ns->iterations++;

        nvtxRangePop();
    }

    snapshot_data_transfer(); // send over everything needed for writeout
    ns->gtime.integration_end =  omp_get_wtime() - ns->gtime.integration_ini;
    logger->write_snapshot(snap_number, ITIME);
    //logger->add_info(std::string("SnapshotNumber:"), std::to_string(snap_number));
    logger->add_info(std::string("SnapshotNumber:"), std::string(SSTR(snap_number)));
}
