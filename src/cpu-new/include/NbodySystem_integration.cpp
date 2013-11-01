#include "NbodySystem.hpp"

void NbodySystem::integration(Hermite4 h, Logger log)
{

    double ATIME = 1.0e+10; // Actual integration time
    double ITIME = 0.0;     // Integration time
    int nact     = 0;       // Active particles
    int nsteps   = 0;       // Amount of steps per particles on the system
    static long long interactions = 0;

    log.print_info(n, e2, eta, integration_time);

    //int max_threads = omp_get_max_threads();
    //omp_set_num_threads( max_threads - 1);

    h.init_acc_jrk(h_p, h_f);     // Initial calculation of a and a1
    h.init_dt(ATIME, h_dt, h_t, h_f);  // Initial calculation of time-steps using simple equation

    en.ini = get_energy();   // Initial calculation of the energy of the system
    en.tmp = en.ini;

    log.print_energy_log(ITIME, iterations, interactions, nsteps, gtime, en, en.ini);
    //log.print_all(ITIME, n, h_r, h_v, h_f, h_dt);

    while (ITIME < integration_time)
    {
        ITIME = ATIME;

        nact = h.find_particles_to_move(h_move, ITIME, h_dt, h_t);


        h.save_old_acc_jrk(nact, h_move, h_old, h_f);

        h.predicted_pos_vel(ITIME, h_p, h_r, h_v, h_f, h_t, gtime);

        h.update_acc_jrk(nact, h_move, h_p, h_f, gtime);

        h.correction_pos_vel(ITIME, nact, h_move, h_r, h_v, h_f, h_t, h_dt, h_p, h_old, h_a3, h_a2, gtime);

        // Update the amount of interactions counter
        interactions += nact * n;

        //if(std::ceil(ITIME) == ITIME)
        if(nact == n)
        {
            log.print_energy_log(ITIME, iterations, interactions, nsteps, gtime, en, get_energy());
            log.print_all(ITIME, n, h_r, h_v, h_f, h_dt);

        }

        // Find the next integration time
        h.next_integration_time(ATIME, h_dt, h_t);

        // Update nsteps with nact
        nsteps += nact;

        // Increase iteration counter
        iterations++;
    }
}
