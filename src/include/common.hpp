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

#ifndef COMMON_HPP
#define COMMON_HPP

#if defined(_MPI)
#include <mpi.h>
#endif

#include <iomanip>
#include <iostream>
#include <ctime>
#include <cstdio>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "nvToolsExt.h"

#if defined(_MPI)
#define MPI_NUM_SLAVES 600
#endif

#if defined(GPU)
/** If we are compiling the CUDA version, we add the definition of the
 * vector types and structs from the CUDA library */
#include <cuda_runtime.h>
#else
/** Defining the «double4» structure based on the CUDA definition for
 * the CPU version, which not include the CUDA headers */
typedef struct double4
{
    double x, y, z, w;
} double4;

/** Defining the «double3» structure based on the CUDA definition for
 * the CPU version, which not include the CUDA headers */
typedef struct double3
{
    double x, y, z;
} double3;
#endif

/** Gravitational constant. Since we are working in N-body units we
 * set G as one. */
const int G = 1;


/** Flag to enable the PN integration for two bodies. */
//#define PN 1

#ifdef PN
/** Speed of light in code units. This will vary with each simulation
 * scenario so it need to be changed to fit user needs */
const float SPEED_OF_LIGHT = 3.8241467907228306;
#endif

/** Amount of neighbours to calculate the center of density of the
 * system (Casertano & Hut 1985)*/
const int J = 6;

/** Color output */
const int COLOR_OUTPUT = 1;

/** Common mass percentage in the core of a globular cluster */
const float CORE_MASS = 0.1;

const float LAGRANGE_RADII[] = {0.01, 0.05, 0.1, 0.2, 0.5, 0.75};

/** Softening parameter */
const double E = 1e-4;

/** Softening parameter squared */
const double E2 = 1e-8;

/** Initial ETA parameter to calculate the first timestep of all the
 * particles of the system. Based on Aarseth formula */
const float ETA_S = 0.01;

/** Iteration ETA parameter to calculate new timestep of all the
 * active particles of the system, in a certain integration time.
 * Based on Aarseth formula */
const float ETA_N = 0.01;

/** Lower boundary for the particles timesteps, \f$2^{-23}\f$ */
const double D_TIME_MIN = 1.1920928955078125e-07;

/** Lower boundary for the binary timesteps, \f$2^{-30}\f$ */
//const double D_MTIME_MIN = 9.313225746154785e-10;
const double D_MTIME_MIN = 7.450580596923828e-09; // 2^-27

/** Upper boundary for the particles timesteps, \f$2^{-3}\f$ */
const double D_TIME_MAX = 0.125;

/** @struct Distance
 *  @brief Structure to handle the distance of the particles, to be
 *  able to identify them while sorted.
 *  @var Distance::index
 *  Member 'index' identification of the particle.
 *  @var Distance::value
 *  Member 'value' distance of the particle respect to a reference.
 *  @fn Distance::operator
 *  Member 'operator' special operator to sort the particles.
 */
struct Distance
{
    int index;
    double value;
    bool operator<(const Distance& a) const
    {
        return value < a.value;
    }
};

/** @struct Energy
 *  @brief This structure contains all the energy variables of the
 *  system.
 *  @var Energy::ini
 *  Member 'ini' contains the initial total energy of the system.
 *  @var Energy::ini
 *  Member 'ini' contains the initial total energy of the system.
 *  @var Energy::end
 *  Member 'end' contains the newest total energy of the system in a
 *  certain time.
 *  @var Energy::tmp
 *  Member 'tmp' contains the previous total total energy of the
 *  system.
 *  @var Energy::kinetic
 *  Member 'kinetic' contains the newest kinetic energy of the system
 *  in a certain time.
 *  @var Energy::potential
 *  Member 'potential' contains the newest kinetic energy of the
 *  system in a certain time.
 *  */
typedef struct Energy
{
    double ini;
    double end;
    double tmp;
    double kinetic;
    double potential;
} Energy;

/** @struct options
 *  @brief Options to handling printing options, like printing the
 *  snapshot on the screen instead of a file; print the informaton of
 *  all the particles (id, mass, position, velocity, acceleration,
 *  jerk, current timestep); calculating and printing the lagrange
 *  radii.
 *  @var options::print_screen
 *  Member 'print_screen' contains the boolean value of printing the
 *  snapshot on the screen (true) or a file (false).
 *  @var options::print_all
 *  Member 'print_all' contains the boolean value of printing the
 *  information of all the particles of the system.
 *  @var options::print_lagrange
 *  Member 'print_lagrange' contains the boolean value for calculating
 *  and printing the lagrange radii of the system.
 */
typedef struct options
{
    bool print_screen;
    bool print_all;
    bool print_lagrange;
} options;

/** @struct Predictor
 *  @brief This structure contains the predicted information of a
 *  particle in some moment of the integration.
 *  @var Predictor::r
 *  Member 'r' contains the position in three dimensions.
 *  @var Predictor::v
 *  Member 'v' contains the velocity in three dimensions.
 *  @var Predictor::m
 *  Member 'm' contains the mass of the particle.
 *  */
typedef struct Predictor {
    double r[3];
    double v[3];
    float  m;
} Predictor;

/** @struct Forces
 *  @brief This structure contains the information of the Forces of a
 *  particle in some moment of the integration.
 *  @var Forces::a
 *  Member 'a' contains the acceleration in three dimensions.
 *  @var Forces::a1
 *  Member 'v' contains the first derivative of the acceleration in
 *  three dimensions (Jerk).
 *  */
typedef struct Forces {
  #if defined(SINGLEFORCE)
    float a[3];
    float a1[3];
  #elif defined(MIXFORCE)
    double a[3];
    float a1[3];
  #else
    double a[3];
    double a1[3];
  #endif

} Forces;

/** @struct Gtime
 *  @brief This structure contains different times of the internal
 *  integration process.
 *  This times are calculated using the function omp_get_wtime() from
 *  the OpenMP library.
 *  @var Gtime::integration_ini
 *  Member 'integration_ini' contains the starting time of the
 *  integration.
 *  @var Gtime::integration_end
 *  Member 'integration_end' contains the final time of the
 *  integration.
 *  @var Gtime::prediction_ini
 *  Member 'prediction_ini' contains the starting time of the
 *  prediction.
 *  @var Gtime::prediction_end
 *  Member 'prediction_end' contains the final time of the prediction.
 *  @var Gtime::update_ini
 *  Member 'update_ini' contains the starting time of the forces
 *  update.
 *  @var Gtime::update_end
 *  Member 'update_end' contains the final time of the forces update.
 *  @var Gtime::correction_ini
 *  Member 'correction_ini' contains the starting time of the
 *  correction.
 *  @var Gtime::correction_end
 *  Member 'correction_end' contains the final time of the correction.
 *  @var Gtime::grav_ini
 *  Member 'grav_ini' contains the starting time of the gravitational
 *  interaction.
 *  @var Gtime::grav_end
 *  Member 'grav_end' contains the final time of the gravitational
 *  interaction.
 *  @var Gtime::reduce_ini
 *  Member 'reduce_ini' contains the starting time of the forces
 *  reduction.
 *  @var Gtime::reduce_end
 *  Member 'reduce_end' contains the final time of the forces
 *  reduction.
 *  @var Gtime::reduce_forces_ini
 *  Member 'reduce_ini' contains the starting time of the forces
 *  reduction on CPU.
 *  @var Gtime::reduce_forces_end
 *  Member 'reduce_end' contains the final time of the forces
 *  reduction on CPU.
 *  @var Gtime::gflops
 *  Member 'gflops' contains the amount of Giga FLOPs of the force
 *  update method.
 *  This is calculated with the following formula:
 *  \f$ 60.10e-9 \cdot \frac{1}{C_{\rm time}}\cdot \sum_{t=0}^{t=T}
 *      N_{\rm act} N \f$
 *  where \f$(N_{\rm act} N)\f$ is the amount of gravitational
 *  interactions, \f$C_{\rm time}\f$ the elapsed clock-time of the
 *  process, \f$T\f$ a determinated integration time.
 *  */
typedef struct Gtime {
    float integration_ini;
    float integration_end;

    float prediction_ini;
    float prediction_end;

    float update_ini;
    float update_end;

    float correction_ini;
    float correction_end;

    float grav_ini;
    float grav_end;

    float reduce_ini;
    float reduce_end;

    float reduce_forces_ini;
    float reduce_forces_end;

    float gflops;
} Gtime;

/** @struct file_data
 *  @brief General structure to read the INPUT file.
 *  @var file_data::id
 *  Member 'id' particle identification.
 *  @var file_data::m
 *  Member 'm' particle mass.
 *  @var file_data::r
 *  Member 'r' Array that contain the particle position.
 *  @var file_data::v
 *  Member 'v' Array that contain the particle velocity.
 *  */
typedef struct file_data
{
    int id;
    float m;
    double r[3];
    double v[3];
} file_data;

const double KERNEL_GFLOP = 48e-9; // 60e-9
#if defined(GPU)
const int BSIZE   = 32;
const int NJBLOCK = 16;
const int BSIZE_LARGE = 512; // for some of the reduce functions; 512 is current max block size by hardware limits
const int LOG2_BSIZE_LARGE = 9; // 2^9 = 512 // if you change BSIZE_LARGE, you HAVE to change this too
//#define KERNEL_ERROR_DEBUG 1
#endif

#endif
