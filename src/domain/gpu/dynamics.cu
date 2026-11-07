// The steps of velocity Verlet on the degrees of freedom of a solid, with the coordinates,
// velocities and accelerations kept on the device.
//
// REAL is the precision, which is defined when the kernels are compiled. The kernels do what the
// integrator of the library does, in the same order, and the masses of held degrees of freedom
// are infinite so that they do not accelerate.

// The first half of a step: the velocity a half step on, and the coordinates a step on with it.
extern "C" __global__ void kick_drift(
    const unsigned int count,
    REAL *__restrict__ coordinates,
    REAL *__restrict__ velocities,
    const REAL *__restrict__ accelerations,
    const REAL dt)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const REAL half_velocity = accelerations[i] * (dt * (REAL)0.5) + velocities[i];
    velocities[i] = half_velocity;
    coordinates[i] = half_velocity * dt + coordinates[i];
}

// Holds the coordinates that are moved at what they are held at: the coordinate dofs[i] is set to
// table[row * count + i], for each of the count of them.
extern "C" __global__ void set_targets(
    const unsigned int count,
    const unsigned int *__restrict__ dofs,
    const REAL *__restrict__ table,
    const unsigned int row,
    REAL *__restrict__ coordinates)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    coordinates[dofs[i]] = table[(size_t)row * count + i];
}

// Subtracts a force on each of count degrees of freedom, which does not change, from the forces
// on them, as the external forces are subtracted from the forces of the elements.
extern "C" __global__ void subtract_constant(
    const unsigned int count,
    const REAL *__restrict__ loads,
    REAL *__restrict__ forces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    forces[i] -= loads[i];
}

// Subtracts a force on each of nodes nodes that is a weight of the node times three components
// that are a function of time, from the forces on them: the components are table[row * 3 + c].
extern "C" __global__ void subtract_weighted(
    const unsigned int nodes,
    const REAL *__restrict__ weights,
    const REAL *__restrict__ table,
    const unsigned int row,
    REAL *__restrict__ forces)
{
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes)
    {
        return;
    }
    const REAL weight = weights[node];
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        forces[3 * (size_t)node + c] -= weight * table[(size_t)row * 3 + c];
    }
}

// The accelerations that balance the forces, which are then cleared for the next step, and
// the velocity another half step on when the step is not the first evaluation.
extern "C" __global__ void accelerate(
    const unsigned int count,
    REAL *__restrict__ forces,
    const REAL *__restrict__ masses,
    REAL *__restrict__ accelerations,
    REAL *__restrict__ velocities,
    const REAL dt,
    const unsigned int kick)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const REAL acceleration = -forces[i] / masses[i];
    forces[i] = 0;
    accelerations[i] = acceleration;
    if (kick)
    {
        velocities[i] = acceleration * (dt * (REAL)0.5) + velocities[i];
    }
}
