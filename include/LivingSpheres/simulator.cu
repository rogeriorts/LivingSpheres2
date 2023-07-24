#include "simulator.h"
#include <stdexcept>
#include <stdio.h>
void Simulator::add_cell(double x,
                         double y,
                         double vx,
                         double vy,
                         double spring_coefficient,
                         double damping_ratio,
                         double density,
                         int wall_id)
{

    cell cell_;
    cell_.x = x;
    cell_.y = y;
    cell_.vx = vx;
    cell_.vy = vy;
    cell_.inverse_spring_coefficient = 1.0 / spring_coefficient;
    cell_.inverse_damping_ratio = 1.0 / damping_ratio;
    double cell_volume = 4.0 / 3.0 * M_PI * this->simulator_settings.radius * this->simulator_settings.radius * this->simulator_settings.radius;
    cell_.mass = density * cell_volume;
    cell_.number_of_contacts = 0;
    cell_.wall_id = wall_id;
    cell_.grid_block = -1;

    if( x > this->simulator_settings.x_max_domain - 0.1*this->simulator_settings.radius || 
        x < this->simulator_settings.x_min_domain + 0.1*this->simulator_settings.radius || 
        y > this->simulator_settings.y_max_domain - 0.1*this->simulator_settings.radius || 
        y < this->simulator_settings.y_min_domain + 0.1*this->simulator_settings.radius)
    {
        cell_.active = false;
    }
    else{
        cell_.active = true;
    }
    cells.push_back(cell_);
}

/*CONTACT DETECTION CODE*/

__host__ __device__ void detect_contact(
    cell *cells, grid_block *grid, int *grid_cell_ids,
    int home_id,
    int grid_nx, int grid_ny,
    int max_number_of_cells_per_grid_block,
    double radius)
{
    
    if (!cells[home_id].active ) 
        return;

    const int grid_k = cells[home_id].grid_block;
    const int grid_j = grid_k / grid_nx;
    const int grid_i = grid_k - grid_j * grid_nx;

    for (size_t grid_i_neighbor = max(0, grid_i - 1);
         grid_i_neighbor <= min(grid_nx - 1, grid_i + 1);
         grid_i_neighbor++)
    {
        for (int grid_j_neighbor = max(0, grid_j - 1);
             grid_j_neighbor <= min(grid_ny - 1, grid_j + 1);
             grid_j_neighbor++)
        {
            int k_neighbor = grid_i_neighbor + grid_nx * grid_j_neighbor;
            for (int c_b = 0; c_b <= grid[k_neighbor].last_position; c_b++)
            {
                const int near_id = grid_cell_ids[k_neighbor * max_number_of_cells_per_grid_block + c_b];

                const double diff_x = cells[near_id].x - cells[home_id].x;
                const double diff_y = cells[near_id].y - cells[home_id].y;
                if (home_id < near_id && cells[near_id].active && (cells[home_id].wall_id == -1 || cells[near_id].wall_id == -1))
                {
                    if (diff_x * diff_x + diff_y * diff_y < 4.0 * radius * radius)
                    {
                        
                        if (cells[home_id].number_of_contacts>=6)
                            return;

                        cells[home_id].number_of_contacts++;

                        cells[home_id].contacts[cells[home_id].number_of_contacts - 1] = near_id;
                    }
                }
            }
        }
    }
    
    
    

    return;
    
}

void contact_detection_CPU(std::vector<cell> &cells, std::vector<grid_block> &grid, int *grid_cell_ids, int grid_nx, int grid_ny, int max_number_of_cells_per_grid_block, double radius)
{
#pragma omp parallel for
    for (int home_id = 0; home_id < cells.size(); home_id++)
    {
        
        detect_contact(
            cells.data(), grid.data(), grid_cell_ids,
            home_id, grid_nx, grid_ny, max_number_of_cells_per_grid_block,
            radius);
    }
    return;
}

__global__ void contact_detection_GPU(cell *device_cells, size_t number_of_cells, grid_block *device_grid, int *device_grid_cell_ids, int grid_nx, int grid_ny, int max_number_of_cells_per_grid_block, double radius)
{
    int home_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (home_id < number_of_cells)
    {
        detect_contact(device_cells, device_grid, device_grid_cell_ids,
                       home_id, grid_nx, grid_ny, max_number_of_cells_per_grid_block,
                       radius);
    }
}

void Simulator::calculate_contacts()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (!this->simulator_settings.run_in_GPU)
    {
        contact_detection_CPU(
            this->cells,
            this->grid, this->grid_cell_ids, this->grid_nx,
            this->grid_ny, this->max_number_of_cells_per_grid_block,
            simulator_settings.radius);
    }
    else
    {
        contact_detection_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
            this->device_cells, cells.size(),
            this->device_grid, this->device_grid_cell_ids,
            this->grid_nx, this->grid_ny, this->max_number_of_cells_per_grid_block,
            simulator_settings.radius);

        cudaDeviceSynchronize();

        evaluate_cuda_error("Error on calculate_contacts", cudaGetLastError());
    }
    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_contact_detection += std::chrono::duration<double>(stop - start).count();

    return;
}

/*FORCES CALCULATION*/

__host__ __device__ void calculate_contact_force(cell *cells, int home_id, double radius, double friction_coefficient)
{
    for (size_t c = 0; c < cells[home_id].number_of_contacts; c++)
    {
        // calculate contact normal
        const int near_id = cells[home_id].contacts[c];

        const double dist_x = cells[near_id].x - cells[home_id].x;
        const double dist_y = cells[near_id].y - cells[home_id].y;

        const double distance = sqrt(dist_x * dist_x + dist_y * dist_y);
        const double x_normal = dist_x / distance;
        const double y_normal = dist_y / distance;

        // spring-dashpit coefficients
        double spring_coefficient = 2.0 / (cells[home_id].inverse_spring_coefficient +
                                           cells[near_id].inverse_spring_coefficient);

        double damping_coefficient = 2.0 * 2.0 / (cells[home_id].inverse_damping_ratio + cells[near_id].inverse_damping_ratio) *
                                     sqrt((cells[home_id].mass + cells[near_id].mass) * 0.5 * radius * spring_coefficient);

        // calculate relative_velocity
        const double vx_rel = cells[near_id].vx - cells[home_id].vx;
        const double vy_rel = cells[near_id].vy - cells[home_id].vy;

        // calculate normal and tangential velocities
        const double v_normal = vx_rel * x_normal + vy_rel * y_normal;
        const double v_tangential = vx_rel * y_normal + vy_rel * (-x_normal);

        // force calculation
        const double spring_force = (2 * radius - distance) * spring_coefficient;

        const double damping_force = -v_normal * damping_coefficient;

        double tangential_force = 0.0;

        if (fabs(v_tangential) > 1e-15)
        {
            tangential_force = -friction_coefficient * spring_force * v_tangential / fabs(v_tangential);
        }

        const double force_x = (-spring_force - damping_force) * x_normal - tangential_force * y_normal;
        const double force_y = (-spring_force - damping_force) * y_normal - tangential_force * (-x_normal);

#ifdef __CUDA_ARCH__

        atomicAdd(&cells[home_id].force_x, force_x);
        atomicAdd(&cells[home_id].force_y, force_y);

        atomicAdd(&cells[near_id].force_x, -force_x);
        atomicAdd(&cells[near_id].force_y, -force_y);
#else
#pragma omp atomic update
        cells[home_id].force_x += force_x;

#pragma omp atomic update
        cells[home_id].force_y += force_y;

#pragma omp atomic update
        cells[near_id].force_x -= force_x;

#pragma omp atomic update
        cells[near_id].force_y -= force_y;
#endif
    }
}

void force_calculation_CPU(std::vector<cell> &cells, double radius, double friction_coefficient)
{
#pragma omp parallel for
    for (int home_id = 0; home_id < cells.size(); home_id++)
    {
        calculate_contact_force(cells.data(), home_id, radius, friction_coefficient);
    }
    return;
}

__global__ void force_calculation_GPU(cell *device_cells, size_t number_of_cells, double radius, double friction_coefficient)
{
    int home_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (home_id < number_of_cells)
    {
        calculate_contact_force(device_cells, home_id, radius, friction_coefficient);
    }
}

void reset_contacts_and_forces_CPU(std::vector<cell> &cells)
{
#pragma omp parallel for
    for (int i = 0; i < cells.size(); i++)
    {
        cells[i].force_x = 0.0;
        cells[i].force_y = 0.0;
        cells[i].number_of_contacts = 0;
    }
    return;
}

__global__ void reset_contacts_and_forces_GPU(cell *device_cells, size_t number_of_cells)
{
    int cell_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (cell_id < number_of_cells)
    {
        device_cells[cell_id].force_x = 0.0;
        device_cells[cell_id].force_y = 0.0;
        device_cells[cell_id].number_of_contacts = 0;
    }
}

void Simulator::reset_contacts_and_forces()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (!this->simulator_settings.run_in_GPU)
    {
        reset_contacts_and_forces_CPU(this->cells);
    }
    else
    {
        reset_contacts_and_forces_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
            this->device_cells, cells.size());

        cudaDeviceSynchronize();

        evaluate_cuda_error("Error on reset_contacts_and_forces", cudaGetLastError());
    }

    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_reset_forces_and_contacts += std::chrono::duration<double>(stop - start).count();

    return;
}

void Simulator::calculate_forces()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (!this->simulator_settings.run_in_GPU)
    {
        force_calculation_CPU(this->cells,
                              this->simulator_settings.radius,
                              this->simulator_settings.friction_coefficient);
    }
    else
    {
        force_calculation_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
            this->device_cells, cells.size(),
            this->simulator_settings.radius,
            this->simulator_settings.friction_coefficient);

        cudaDeviceSynchronize();

        evaluate_cuda_error("Error on calculate_forces", cudaGetLastError());
    }

    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_force_calculation += std::chrono::duration<double>(stop - start).count();

    return;
}

/*POSITION CALCULATION*/

__host__ __device__ void update_positions_and_velocities(cell *cells, 
    int id, double time_step, double gravity, 
    double x_min, double x_max,
    double y_min, double y_max, double radius

)
{
    if (cells[id].wall_id > -1)
        return;

    const double acc_x = cells[id].force_x / cells[id].mass;
    const double acc_y = cells[id].force_y / cells[id].mass + gravity;

    cells[id].x += cells[id].vx * time_step + acc_x * time_step * time_step * 0.5;
    cells[id].y += cells[id].vy * time_step + acc_y * time_step * time_step * 0.5;
    
    if( cells[id].x > x_max - 0.1*radius || 
        cells[id].x < x_min + 0.1*radius ||
        cells[id].y > y_max - 0.1*radius || 
        cells[id].y < y_min + 0.1*radius)
    {
        cells[id].active = false;
    }

    cells[id].vx += acc_x * time_step;
    cells[id].vy += acc_y * time_step;

    return;
}

__global__ void update_positions_and_velocities_GPU(
    cell *device_cells, size_t number_of_cells, double timestep, double gravity, 
    double x_min, double x_max,
    double y_min, double y_max, double radius
    )
{
    int cell_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (cell_id < number_of_cells)
    {
        update_positions_and_velocities(device_cells, cell_id, timestep, gravity, 
        x_min,  x_max,
        y_min, y_max, radius);
    }
}

void update_positions_and_velocities_CPU(std::vector<cell> &cells, double time_step, double gravity, 
    double x_min, double x_max,
    double y_min, double y_max, double radius)
{

#pragma omp parallel for
    for (int home_id = 0; home_id < cells.size(); home_id++)
    {
        update_positions_and_velocities(cells.data(), home_id, time_step, gravity, 
        x_min, x_max,
        y_min, y_max, radius);
    }

    return;
}

void Simulator::update_position()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (!this->simulator_settings.run_in_GPU)
    {
        update_positions_and_velocities_CPU(this->cells, this->simulator_settings.time_step,
                                            this->simulator_settings.gravity,
                                            this->simulator_settings.x_min_domain,
                                            this->simulator_settings.x_max_domain,
                                            this->simulator_settings.y_min_domain,
                                            this->simulator_settings.y_max_domain,
                                            this->simulator_settings.radius
                                            );
    }
    else
    {

        update_number_of_blocks(cells.size());

        update_positions_and_velocities_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
            this->device_cells, cells.size(),
            this->simulator_settings.time_step,
            this->simulator_settings.gravity,
            this->simulator_settings.x_min_domain,
            this->simulator_settings.x_max_domain,
            this->simulator_settings.y_min_domain,
            this->simulator_settings.y_max_domain,
            this->simulator_settings.radius);

        cudaDeviceSynchronize();

        evaluate_cuda_error("Error on update_positions_and_velocities", cudaGetLastError());
    }

    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_move += std::chrono::duration<double>(stop - start).count();

    return;
}

void Simulator::update_number_of_blocks(int work_size)
{

    this->number_of_blocks = (work_size + this->threads_per_block - 1) / this->threads_per_block;
}

void Simulator::evaluate_cuda_error(std::string message, cudaError_t err)
{

    if (err != cudaSuccess)
    {
        std::cout << message << std::endl;
        fprintf(stderr, " (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void Simulator::reset_timers()
{
    this->time_in_move = 0.0;
    this->time_in_force_calculation = 0.0;
    this->time_in_contact_detection = 0.0;
    this->time_in_reset_forces_and_contacts = 0.0;
    this->time_in_find_max_number_of_cells_per_grid_block = 0.0;
    this->time_in_grid_realloc=0.0;
    this->time_in_calculate_number_of_cells_per_grid=0.0;
    this->time_in_reset_grid = 0.0;

}

void Simulator::print_stats()
{
    std::cout << "\n";
    
    std::cout << "n_cells: " << this->cells.size() << "\n";
    std::cout << "max_number_of_cells_per_grid_block: " << this->max_number_of_cells_per_grid_block<< "\n";
    std::cout << "\n";
    std::cout << "times";
    std::cout << "\n";
    double total_time = 0.0;
    std::cout << "time_in_move: " << this->time_in_move << "\n";
     total_time += this->time_in_move;
    std::cout << "time_in_force_calculation: " << this->time_in_force_calculation << "\n";
     total_time += this->time_in_force_calculation;
    std::cout << "time_in_contact_detection: " << this->time_in_contact_detection << "\n";
    total_time += this->time_in_contact_detection;
    std::cout << "time_in_reset_forces_and_contacts: " << this->time_in_reset_forces_and_contacts << "\n";
    total_time += this->time_in_reset_forces_and_contacts;
    std::cout << "time_in_find_max_number_of_cells_per_grid_block: " << this->time_in_find_max_number_of_cells_per_grid_block << "\n";
    total_time += this->time_in_find_max_number_of_cells_per_grid_block;
    std::cout << "time_in_grid_realloc: " << this->time_in_grid_realloc << "\n";
    total_time += this->time_in_grid_realloc;
    std::cout << "time_in_calculate_number_of_cells_per_grid: " << this->time_in_calculate_number_of_cells_per_grid << "\n";
    total_time += this->time_in_calculate_number_of_cells_per_grid;
    std::cout << "time_in_reset_grid: " << this->time_in_reset_grid << "\n";
    total_time += this->time_in_reset_grid;
    std::cout << "total_time: " << total_time << "\n";
    std::cout << "\n";
}

// Grid related stuff

__host__ __device__ int get_grid_index_from_position(
    double x, double y, double x_min, double y_min, int nx, double block_size)
{
    const int i = static_cast<int>(floor((x - x_min) / block_size));
    const int j = static_cast<int>(floor((y - y_min) / block_size));

    return i + nx * j;
}

void reset_grid_CPU(std::vector<grid_block> &grid)
{
#pragma omp parallel for
    for (int grid_id = 0; grid_id < grid.size(); grid_id++)
    {
        grid[grid_id].number_of_cells = 0;
        grid[grid_id].last_position = -1;
    }
    return;
}

__global__ void reset_grid_GPU(grid_block *grid, int number_of_grid_blocks)
{

    int grid_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (grid_id < number_of_grid_blocks)
    {
        grid[grid_id].number_of_cells = 0;
        grid[grid_id].last_position = -1;
    }
    return;
}


void Simulator::reset_grid()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (!this->simulator_settings.run_in_GPU)
    {
        reset_grid_CPU(this->grid);
    }
    else
    {
        reset_grid_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
            this->device_grid,this->grid.size());

        cudaDeviceSynchronize();

        evaluate_cuda_error("Error on reset_grid", cudaGetLastError());
    }

    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_reset_grid += std::chrono::duration<double>(stop - start).count();

    return;


}


__host__ __device__ void count_cell_on_grid(cell *cells, grid_block *grid, int cell_id,
    double x_min_domain,
    double y_min_domain,
    int grid_nx,
    double block_size)
{
    if (!cells[cell_id].active) 
        return;

    const int block_index = get_grid_index_from_position(
            cells[cell_id].x,
            cells[cell_id].y,
            x_min_domain,
            y_min_domain,
            grid_nx,
            block_size);

#ifdef __CUDA_ARCH__
    atomicAdd(&grid[block_index].number_of_cells, 1);
#else
#pragma omp atomic update
    grid[block_index].number_of_cells +=1;
#endif
        
    cells[cell_id].grid_block = block_index;


    return;
}


void calculate_number_of_cells_per_grid_CPU(std::vector<cell> &cells,std::vector<grid_block> &grid, 
    double x_min_domain,
    double y_min_domain,
    int grid_nx,
    double block_size)
{

    for (int home_id = 0; home_id < cells.size(); home_id++)
    {

        count_cell_on_grid(cells.data(), grid.data(), home_id,
            x_min_domain,
            y_min_domain,
            grid_nx,
            block_size);
    }
}

__global__ void calculate_number_of_cells_per_grid_GPU(cell* device_cells, size_t number_of_cells,grid_block *device_grid, 
    double x_min_domain,
    double y_min_domain,
    int grid_nx,
    double block_size)
{

    int cell_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (cell_id < number_of_cells)
    {
        count_cell_on_grid(device_cells, device_grid, cell_id,
            x_min_domain,
            y_min_domain,
            grid_nx,
            block_size);
    }
    return;
}




void Simulator::calculate_number_of_cells_per_grid()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (!this->simulator_settings.run_in_GPU)
    {
            calculate_number_of_cells_per_grid_CPU(this->cells,this->grid, 
                this->simulator_settings.x_min_domain,
                this->simulator_settings.y_min_domain,
                this->grid_nx,
                this->block_size);
    }

    else
    {

        calculate_number_of_cells_per_grid_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
                this->device_cells,cells.size(),this->device_grid, 
                this->simulator_settings.x_min_domain,
                this->simulator_settings.y_min_domain,
                this->grid_nx,
                this->block_size);

        cudaDeviceSynchronize();

        evaluate_cuda_error("Error on update_cells_ids_on_grid_GPU", cudaGetLastError());
    }

    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_calculate_number_of_cells_per_grid += std::chrono::duration<double>(stop - start).count();

    return;
}

__host__ __device__ void update_cell_on_grid(int cell_id, cell *cells, double block_size,
                                             grid_block *grid, int *grid_cells_ids, int nx,
                                             int max_number_of_cells_per_grid_block, double x_min, double y_min)
{
    if (!cells[cell_id].active) 
        return;

    const int block_index = get_grid_index_from_position(
        cells[cell_id].x, cells[cell_id].y, x_min, y_min, nx, block_size);

    int last_index_old;
#ifdef __CUDA_ARCH__
    last_index_old = atomicAdd(&grid[block_index].last_position, 1);
#else
#pragma omp atomic capture
    {
        last_index_old = grid[block_index].last_position; // the two occurences of x[f(i)] must evaluate to the
        grid[block_index].last_position += 1;             // same memory location, otherwise behavior is undefined.
    }
#endif

    const int index = block_index * max_number_of_cells_per_grid_block + last_index_old + 1;
    grid_cells_ids[index] = cell_id;

    return;
}

void update_cells_ids_on_grid_CPU(
    std::vector<cell> &cells, double block_size, std::vector<grid_block> &grid, int *grid_cells_ids, int nx,
    int max_number_of_cells_per_grid_block, double x_min, double y_min)
{
#pragma omp parallel for
    for (int cell_id = 0; cell_id < cells.size(); cell_id++)
    {
        update_cell_on_grid(cell_id, cells.data(), block_size, grid.data(), grid_cells_ids, nx,
                            max_number_of_cells_per_grid_block, x_min, y_min);
    }
    return;
}

__global__ void update_cells_ids_on_grid_GPU(cell *device_cells, size_t number_of_cells,
                                             double block_size, grid_block *device_grid, int *device_grid_cells_ids, int nx,
                                             int max_number_of_cells_per_grid_block, double x_min, double y_min)
{

    int cell_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (cell_id < number_of_cells)
    {
        update_cell_on_grid(cell_id, device_cells, block_size, device_grid, device_grid_cells_ids, nx,
                            max_number_of_cells_per_grid_block, x_min, y_min);
    }
    return;
}

void Simulator::update_cells_ids_on_grid()
{
    auto start = std::chrono::high_resolution_clock::now();

    //

    // if(true)
    if (!this->simulator_settings.run_in_GPU)
    {
        update_cells_ids_on_grid_CPU(this->cells, this->block_size, this->grid, this->grid_cell_ids, this->grid_nx,
                                     this->max_number_of_cells_per_grid_block,
                                     this->simulator_settings.x_min_domain,
                                     this->simulator_settings.y_min_domain);
    }

    else
    {

        update_cells_ids_on_grid_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
            this->device_cells, cells.size(),
            this->block_size, this->device_grid,
            this->device_grid_cell_ids, this->grid_nx,
            this->max_number_of_cells_per_grid_block,
            this->simulator_settings.x_min_domain,
            this->simulator_settings.y_min_domain);

        cudaDeviceSynchronize();

        evaluate_cuda_error("Error on update_cells_ids_on_grid_GPU", cudaGetLastError());
    }

    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_update_cells_ids_on_grid += std::chrono::duration<double>(stop - start).count();

    return;
}

void Simulator::allocate_cells_in_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;

    size_t size_of_cells = sizeof(cell) * this->cells.size();

    err = cudaMalloc((void **)&this->device_cells, size_of_cells);

    evaluate_cuda_error("Error on allocate_cells_in_GPU", err);
}

void Simulator::copy_cells_to_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;

    size_t size_of_cells = sizeof(cell) * cells.size();

    err = cudaMemcpy(this->device_cells, cells.data(), size_of_cells, cudaMemcpyHostToDevice);

    evaluate_cuda_error("Error on copy_cells_to_GPU", err);
}
void Simulator::copy_cells_to_CPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;

    size_t size_of_cells = sizeof(cell) * cells.size();

    err = cudaMemcpy(cells.data(), this->device_cells, size_of_cells, cudaMemcpyDeviceToHost);

    evaluate_cuda_error("Error on copy_cells_to_CPU", err);
}

void Simulator::free_cells_on_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;

    err = cudaFree(this->device_cells);

    evaluate_cuda_error("Error on free device cells", err);
}

void Simulator::allocate_grid_in_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    size_t size_of_grid = sizeof(grid_block) * this->grid.size();
    err = cudaMalloc((void **)&this->device_grid, size_of_grid);
    evaluate_cuda_error("Error on allocate_cells_in_GPU on grid struct allocation", err);
}

void Simulator::copy_grid_to_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    size_t size_of_grid = sizeof(grid_block) * this->grid.size();
    err = cudaMemcpy(this->device_grid, grid.data(), size_of_grid, cudaMemcpyHostToDevice);
    evaluate_cuda_error("Error on copy_cells_to_GPU  on grid struct copy", err);
}
void Simulator::copy_grid_to_CPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    size_t size_of_grid = sizeof(grid_block) * this->grid.size();
    err = cudaMemcpy(grid.data(), this->device_grid, size_of_grid, cudaMemcpyDeviceToHost);
    evaluate_cuda_error("Error on copy_cells_to_CPU on grid struct copy", err);
}

void Simulator::free_grid_on_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    err = cudaFree(this->device_grid);
    evaluate_cuda_error("Error on free device cells", err);
}

void Simulator::allocate_grid_cells_ids_in_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    size_t size_of_grid_ids = sizeof(int) * this->grid.size() * this->max_number_of_cells_per_grid_block;
    err = cudaMalloc((void **)&this->device_grid_cell_ids, size_of_grid_ids);
    evaluate_cuda_error("Error on allocate_cells_in_GPU  on grid_cell_ids allocation", err);
}

void Simulator::copy_grid_cells_ids_to_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    size_t size_of_grid_ids = sizeof(int) * this->grid.size() * this->max_number_of_cells_per_grid_block;
    err = cudaMemcpy(this->device_grid_cell_ids, this->grid_cell_ids, size_of_grid_ids, cudaMemcpyHostToDevice);
    evaluate_cuda_error("Error on copy_cells_to_GPU on grid_cell_ids allocation copy", err);
}
void Simulator::copy_grid_cells_ids_to_CPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    size_t size_of_grid_ids = sizeof(int) * this->grid.size() * this->max_number_of_cells_per_grid_block;
    err = cudaMemcpy(this->grid_cell_ids, this->device_grid_cell_ids, size_of_grid_ids, cudaMemcpyDeviceToHost);
    evaluate_cuda_error("Error on copy_cells_to_CPU on grid_cell_ids copy", err);
}

void Simulator::free_grid_cells_ids_on_GPU()
{

    if (!this->simulator_settings.run_in_GPU)
        return;

    cudaError_t err = cudaSuccess;
    err = cudaFree(this->device_grid_cell_ids);
    evaluate_cuda_error("Error on free device cells", err);
}

__global__ void find_max_number_of_cells_per_grid_block_GPU(
    grid_block *device_grid, int *device_max_number_of_cells_per_grid_block, int *d_mutex, unsigned int grid_size)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ float cache[256];

    int temp = -1;
    while (index + offset < grid_size)
    {
        temp = fmaxf(temp, device_grid[index + offset].number_of_cells);

        offset += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        while (atomicCAS(d_mutex, 0, 1) != 0)
            ; // lock
        *device_max_number_of_cells_per_grid_block = fmaxf(*device_max_number_of_cells_per_grid_block, cache[0]);
        atomicExch(d_mutex, 0); // unlock
    }
    __syncthreads();
}

void find_max_number_of_cells_per_grid_block_CPU(grid_block *grid, int &max_number_of_cells_per_grid_block, int grid_size)
{
    for (int grid_id = 0; grid_id < grid_size; grid_id++)
        if (grid[grid_id].number_of_cells > max_number_of_cells_per_grid_block)
            max_number_of_cells_per_grid_block = grid[grid_id].number_of_cells;
}

void Simulator::find_max_number_of_cells_per_grid_block()
{
    auto start = std::chrono::high_resolution_clock::now();

    if (!this->simulator_settings.run_in_GPU)
    {

        find_max_number_of_cells_per_grid_block_CPU(
            grid.data(), this->max_number_of_cells_per_grid_block, grid.size());
    }
    else
    {
        cudaMemset(this->d_mutex, 0, sizeof(int));
        find_max_number_of_cells_per_grid_block_GPU<<<this->number_of_blocks, this->threads_per_block>>>(
            this->device_grid,
            this->device_max_number_of_cells_per_grid_block, this->d_mutex, grid.size());

        cudaMemcpy(&this->max_number_of_cells_per_grid_block,
                   this->device_max_number_of_cells_per_grid_block, sizeof(int),
                   cudaMemcpyDeviceToHost);

        evaluate_cuda_error("Error on update_positions_and_velocities", cudaGetLastError());
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_find_max_number_of_cells_per_grid_block += std::chrono::duration<double>(stop - start).count();

    return;
}

void Simulator::grid_realloc()
{
        auto start = std::chrono::high_resolution_clock::now();

    // now we have to allocate the grid
    if (this->simulator_settings.run_in_GPU)
    {
        free_grid_cells_ids_on_GPU();
        allocate_grid_cells_ids_in_GPU();
    }
    else
    {
        delete this->grid_cell_ids;
        this->grid_cell_ids = new int[this->max_number_of_cells_per_grid_block * grid.size()];
    }
        auto stop = std::chrono::high_resolution_clock::now();
    this->time_in_grid_realloc += std::chrono::duration<double>(stop - start).count();

    return;
}

void Simulator::do_one_iteration()
{

    
    update_number_of_blocks(grid.size());
    
    reset_grid();
    
    const int old_max_number_of_cells_per_grid = this->max_number_of_cells_per_grid_block;
    
    update_number_of_blocks(cells.size());
    
    calculate_number_of_cells_per_grid();
    
    update_number_of_blocks(grid.size());
    
    find_max_number_of_cells_per_grid_block();

    if (this->max_number_of_cells_per_grid_block != old_max_number_of_cells_per_grid)
    {
        grid_realloc();
    }

    update_number_of_blocks(cells.size());

    update_cells_ids_on_grid();

    reset_contacts_and_forces();

    calculate_contacts();
   

    calculate_forces();

    update_position();

    // deactivate cells!
}

void Simulator::add_wall(
    double x_min, double y_min,
    double x_max, double y_max,
    double spacing, double spring_coefficient,
    double damping_ratio, double density)
{
    double x = x_min;
    double y = y_min;

    double wall_vector_x = x_max - x_min;
    double wall_vector_y = y_max - y_min;

    double wall_length = sqrt(wall_vector_x * wall_vector_x + wall_vector_y * wall_vector_y);

    wall_vector_x /= wall_length;
    wall_vector_y /= wall_length;

    double length = 0.0;

    int wall_id = this->number_of_walls;

    while (true)
    {
        add_cell(x, y, 0.0, 0.0, spring_coefficient, damping_ratio, density, wall_id);

        x += wall_vector_x * spacing;
        y += wall_vector_y * spacing;

        length += spacing;

        if (length >= wall_length)
            break;
    }

    this->number_of_walls++;
}

void Simulator::add_grid_of_cells(
    double x_min, double y_min,
    double x_max, double y_max, double spring_coefficient,
    double damping_ratio, double density)
{
    double x = x_min;
    double y = y_min;

    if (x_min > x_max || y_min > y_max)
    {
        throw std::invalid_argument("x_min and y_min must greater than x_max and y_max");
    }

    while (true)
    {
        x = x_min;

        while (true)
        {

            add_cell(x, y, 0.0, 0.0, spring_coefficient, damping_ratio, density);
            
            x += 2.0 * this->simulator_settings.radius;

            if (x > x_max)
                break;
        }

        y += 2.0 * this->simulator_settings.radius;

        if (y > y_max)
            break;
    }
    return;
}

void Simulator::initialize_grid()
{

    this->block_size = simulator_settings.grid_size_multiplier * simulator_settings.radius;

    this->grid_nx = static_cast<int>(
        ceil((simulator_settings.x_max_domain - simulator_settings.x_min_domain) /
             (block_size)));

    this->grid_ny = static_cast<int>(ceil(
        (simulator_settings.y_max_domain - simulator_settings.y_min_domain) / block_size));

    grid.resize(this->grid_nx * this->grid_ny);

    this->max_number_of_cells_per_grid_block = 0;

    this->grid_cell_ids = new int[this->max_number_of_cells_per_grid_block * grid.size()];
}

void Simulator::start_simulation(
    double time_step, double simulation_duration, bool run_in_GPU, double output_interval)
{
    this->simulator_settings.run_in_GPU = run_in_GPU;
    this->simulator_settings.time_step = time_step;
    this->threads_per_block = 256;
    this->max_number_of_cells_per_grid_block = 0;

    double time_next_output = output_interval;

    initialize_grid();
    reset_timers();

    allocate_cells_in_GPU();
    allocate_grid_in_GPU();
    allocate_grid_cells_ids_in_GPU();

    cudaMalloc((void **)&this->d_mutex, sizeof(int));
    cudaMalloc((void **)&this->device_max_number_of_cells_per_grid_block, sizeof(int));
    cudaMemset(this->device_max_number_of_cells_per_grid_block, -1, sizeof(int));
    
    copy_cells_to_GPU();
    copy_grid_to_GPU();
    
    while (true)
    {

        do_one_iteration();

        this->current_time += time_step;

        if (this->current_time > time_next_output)
        {
            copy_cells_to_CPU();
            time_next_output += output_interval;
        }

        if (current_time >= simulation_duration)
            break;
    }

    print_stats();

    cudaFree(this->d_mutex);
    cudaFree(this->device_max_number_of_cells_per_grid_block);

    return;
}
