#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>


struct cell
{
    double x;
    double y;
    double vx;
    double vy;
    double force_x;
    double force_y;
    double inverse_spring_coefficient;
    double inverse_damping_ratio;
    double mass;
    int contacts[6];
    int grid_block;
    int number_of_contacts;
    int wall_id;
    bool active;
};

struct grid_block
{
    
    int number_of_cells;
    int last_position;
};


struct SimulationSettings
{
    double radius;
    bool run_in_GPU;
    double friction_coefficient;
    double time_step;
    double x_min_domain;
    double y_min_domain;
    double x_max_domain;
    double y_max_domain;
    double gravity;
    double grid_size_multiplier;
};

struct Simulator
{

    std::vector<cell> cells;
    std::vector<grid_block> grid;
    int *grid_cell_ids;

    SimulationSettings simulator_settings;

    cell *device_cells;
    grid_block *device_grid;
    int *device_grid_cell_ids;

    int number_of_walls;

    double grid_block_multiplier;

    double block_size;

    int grid_nx;
    int grid_ny;

    int max_number_of_cells_per_grid_block;
    int *device_max_number_of_cells_per_grid_block;

    double current_time;

    int threads_per_block;
    int number_of_blocks;
    
    double time_in_move;
    double time_in_force_calculation;
    double time_in_contact_detection;
    double time_in_reset_forces_and_contacts;
    double time_in_update_cells_ids_on_grid;
    double time_in_find_max_number_of_cells_per_grid_block;
    double time_in_grid_realloc;
    double time_in_calculate_number_of_cells_per_grid;
    double time_in_reset_grid;

    int *d_mutex;

    Simulator(double radius, 
        double friction_coefficient=0.3,
        double time_step=0.1,
        double x_min_domain = -0.5,
        double y_min_domain = -0.5,
        double x_max_domain = 0.5,
        double y_max_domain = 0.5,
        double gravity = 0.0,
        double grid_size_multiplier = 2.0
        )
    {
        simulator_settings.radius = radius;
        simulator_settings.friction_coefficient = friction_coefficient;
        simulator_settings.time_step = time_step;
        simulator_settings.x_min_domain = x_min_domain;
        simulator_settings.y_min_domain = y_min_domain;
        simulator_settings.x_max_domain = x_max_domain;
        simulator_settings.y_max_domain = y_max_domain;
        simulator_settings.gravity = gravity;
        simulator_settings.grid_size_multiplier = grid_size_multiplier;
        number_of_walls = 0;
        initialize_grid();
        current_time = 0.0;

    }

    void add_cell(double x,
                  double y,
                  double vx,
                  double vy,
                  double spring_coefficient = 1e-4,
                  double damping_ratio = 0.5,
                  double density = 1.0,
                  int wall_id = -1
                  );

    

    
    void calculate_contacts();
    
    void reset_contacts_and_forces();
    
    void calculate_forces();
    
    void update_position();
    
    void do_one_iteration();
    
    void add_wall(    
        double x_min, double y_min, 
        double x_max, double y_max, 
        double spacing,double spring_coefficient, 
        double damping_ratio, double density
    );

    void add_grid_of_cells(    
        double x_min, double y_min, 
        double x_max, double y_max,double spring_coefficient, 
        double damping_ratio, double density
    );

    void start_simulation(
        double time_step, double simulation_duration, bool run_in_GPU, double output_interval);

    void initialize_grid();
    void calculate_number_of_cells_per_grid();


    void reset_grid();

    void grid_realloc();
    void update_cells_ids_on_grid();
    void find_max_number_of_cells_per_grid_block();

    void allocate_cells_in_GPU();
    void copy_cells_to_GPU();
    void copy_cells_to_CPU();
    void free_cells_on_GPU();

    void allocate_grid_in_GPU();
    void copy_grid_to_GPU();
    void copy_grid_to_CPU();
    void free_grid_on_GPU();

    void allocate_grid_cells_ids_in_GPU();
    void copy_grid_cells_ids_to_GPU();
    void copy_grid_cells_ids_to_CPU();
    void free_grid_cells_ids_on_GPU();


    void update_number_of_blocks(int work_size);
    void evaluate_cuda_error(std::string message,cudaError_t err);
    void reset_timers();
    void print_stats();

    
};
