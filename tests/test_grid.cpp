#define CATCH_CONFIG_MAIN
#define _USE_MATH_DEFINES
#include <catch2/catch.hpp>
#include "simulator.h"
#include <iostream>

TEST_CASE("test simulation CPU")
{
   
    std::vector<bool> run_in_GPU_vec = {false,true};
    for(bool run_in_GPU : run_in_GPU_vec)
    {

    double radius = 0.5;
    double spring_coefficient = 1e3;
    double damping_ratio = 0.5;
    double density = 1.0 / (4.0 / 3.0 * M_PI * radius * radius * radius); //kg / m3
    double grid_size_multiplier = 2.0;
    Simulator sim = Simulator(radius);
    sim.simulator_settings.run_in_GPU = run_in_GPU;
    
    sim.simulator_settings.x_min_domain = 0.0;
    sim.simulator_settings.x_max_domain = 4.0;
    sim.simulator_settings.y_min_domain = 0.0;
    sim.simulator_settings.y_max_domain = 4.0;

    sim.simulator_settings.grid_size_multiplier = grid_size_multiplier;
    
    double x_min_cells = 1.5;
    double y_min_cells = 1.5;

    double x_max_cells = 2.5;
    double y_max_cells = 2.5;
    
    sim.add_grid_of_cells(
        x_min_cells,
        y_min_cells,
        x_max_cells,
        y_max_cells,
        spring_coefficient,
        damping_ratio,
        density
    );

    CHECK(sim.cells.size() == 4);

    sim.simulator_settings.gravity = 0.0;

    double time_step = 1e-4;
    double simulation_duration = time_step;

    std::vector<cell> cells;
    double time = 0.0;

    sim.start_simulation(time_step,simulation_duration,run_in_GPU, 10*time_step);

    if(run_in_GPU)
    {
        sim.copy_grid_to_CPU();
        sim.copy_cells_to_CPU();
        sim.copy_grid_cells_ids_to_CPU();
        
    }
    CHECK(sim.grid.size() == 4*4);

    CHECK(sim.max_number_of_cells_per_grid_block == 1);

    CHECK(sim.cells[0].grid_block == 5);
    CHECK(sim.cells[1].grid_block == 6);
    CHECK(sim.cells[2].grid_block == 9);
    CHECK(sim.cells[3].grid_block == 10);


    CHECK(sim.grid[5].number_of_cells == 1);
    CHECK(sim.grid[6].number_of_cells == 1);
    CHECK(sim.grid[9].number_of_cells == 1);
    CHECK(sim.grid[10].number_of_cells == 1);
    

    CHECK(sim.grid_cell_ids[5*sim.max_number_of_cells_per_grid_block + 0] == 0);
    CHECK(sim.grid_cell_ids[6*sim.max_number_of_cells_per_grid_block + 0] == 1);
    CHECK(sim.grid_cell_ids[9*sim.max_number_of_cells_per_grid_block + 0] == 2);
    CHECK(sim.grid_cell_ids[10*sim.max_number_of_cells_per_grid_block + 0] == 3);
    }

}


