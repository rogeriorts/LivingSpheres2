#define CATCH_CONFIG_MAIN
#define _USE_MATH_DEFINES
#include <catch2/catch.hpp>
#include "simulator.h"
#include <iostream>

void set_up_and_run_one_timestep(bool run_in_gpu, double *y)
{
    bool show_animation = false;
    double radius = 3.0;
    double spring_coefficient = 1e3;
    double damping_ratio = 0.5;
    double density = 1.0 / (4.0 / 3.0 * M_PI * radius * radius * radius); //kg / m3
    Simulator sim = Simulator(radius);
    sim.simulator_settings.run_in_GPU = run_in_gpu;
    
    sim.simulator_settings.gravity = -9.81;
    
    sim.simulator_settings.x_min_domain = 0.0;
    sim.simulator_settings.x_max_domain = 100.0;
    sim.simulator_settings.y_min_domain = 0.0;
    sim.simulator_settings.y_max_domain = sim.simulator_settings.x_max_domain;
    
    
    double x_min_cells = 20.0;
    double y_min_cells = 20.0;

    double x_max_cells = 20.0;
    double y_max_cells = 37.0;
    
    sim.add_grid_of_cells(
        x_min_cells,
        y_min_cells,
        x_max_cells,
        y_max_cells,
        spring_coefficient,
        damping_ratio,
        density
    );
    sim.add_wall(0.0,0.0,sim.simulator_settings.x_max_domain,0.0,radius * 2.0,spring_coefficient,damping_ratio,density);
    sim.add_wall(sim.simulator_settings.x_max_domain,0.0,sim.simulator_settings.x_max_domain,sim.simulator_settings.x_max_domain,radius * 2.0,spring_coefficient,damping_ratio,density);
    sim.add_wall(sim.simulator_settings.x_max_domain,sim.simulator_settings.x_max_domain,0.0,sim.simulator_settings.x_max_domain,radius * 2.0,spring_coefficient,damping_ratio,density);
    sim.add_wall(0.0,sim.simulator_settings.x_max_domain,0.0,0.0,radius * 2.0,spring_coefficient,damping_ratio,density);
    double time_step = 0.1;
    double output_interval = 0.01;
    double simulation_duration = time_step;

    sim.start_simulation(time_step,simulation_duration,run_in_gpu,output_interval);
    
    for(int i=0;i<3;i++)
        y[i]=sim.cells[i].y;

    return;

}

TEST_CASE("test move calculation")
{
    

    double y_in_cpu[3]={0.0,0.0,0.0};
    double y_in_gpu[3]={0.0,0.0,0.0};
    
    set_up_and_run_one_timestep(false, y_in_cpu);
    set_up_and_run_one_timestep(true, y_in_gpu);

    for(int i=0; i<3; i++)
    {
        CHECK(y_in_cpu[i] == Approx(y_in_gpu[i]).margin(1e-6));

        
    }

    return;
}
