#define _USE_MATH_DEFINES
#include <cmath>
#include "user_interface.h"

int main()
{

    std::vector<bool> run_in_GPU_vec = {true};
    int CPU_threads = 8;

    for (bool run_in_GPU : run_in_GPU_vec)
    {

        std::cout << "Run in GPU" << run_in_GPU << std::endl;
        
        bool show_animation = true;
        double radius = 5.0;
        double spring_coefficient = 5e3;
        double damping_ratio = 0.8;
        double density = 1.0 / (4.0 / 3.0 * M_PI * radius * radius * radius); // kg / m3
        Simulator sim = Simulator(radius);
        sim.simulator_settings.run_in_GPU = run_in_GPU;

        sim.simulator_settings.gravity = -9.81;

        sim.simulator_settings.x_min_domain = -5.0;
        sim.simulator_settings.x_max_domain = 1500.0;
        sim.simulator_settings.y_min_domain = -5.0;
        sim.simulator_settings.y_max_domain = 505.0;

        double x_min_cells = 20.0;
        double y_min_cells = 50.0;

        double x_max_cells = 1400.2;
        double y_max_cells = 470.6;

        sim.add_grid_of_cells(
            x_min_cells,
            y_min_cells,
            x_max_cells,
            y_max_cells,
            spring_coefficient,
            damping_ratio,
            density);
        
        sim.add_wall(radius, y_min_cells*0.7, sim.simulator_settings.x_max_domain - radius, radius, radius * 2.0, spring_coefficient, damping_ratio, density);
        sim.add_wall(sim.simulator_settings.x_max_domain - radius, radius, sim.simulator_settings.x_max_domain - radius, sim.simulator_settings.y_max_domain - radius, radius * 2.0, spring_coefficient, damping_ratio, density);
        sim.add_wall(sim.simulator_settings.x_max_domain - radius, sim.simulator_settings.y_max_domain - radius, radius, sim.simulator_settings.y_max_domain - radius, radius * 2.0, spring_coefficient, damping_ratio, density);
        sim.add_wall(radius, sim.simulator_settings.y_max_domain - radius, radius, radius, radius * 2.0, spring_coefficient, damping_ratio, density);
        
        double time_step = 1e-3;
        double output_interval = time_step*60;
        double simulation_duration = 60.0;

        launch_simulation(sim, time_step, output_interval, simulation_duration, run_in_GPU, CPU_threads);
    }
    return 0;
}
