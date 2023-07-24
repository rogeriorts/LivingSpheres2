#define CATCH_CONFIG_MAIN
#define _USE_MATH_DEFINES
#include <catch2/catch.hpp>
#include "simulator.h"
#include <iostream>

TEST_CASE("test force calculation CPU")
{


    // cell initial properties
    double radius = 1.0;
    double initial_distance = 0.1;

    double x_home = 0.0;
    double y_home = 0.0;

    double x_near = initial_distance / sqrt(2);
    double y_near = initial_distance / sqrt(2);

    double vx_home = 0.41;
    double vy_home = 0.3;

    double vx_near = -0.5;
    double vy_near = -0.2;

    double damping_ratio_home = 0.4;
    double damping_ratio_near = 0.3;

    double density_home = 0.4;
    double density_near = 0.3;

    double cell_volume = 4.0 / 3.0 * M_PI * radius * radius * radius;

    double mass_home = density_home * cell_volume;
    double mass_near = density_near * cell_volume;

    double friction_coefficient = 0.3;

    double dt = 0.1;

    // calculate normal force
    double spring_coefficient_home = 1e2;
    double spring_coefficient_near = 2e2;

    double spring_coefficient = 2.0/(1.0 / spring_coefficient_home + 1.0 / spring_coefficient_near);

    double spring_deformation = 2.0 * 1.0 - 0.1;
    double spring_force = spring_deformation * spring_coefficient;

    // calculate damping force
    double mass = (mass_home + mass_near) * 0.5;

    double damping_ratio = 2.0 / (1.0 / damping_ratio_home + 1.0 / damping_ratio_near);

    double damp_coeff = 2.0 * damping_ratio * sqrt(mass * radius * spring_coefficient);

    double dist_x = x_near;
    double dist_y = y_near;
    double squared_distance = dist_x * dist_x + dist_y * dist_y;

    double distance = sqrt(squared_distance);
    double x_normal = dist_x / distance;
    double y_normal = dist_y / distance;

    double x_tangential = y_normal;
    double y_tangential = -x_normal;

    double vx_rel = vx_near - vx_home;
    double vy_rel = vy_near - vy_home;

    double v_normal = vx_rel * x_normal + vy_rel * y_normal;
    double v_tangential = vx_rel * x_tangential + vy_rel * y_tangential;

    double vx_normal = v_normal * x_normal;
    double vy_normal = v_normal * y_normal;

    double damping_force_x = -vx_normal * damp_coeff;
    double damping_force_y = -vy_normal * damp_coeff;

    //tangential force
    double vx_tangential = v_tangential * x_tangential;
    double vy_tangential = v_tangential * y_tangential;

    double abs_tangential = fabs(v_tangential);
    double tangential_force_x = 0.0;
    double tangential_force_y = 0.0;
    if (abs_tangential > 1e-15)
    {
        tangential_force_x = -friction_coefficient * spring_force * vx_tangential / abs_tangential;
        tangential_force_y = -friction_coefficient * spring_force * vy_tangential / abs_tangential;
    }

    double force_x = -tangential_force_x - damping_force_x - x_normal*spring_force;
    double force_y = -tangential_force_y - damping_force_y - y_normal*spring_force;

    double force = sqrt(force_x*force_x + force_y*force_y);

    double ax_home = force_x / mass_home;
    double ay_home = force_y / mass_home;

    double ax_near = -force_x / mass_near;
    double ay_near = -force_y / mass_near;

    double new_vx_home = vx_home + ax_home * dt;
    double new_vy_home = vy_home + ay_home * dt;

    double new_x_home = x_home + vx_home * dt + ax_home * dt * dt * 0.5;
    double new_y_home = y_home + vy_home * dt + ay_home * dt * dt * 0.5;

    double new_vx_near = vx_near + ax_near * dt;
    double new_vy_near = vy_near + ay_near * dt;

    double new_x_near = x_near + vx_near * dt + ax_near * dt * dt * 0.5;
    double new_y_near = y_near + vy_near * dt + ay_near * dt * dt * 0.5;

    std::vector<bool> run_in_GPU_vec = {true,false};
    for(bool run_in_GPU : run_in_GPU_vec)
    {

        Simulator sim = Simulator(radius,friction_coefficient,dt);
        sim.simulator_settings.run_in_GPU = run_in_GPU;

        sim.add_cell(
            x_home, y_home,
            vx_home, vy_home,
            spring_coefficient_home, damping_ratio_home,
            density_home);

        sim.add_cell(
            x_near,y_near,
            vx_near, vy_near,
            spring_coefficient_near, damping_ratio_near,
            density_near);



        sim.start_simulation(dt,dt,false,dt);
        
        if(run_in_GPU)
        {
            sim.copy_cells_to_CPU();
        }

        CHECK(force_x == Approx(sim.cells[0].force_x).margin(1e-8));
        CHECK(force_y == Approx(sim.cells[0].force_y).margin(1e-8));


        CHECK(-force_x == Approx(sim.cells[1].force_x).margin(1e-8));
        CHECK(-force_y == Approx(sim.cells[1].force_y).margin(1e-8));
        
        CHECK(new_vx_home == Approx(sim.cells[0].vx).margin(1e-8));
        CHECK(new_vy_home == Approx(sim.cells[0].vy).margin(1e-8));

        CHECK(new_vx_near == Approx(sim.cells[1].vx).margin(1e-8));
        CHECK(new_vy_near == Approx(sim.cells[1].vy).margin(1e-8));


        CHECK(new_x_home == Approx(sim.cells[0].x).margin(1e-8));
        CHECK(new_y_home == Approx(sim.cells[0].y).margin(1e-8));

        CHECK(new_x_near == Approx(sim.cells[1].x).margin(1e-8));
        CHECK(new_y_near == Approx(sim.cells[1].y).margin(1e-8));
    }
    return;
}

TEST_CASE("test walls CPU")
{

    double radius = 0.1;

    double friction_coefficient = 0.3;
    
    
    double dt = 0.1;

    double x_wall_min = 0.3;
    double y_wall_min = 0.4;
    
    double x_wall_max = -0.8;
    double y_wall_max = 0.9;

    double spacing = 2.0 * radius;

    double density = 0.7;

    double spring_coeff = 2.3;
    
    double damping_ratio = 0.4;

    double dx = -0.18207329;
    double dy = 0.082760588;

    bool run_in_GPU = false;
    
    Simulator sim = Simulator(radius,friction_coefficient,dt,-0.8,-0.8,0.8,0.8);
    sim.simulator_settings.run_in_GPU = run_in_GPU;

    sim.add_wall(x_wall_min,y_wall_min,x_wall_max,y_wall_max,spacing,spring_coeff,damping_ratio,density);

    for(size_t i = 0; i < 7; i++)
    {

        CHECK(x_wall_min + (double)i*dx == Approx(sim.cells[i].x).margin(1e-6));
        CHECK(y_wall_min + (double)i*dy == Approx(sim.cells[i].y).margin(1e-6));
        CHECK(0 == sim.cells[i].wall_id);
    }

    sim.add_cell(x_wall_min - radius, y_wall_min,0.0,0.0,spring_coeff,damping_ratio,density);

    CHECK(sim.number_of_walls == 1);

    sim.start_simulation(dt,dt,false,dt);


    CHECK(sim.cells.size() == 8);
    
    for(size_t i = 0; i < 7; i++)
    {

        CHECK(x_wall_min + (double)i*dx == Approx(sim.cells[i].x).margin(1e-6));
        CHECK(y_wall_min + (double)i*dy == Approx(sim.cells[i].y).margin(1e-6));
        CHECK(0 == sim.cells[i].wall_id);
    }
    
    CHECK(sim.cells[7].x < 0.2);
    
    
    return;
}
