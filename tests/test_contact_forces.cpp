#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../include/LiVingSpheres/simulator.h"


TEST_CASE("test contact detection in gpu")
{
    Simulator sim = Simulator(1.0);

    CHECK(sim.simulator_settings.radius == 1.0);
       
    sim.add_cell(0.0,0.0,0.0,0.0);

    CHECK(sim.cells[0].x == 0.0);
    CHECK(sim.cells[0].y == 0.0);
    CHECK(sim.cells[0].vx == 0.0);
    CHECK(sim.cells[0].vy == 0.0);
    
    sim.add_cell(0.1,0.0,0.0,0.0);

    sim.copy_cells_to_gpu();
    
    sim.calculate_contacts();

    sim.copy_cells_to_cpu();

    CHECK(sim.cells[0].contacts[0]==1);

    
    
}

