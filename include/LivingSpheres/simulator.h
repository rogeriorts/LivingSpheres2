#include <vector>


typedef struct {
    double x;
    double y;
    double vx;
    double vy;
    int contacts[6];
    int number_of_contacts;
} cell;

struct SimulationSettings
{
    double radius;
};

struct Simulator{
    
    std::vector<cell> cells;

    SimulationSettings simulator_settings;

    cell * d_pointer;

    Simulator(double radius)
    {
        simulator_settings.radius = radius;
    }

    void add_cell(double x, double y, double vx, double vy);

    void copy_cells_to_gpu();
    void copy_cells_to_cpu();
    void calculate_contacts();
};