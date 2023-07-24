#include "simulator.h"
#include <omp.h>
#include <iostream>
#include "CImg.h"

void launch_simulation(
    Simulator sim,double time_step,double output_interval, double simulation_duration, 
    bool run_in_GPU,int CPU_threads)
{
    
    std::vector<cell> cells;

    double next_output = output_interval;
    omp_set_nested(1);
    
    #pragma omp parallel
    {
        
        #pragma omp single 
        {
            #pragma omp task
            {
                      
                omp_set_num_threads(CPU_threads);
                sim.start_simulation(time_step, simulation_duration,run_in_GPU, output_interval);   
                
            }
            
            #pragma omp task
            {

                const unsigned char bluegreen[] = {0, 170, 255};
                const unsigned char black[] = {0, 0, 0};

                double width = sim.simulator_settings.x_max_domain - sim.simulator_settings.x_min_domain;
                double height = sim.simulator_settings.y_max_domain - sim.simulator_settings.y_min_domain;

                cimg_library::CImg<unsigned char> bg((int)width, (int)height, 1, 3, 255);

                bg.draw_rectangle(0, 0, (int)width, (int)height, bluegreen);

                cimg_library::CImgDisplay dsp((int)width, (int)height, "EvolutionSimulator", 0);

                dsp.display(bg);

                cimg_library::CImg<unsigned char> img(bg);

                while(1)
                {
                    double new_time;

                    #pragma omp atomic read                    
                    new_time = sim.current_time;
                    
                    if(new_time > next_output)
                    {
                        img = bg;
                        
                        for (int i = 0; i < sim.cells.size(); i++)
                        {
                            double x,y;
                            bool active;
                            #pragma omp atomic read 
                            active = sim.cells[i].active;

                            if(active)
                            {

                            #pragma omp atomic read 
                            x = sim.cells[i].x;

                            #pragma omp atomic read 
                            y = sim.cells[i].y;

                            img.draw_circle(
                                (int)width - (int)x,
                                (int)height - (int)y,
                                (int)sim.simulator_settings.radius, black);
                                
                            }
                        }
                        dsp.display(img);
                        next_output+=output_interval;
                        
                    }
                    
                    if (new_time >= simulation_duration)
                        break;
                                        
                }
                
            }
        } 
    }
}