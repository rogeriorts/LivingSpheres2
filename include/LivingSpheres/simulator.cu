#include "simulator.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void contact_detection(cell* cells, int N, double radius)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N)
    {
        for (size_t i = 0; i < N; i++)
        {
            if (tid < i+0)
            {
                double diff_x = cells[i].x - cells[tid].x;
                double diff_y = cells[i].y - cells[tid].y;

                if (diff_x * diff_x + diff_y * diff_y < radius * radius)
                {
                    cells[tid].number_of_contacts++;

                    cells[tid].contacts[cells[tid].number_of_contacts - 1] = i;
                }
            }
        }
    }
}

void Simulator::add_cell(double x, double y, double vx, double vy)
{
    
    cell cell_;
    cell_.x = x;
    cell_.y = y;
    cell_.vx = vx;
    cell_.vy = vy;
    cells.push_back(cell_);

    
}




void Simulator::copy_cells_to_gpu()
{
    //allocate memory on gpu

    size_t bytes = sizeof(cell) * cells.size();

    cudaMalloc((void**)&this->d_pointer, bytes);
    
    cudaMemcpy(d_pointer, cells.data(),bytes,cudaMemcpyHostToDevice);

    
    for(int k=0;k<cells.size();k++)
    {
        int *device_data;
        
        cudaMalloc((void**)&(device_data), 6 * sizeof(int));
    
        cudaMemcpy(device_data, cells[k].contacts,6 * sizeof(int),cudaMemcpyHostToDevice);
    }


}


void Simulator::copy_cells_to_cpu()
{
    //allocate memory on gpu

    size_t bytes = sizeof(cell) * cells.size();
    
    cudaMemcpy(cells.data(), d_pointer,bytes,cudaMemcpyHostToDevice);
    
    for(int k=0;k<cells.size();k++)
    {
    
        cudaMemcpy(cells[k].contacts, d_pointer[k].contacts,6 * sizeof(int),cudaMemcpyHostToDevice);
    }


}


void Simulator::calculate_contacts()
{
    int size = cells.size();

    contact_detection<<<1, 32>>>(cells.data(), size, simulator_settings.radius);
    cudaDeviceSynchronize();
}
