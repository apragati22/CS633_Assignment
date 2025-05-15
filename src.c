#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

// This amount of data can be read in RAM by any process
#define MAX_RAM_READ 150ll*1024*1024*1024 // 150 GB

// To set all elements of an array to zero
#define MEMSET0(arr, size) do{ \
    memset(arr, 0, (size)*(sizeof(float))); \
}while(0);

// Utility function to return minimum of two floats  
float min(float x, float y){
    if(x < y) return x;
    else return y;
}

// Utility function to return maximum of two floats
float max(float x, float y){
    if(x > y) return x;
    else return y;
}

// struct to store 3D data
typedef struct{
    float* data;
    int nx, ny, nz;
} Data3D;

// Utility functions to create, set, get and free 3D data
Data3D* createData3D(int nx, int ny, int nz);
float getData3D(Data3D* data3D, int x, int y, int z);
void setData3D(Data3D* data3D, int x, int y, int z, float val);
void printData3D(Data3D* data3D);
void freeData3D(Data3D* data3D);

int main(int argc, char* argv[]) {

    // Check if the number of arguments is correct
    if (argc != 10) {
        printf("Usage: %s <input_file> PX PY PZ NX NY NZ NC <output_file>\n", argv[0]);
        return 1;
    }

    // Read the command line arguments
    long PX = atol(argv[2]), PY = atol(argv[3]), PZ = atol(argv[4]);
    long NX = atol(argv[5]), NY = atol(argv[6]), NZ = atol(argv[7]);
    long NC = atol(argv[8]); 
    
    // Take out input and output files
    char input_file[1024];
    strcpy(input_file, argv[1]);
    char output_file[1024];
    strcpy(output_file, argv[9]);

    // local variables
    long local_nx = NX / PX;
    long local_ny = NY / PY;
    long local_nz = NZ / PZ;
    long local_nt = NC;
    long local_data_size = local_nz*local_ny*local_nx*local_nt;
    long total_data_size = NX*NY*NZ*NC;

    // linear array to store subdomain data of each process
    float* local_data;
    local_data = (float*)malloc(local_nx * local_ny * local_nz * local_nt * sizeof(float));

    // Doubles to store times
    double time1, time2, time3;
    double read_time = 0, main_code_time = 0, total_time = 0;

    // Local structures to store values per unit time
    long long local_min_cnt[NC];
    long long local_max_cnt[NC];
    float local_min_val[NC];
    for(int i = 0; i < NC; i++) {
        local_min_val[i] = DBL_MAX;
        local_min_cnt[i] = 0;
    }
    float local_max_val[NC];
    for(int i = 0; i < NC; i++) {
        local_max_val[i] = -DBL_MAX;
        local_max_cnt[i] = 0;
    }

    // Structures to store global values per unit time 
    long long global_min_count[NC];
    long long global_max_count[NC];
    float global_minimum[NC];
    float global_maximum[NC];
    for(int i = 0; i < NC; i++) {
        global_minimum[i] = DBL_MAX;
        global_min_count[i]=0;
    }
    for(int i = 0; i < NC; i++) {
        global_maximum[i] = -DBL_MAX;
        global_max_count[i]=0;
    }
    
    // Variables to store the maximum read time, main code time and total time across all processes
    double max_read_time, max_main_code_time, max_total_time;

    MPI_Init(&argc, &argv);
    time1 = MPI_Wtime();
    
    // Get the rank and total number of processes 
    int rank, size, err, err2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate X,Y and Z coordinates of the process
    long X = rank % PX;
    long Y = (rank / PX) % PY;   
    long Z = rank / (PX * PY);

    // Check if number of processes is correct
    long P = PX * PY * PZ;
    if (P != size) {
        if (rank == 0)
            printf("Error: Expected %ld processes, but got %d\n", P, size);
        exit(0);
    }

    // Open the input file
    MPI_File in_file;
    err = MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file);
    if (err != MPI_SUCCESS) {
        printf("Error opening %s!\n", input_file);
        exit(0);
    }

    // Calculate displacement of each process starting from where data in file has to be read
    MPI_Offset displ = (MPI_Offset)(((Z*PX*PY*NX*NY*NZ/(PX*PY*PZ) + Y*PX*NX*NY/(PX*PY) + X*(NX/PX))*NC)*sizeof(float));

    // if the data size per process is larger than the maximum RAM read size, we need to read the data in chunks
    if(local_data_size*sizeof(float) > MAX_RAM_READ){
        
        // Read one time step data at a time
        local_nt = 1;

        // Structures to create Type_create_subarray
        int sizes[4] = {NZ, NY, NX, NC};                                          // Total matrix size
        int subsizes[4] = {local_nz, local_ny, local_nx, local_nt};               // Size of the subarray block
        int starts[4] = {0, 0, 0, 0};                                             // Starting offset (0,0,0,0)  
       
        // Type_create_subarray to set the correct offsets in each dimension so that each process reads its own data
        MPI_Datatype subarray_type;
        MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray_type);
        MPI_Type_commit(&subarray_type);

        // 3D data structures to store the data of the faces received from neighbouring processes
        Data3D* x_left = createData3D(local_nt, local_ny, local_nz); // data from left face
        Data3D* x_right = createData3D(local_nt, local_ny, local_nz); // data from right face
        Data3D* y_front = createData3D(local_nt, local_nx, local_nz); // data from front face
        Data3D* y_back = createData3D(local_nt, local_nx, local_nz); // data from back face
        Data3D* z_bottom = createData3D(local_nt, local_nx, local_ny); // data from bottom face
        Data3D* z_top = createData3D(local_nt, local_nx, local_ny); // data from top face

        // MPI Type vector to send data of left and right faces to neighbouring processes 
        MPI_Datatype vector_type1;
        MPI_Type_vector(local_ny*local_nz, local_nt, local_nx*local_nt, MPI_FLOAT, &vector_type1);
        MPI_Type_commit(&vector_type1);

        // MPI Type vector to send data of front and back faces to neighbouring processes
        MPI_Datatype vector_type2;
        MPI_Type_vector(local_nz, local_nx*local_nt, local_nx*local_ny*local_nt, MPI_FLOAT, &vector_type2);
        MPI_Type_commit(&vector_type2);

        float min_temp, max_temp;

        for (int t=0; t<NC; t++){ 
    
            int request_count = 0;
            MPI_Request req2[12];
            MPI_Status stat2[12];
            MPI_Status stat0;    

            // read file per time step
            time1 = MPI_Wtime();
            // Set the view of the file so that each process reads its own data
            MPI_File_set_view(in_file, (MPI_Offset) displ, MPI_FLOAT, subarray_type, "native", MPI_INFO_NULL);
            MPI_File_read_at_all(in_file, 0, local_data, local_data_size, MPI_FLOAT, &stat0);
            time2 = MPI_Wtime();
    
            // Send-Recv from left face (x = 0)
            if(rank % PX > 0){
                err = MPI_Isend(local_data, 1, vector_type1, rank-1, rank, MPI_COMM_WORLD, &req2[request_count++]);
                err2 = MPI_Irecv(x_left->data, local_ny*local_nz*local_nt, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, &req2[request_count++]);
                if (err != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Isend error (left face), code = %d\n", rank, err);
                }
                if (err2 != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Irecv error (left face), code = %d\n", rank, err);
                }
            }

            // Send-Recv from right face (x= local_nx - 1)
            if (rank % PX < PX - 1) {
                err = MPI_Isend((local_data + (local_nx-1)*(local_nt)), 1, vector_type1, rank+1, rank, MPI_COMM_WORLD, &req2[request_count++]);
                err2 = MPI_Irecv(x_right->data, local_ny*local_nz*local_nt, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, &req2[request_count++]);
                if (err != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Isend error (right face), code = %d\n", rank, err);
                }
                if (err2 != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Irecv error (right face), code = %d\n", rank, err);
                }
            }

            // Send-Recv from front face (y = 0)
            if ((rank / PX) % PY > 0) {
                err = MPI_Isend(local_data, 1, vector_type2, rank-PX, rank, MPI_COMM_WORLD, &req2[request_count++]);
                err2 = MPI_Irecv(y_front->data, local_nx*local_nz*local_nt, MPI_FLOAT, rank-PX, rank-PX, MPI_COMM_WORLD, &req2[request_count++]);
                if (err != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Isend error (front face), code = %d\n", rank, err);
                }
                if (err2 != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Irecv error (front face), code = %d\n", rank, err);
                }
            }

            // Send-Recv from back face (y = local_ny -1)
            if ((rank / PX) % PY < PY - 1) {
                err = MPI_Isend((local_data + (local_ny-1)*(local_nx)*(local_nt)), 1, vector_type2, rank+PX, rank, MPI_COMM_WORLD, &req2[request_count++]);
                err2 = MPI_Irecv(y_back->data, local_nx*local_nz*local_nt, MPI_FLOAT, rank+PX, rank+PX, MPI_COMM_WORLD, &req2[request_count++]);
                if (err != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Isend error (back face), code = %d\n", rank, err);
                }
                if (err2 != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Irecv error (back face), code = %d\n", rank, err);
                }
            }

            // Send-Recv from bottom face (z = 0)
            if (rank / (PX * PY) > 0) {
                err = MPI_Isend(local_data, local_nx*local_ny*local_nt, MPI_FLOAT, rank-PX*PY, rank, MPI_COMM_WORLD, &req2[request_count++]);
                err2 = MPI_Irecv(z_bottom->data, local_nx*local_ny*local_nt, MPI_FLOAT, rank-PX*PY, rank-PX*PY, MPI_COMM_WORLD, &req2[request_count++]);
                if (err != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Isend error (bottom face), code = %d\n", rank, err);
                }
                if (err2 != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Irecv error (bottom face), code = %d\n", rank, err);
                }
            }

            // Send-Recv from top face (z = local_nz-1)
            if (rank / (PX * PY) < PZ - 1) {
                err = MPI_Isend((local_data + (local_nx*local_ny*(local_nz-1)*local_nt)), local_nx*local_ny*local_nt, MPI_FLOAT, rank+PX*PY, rank, MPI_COMM_WORLD, &req2[request_count++]);
                err2 = MPI_Irecv(z_top->data, local_nx*local_ny*local_nt, MPI_FLOAT, rank+PX*PY, rank+PX*PY, MPI_COMM_WORLD, &req2[request_count++]);
                if (err != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Isend error (top face), code = %d\n", rank, err);
                }
                if (err2 != MPI_SUCCESS) {
                    printf("Rank %d: MPI_Irecv error (top face), code = %d\n", rank, err);
                }
            }

            // Wait for entire data to come before proceeding with computation
            MPI_Waitall(request_count, req2, stat2);
            
            for (int z = 0; z < local_nz; z++) {
                for (int y = 0; y < local_ny; y++) {
                    for (int x = 0; x < local_nx; x++) {
                        min_temp = DBL_MAX;
                        max_temp = -DBL_MAX;
                        if(x > 0) { // data point to the left in the same process
                            min_temp = min(min_temp, (local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x-1)*local_nt]));
                            max_temp = max(max_temp, (local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x-1)*local_nt]));
                        }
                        else if(rank %PX > 0) { // data point to the left in the neighbouring process
                            min_temp = min(min_temp, getData3D(x_left, 0, y, z));
                            max_temp = max(max_temp, getData3D(x_left, 0, y, z));
                        }
                        
                        if (x < local_nx - 1){ // data point to the right in the same process
                            min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x+1)*local_nt]);
                            max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x+1)*local_nt]);
                        }
                        else if (rank % PX < PX - 1) { // data point to the right in the neighbouring process
                            min_temp = min(min_temp, getData3D(x_right,0,y,z));
                            max_temp = max(max_temp, getData3D(x_right,0,y,z));
                        }

                        if (y > 0) { // data point to the front in the same process
                            min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + (y-1)*local_nx*local_nt + x*local_nt]);
                            max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + (y-1)*local_nx*local_nt + x*local_nt]);
                        }
                        else if ((rank / PX) % PY > 0) { // data point to the front in the neighbouring process
                            min_temp = min(min_temp,getData3D(y_front,0,x,z));
                            max_temp = max(max_temp,getData3D(y_front,0,x,z));
                        }

                        if (y < local_ny - 1) { // data point to the back in the same process
                            min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + (y+1)*local_nx*local_nt + x*local_nt]);
                            max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + (y+1)*local_nx*local_nt + x*local_nt]);
                        }
                        else if ((rank / PX) % PY < PY - 1) { // data point to the back in the neighbouring process
                            min_temp = min(min_temp, getData3D(y_back,0,x,z));
                            max_temp = max(max_temp, getData3D(y_back,0,x,z));
                        }

                        if (z > 0) { // data point to the bottom in the same process
                            min_temp = min(min_temp, local_data[(z-1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt]);
                            max_temp = max(max_temp, local_data[(z-1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt]);
                        }
                        else if (rank / (PX * PY) > 0) { // data point to the bottom in the neighbouring process
                            min_temp = min(min_temp, getData3D(z_bottom,0,x,y));
                            max_temp = max(max_temp, getData3D(z_bottom,0,x,y));
                        }

                        if (z < local_nz - 1) { // data point to the top in the same process
                            min_temp = min(min_temp, local_data[(z+1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt]);
                            max_temp = max(max_temp, local_data[(z+1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt]);
                        }
                        else if (rank / (PX * PY) < PZ - 1) { // data point to the top in the neighbouring process
                            min_temp = min(min_temp,getData3D(z_top,0,x,y));
                            max_temp = max(max_temp,getData3D(z_top,0,x,y));
                        }

                        // Compute local minima and maxima counts and values and store in the correct time step
                        if(local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt] < min_temp){
                            local_min_cnt[t]++;
                            local_min_val[t] = min(local_min_val[t], local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt]);
                        }
                        if(local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt] > max_temp){
                            local_max_cnt[t]++;
                            local_max_val[t] = max(local_max_val[t], local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt]);
                        }
                    }
                }
            }
            // increment displacement to read the next time step
            displ += sizeof(float);

            time3 = MPI_Wtime();

            // add the time elapsed for this time step to the total time of the process
            read_time += time2 - time1;
            main_code_time += time3 - time2;
            total_time += time3 - time1;
        }

        // reduce the final values of the local minima and maxima counts and values across all processes
        MPI_Reduce(&local_max_cnt, &global_max_count, NC, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min_cnt, &global_min_count, NC, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min_val, &global_minimum, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max_val, &global_maximum, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

        // reduce the maximum times across all processes
        MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&main_code_time, &max_main_code_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            
        // free the allocated data types
        MPI_Type_free(&vector_type1);
        MPI_Type_free(&vector_type2);
        MPI_Type_free(&subarray_type);
        freeData3D(x_left);
        freeData3D(x_right);
        freeData3D(y_front);
        freeData3D(y_back);
        freeData3D(z_bottom);
        freeData3D(z_top);
        free(local_data);
    }
       
    else{ // Read the entire data of that process in RAM at once
        
        //  Structures to create Type_create_subarray
        int sizes[4] = {NZ, NY, NX, NC};                                          // Total matrix size
        int subsizes[4] = {local_nz, local_ny, local_nx, local_nt};               // Size of the subarray block
        int starts[4] = {0, 0, 0, 0};                                             // Starting offset (0,0,0,0)        
        MPI_Status stat0; 
        int request_count = 0;
        MPI_Request req2[12];
        MPI_Status stat2[12];                                         
        
        // Type_create_subarray to set the correct offsets in each dimension
        MPI_Datatype subarray_type;
        MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray_type);
        MPI_Type_commit(&subarray_type);

        // Set the file view and read the data
        time1 = MPI_Wtime();
        MPI_File_set_view(in_file, (MPI_Offset) displ, MPI_FLOAT, subarray_type, "native", MPI_INFO_NULL);
        MPI_File_read_at_all(in_file, 0, local_data, local_data_size, MPI_FLOAT, &stat0);        
        time2 = MPI_Wtime();

        // 3D data structures to store the data of the faces received from neighbouring processes
        Data3D* x_left = createData3D(local_nt, local_ny, local_nz); // data from left face
        Data3D* x_right = createData3D(local_nt, local_ny, local_nz); // data from right face
        Data3D* y_front = createData3D(local_nt, local_nx, local_nz); // data from front face
        Data3D* y_back = createData3D(local_nt, local_nx, local_nz); // data from back face
        Data3D* z_bottom = createData3D(local_nt, local_nx, local_ny); // data from bottom face
        Data3D* z_top = createData3D(local_nt, local_nx, local_ny); // data from top face

        // MPI Type vector to send data of left and right faces to neighbouring processes
        MPI_Datatype vector_type1;
        MPI_Type_vector(local_ny*local_nz, local_nt, local_nx*local_nt, MPI_FLOAT, &vector_type1);
        MPI_Type_commit(&vector_type1);

        // MPI Type vector to send data of front and back faces to neighbouring processes
        MPI_Datatype vector_type2;
        MPI_Type_vector(local_nz, local_nx*local_nt, local_nx*local_ny*local_nt, MPI_FLOAT, &vector_type2);
        MPI_Type_commit(&vector_type2);
        
        // Send-Recv from left face (x = 0)
        if(rank % PX > 0){
            err = MPI_Isend(local_data, 1, vector_type1, rank-1, rank, MPI_COMM_WORLD, &req2[request_count++]);
            err2 = MPI_Irecv(x_left->data, local_ny*local_nz*local_nt, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, &req2[request_count++]);
            if (err != MPI_SUCCESS) {
                printf("Rank %d: MPI_Isend error (left face), code = %d\n", rank, err);
            }
            if (err2 != MPI_SUCCESS) {
                printf("Rank %d: MPI_Irecv error (left face), code = %d\n", rank, err);
            }
        }

        // Send-Recv from right face (x= local_nx - 1)
        if (rank % PX < PX - 1) {
            err = MPI_Isend((local_data + (local_nx-1)*(local_nt)), 1, vector_type1, rank+1, rank, MPI_COMM_WORLD, &req2[request_count++]);
            err2 = MPI_Irecv(x_right->data, local_ny*local_nz*local_nt, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, &req2[request_count++]);
            if (err != MPI_SUCCESS) {
                printf("Rank %d: MPI_Isend error (right face), code = %d\n", rank, err);
            }
            if (err2 != MPI_SUCCESS) {
                printf("Rank %d: MPI_Irecv error (right face), code = %d\n", rank, err);
            }
        }

        // Send-Recv from front face (y = 0)
        if ((rank / PX) % PY > 0) {
            err = MPI_Isend(local_data, 1, vector_type2, rank-PX, rank, MPI_COMM_WORLD, &req2[request_count++]);
            err2 = MPI_Irecv(y_front->data, local_nx*local_nz*local_nt, MPI_FLOAT, rank-PX, rank-PX, MPI_COMM_WORLD, &req2[request_count++]);
            if (err != MPI_SUCCESS) {
                printf("Rank %d: MPI_Isend error (front face), code = %d\n", rank, err);
            }
            if (err2 != MPI_SUCCESS) {
                printf("Rank %d: MPI_Irecv error (front face), code = %d\n", rank, err);
            }
        }

        // Send-Recv from back face (y = local_ny -1)
        if ((rank / PX) % PY < PY - 1) {
            err = MPI_Isend((local_data + (local_ny-1)*(local_nx)*(local_nt)), 1, vector_type2, rank+PX, rank, MPI_COMM_WORLD, &req2[request_count++]);
            err2 = MPI_Irecv(y_back->data, local_nx*local_nz*local_nt, MPI_FLOAT, rank+PX, rank+PX, MPI_COMM_WORLD, &req2[request_count++]);
            if (err != MPI_SUCCESS) {
                printf("Rank %d: MPI_Isend error (back face), code = %d\n", rank, err);
            }
            if (err2 != MPI_SUCCESS) {
                printf("Rank %d: MPI_Irecv error (back face), code = %d\n", rank, err);
            }
        }

        // Send-Recv from bottom face (z = 0)
        if (rank / (PX * PY) > 0) {
            err = MPI_Isend(local_data, local_nx*local_ny*local_nt, MPI_FLOAT, rank-PX*PY, rank, MPI_COMM_WORLD, &req2[request_count++]);
            err2 = MPI_Irecv(z_bottom->data, local_nx*local_ny*local_nt, MPI_FLOAT, rank-PX*PY, rank-PX*PY, MPI_COMM_WORLD, &req2[request_count++]);
            if (err != MPI_SUCCESS) {
                printf("Rank %d: MPI_Isend error (bottom face), code = %d\n", rank, err);
            }
            if (err2 != MPI_SUCCESS) {
                printf("Rank %d: MPI_Irecv error (bottom face), code = %d\n", rank, err);
            }
        }

        // Send-Recv from top face (z = local_nz-1)
        if (rank / (PX * PY) < PZ - 1) {
            err = MPI_Isend((local_data + (local_nx*local_ny*(local_nz-1)*local_nt)), local_nx*local_ny*local_nt, MPI_FLOAT, rank+PX*PY, rank, MPI_COMM_WORLD, &req2[request_count++]);
            err2 = MPI_Irecv(z_top->data, local_nx*local_ny*local_nt, MPI_FLOAT, rank+PX*PY, rank+PX*PY, MPI_COMM_WORLD, &req2[request_count++]);
            if (err != MPI_SUCCESS) {
                printf("Rank %d: MPI_Isend error (top face), code = %d\n", rank, err);
            }
            if (err2 != MPI_SUCCESS) {
                printf("Rank %d: MPI_Irecv error (top face), code = %d\n", rank, err);
            }
        }

        // Wait for entire data to come before proceeding with computation
        MPI_Waitall(request_count, req2, stat2);
        
        float min_temp, max_temp;
        for (int z = 0; z < local_nz; z++) {
            for (int y = 0; y < local_ny; y++) {
                for (int x = 0; x < local_nx; x++) {
                    for(int t = 0; t < local_nt; t++){
                        min_temp = DBL_MAX;
                        max_temp = -DBL_MAX;
                        if(x > 0) { // data point to the left in the same process
                            min_temp = min(min_temp, (local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x-1)*local_nt + t]));
                            max_temp = max(max_temp, (local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x-1)*local_nt + t]));
                        }
                        else if(rank %PX > 0) { // data point to the left in the neighbouring process
                            min_temp = min(min_temp, getData3D(x_left, t, y, z));
                            max_temp = max(max_temp, getData3D(x_left, t, y, z));
                        }
                        
                        if (x < local_nx - 1){ // data point to the right in the same process
                            min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x+1)*local_nt + t]);
                            max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x+1)*local_nt + t]);
                        }
                        else if (rank % PX < PX - 1) { // data point to the right in the neighbouring process
                            min_temp = min(min_temp, getData3D(x_right,t,y,z));
                            max_temp = max(max_temp, getData3D(x_right,t,y,z));
                        }

                        if (y > 0) { // data point to the front in the same process
                            min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + (y-1)*local_nx*local_nt + x*local_nt + t]);
                            max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + (y-1)*local_nx*local_nt + x*local_nt + t]);
                        }
                        else if ((rank / PX) % PY > 0) { // data point to the front in the neighbouring process
                            min_temp = min(min_temp,getData3D(y_front,t,x,z));
                            max_temp = max(max_temp,getData3D(y_front,t,x,z));
                        }

                        if (y < local_ny - 1) { // data point to the back in the same process
                            min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + (y+1)*local_nx*local_nt + x*local_nt + t]);
                            max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + (y+1)*local_nx*local_nt + x*local_nt + t]);
                        }
                        else if ((rank / PX) % PY < PY - 1) { // data point to the back in the neighbouring process
                            min_temp = min(min_temp, getData3D(y_back,t,x,z));
                            max_temp = max(max_temp, getData3D(y_back,t,x,z));
                        }

                        if (z > 0) { // data point to the bottom in the same process
                            min_temp = min(min_temp, local_data[(z-1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                            max_temp = max(max_temp, local_data[(z-1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                        }
                        else if (rank / (PX * PY) > 0) { // data point to the bottom in the neighbouring process
                            min_temp = min(min_temp, getData3D(z_bottom,t,x,y));
                            max_temp = max(max_temp, getData3D(z_bottom,t,x,y));
                        }

                        if (z < local_nz - 1) { // data point to the top in the same process
                            min_temp = min(min_temp, local_data[(z+1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                            max_temp = max(max_temp, local_data[(z+1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                        }
                        else if (rank / (PX * PY) < PZ - 1) { // data point to the top in the neighbouring process
                            min_temp = min(min_temp,getData3D(z_top,t,x,y));
                            max_temp = max(max_temp,getData3D(z_top,t,x,y));
                        }

                        // Compute local minima and maxima counts and values and store in the correct time step
                        if(local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt+t] < min_temp){
                            local_min_cnt[t]++;
                            local_min_val[t] = min(local_min_val[t], local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt+t]);
                        }
                        if(local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt+t] > max_temp){
                            local_max_cnt[t]++;
                            local_max_val[t] = max(local_max_val[t], local_data[z*local_ny*local_nx*local_nt+y*local_nx*local_nt+x*local_nt+t]);
                        }
                    }
                }
            }
        }

        // reduce the final values of the local minima and maxima counts and values across all processes
        MPI_Reduce(&local_max_cnt, &global_max_count, NC, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min_cnt, &global_min_count, NC, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min_val, &global_minimum, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max_val, &global_maximum, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        time3 = MPI_Wtime();

        // Calculate the time taken for reading and main code
        read_time = time2 - time1;
        main_code_time = time3 - time2;
        total_time = time3 - time1;
    
        // reduce the maximum times across all processes
        MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&main_code_time, &max_main_code_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
       
        // free the allocated data types
        MPI_Type_free(&subarray_type); 
        MPI_Type_free(&vector_type1);
        MPI_Type_free(&vector_type2);

        freeData3D(x_left);
        freeData3D(x_right);
        freeData3D(y_front);
        freeData3D(y_back);
        freeData3D(z_bottom);
        freeData3D(z_top);
        free(local_data);
    }

    // write the results to the output file
    if(rank == 0){
        FILE *out_file;
        out_file = fopen(output_file, "w");
        if (out_file == NULL) {
            printf("Error opening %s!\n", output_file);
            exit(0);
        }

        // write the counts of the minima and maxima
        for(int i = 0; i < NC-1; i++){
            fprintf(out_file, "(%lld, %lld), ", global_min_count[i], global_max_count[i]);
        }
        fprintf(out_file, "(%lld, %lld)\n", global_min_count[NC-1], global_max_count[NC-1]);
        
        // write the values of the minima and maxima
        for(int i = 0; i < NC-1; i++){
            fprintf(out_file, "(%f, %f), ", global_minimum[i], global_maximum[i]);
        }
        fprintf(out_file, "(%f, %f)\n", global_minimum[NC-1], global_maximum[NC-1]);
        
        // write the times across all processes
        fprintf(out_file, "%lf, %lf, %lf\n", max_read_time, max_main_code_time, max_total_time);
        fclose(out_file);
    }
    MPI_File_close(&in_file);
    MPI_Finalize();
    return 0;
}

// To create a 3D data structure
Data3D* createData3D(int nx, int ny, int nz){
    Data3D* data3D = (Data3D*) malloc(sizeof(Data3D));
    data3D->data = (float*) malloc(nz*ny*nx*sizeof(float));
    MEMSET0(data3D->data, nx*ny*nz);
    data3D->nx = nx;
    data3D->ny = ny;
    data3D->nz = nz;
    return data3D;
}

// To get and set data in the 3D data structure
float getData3D(Data3D* data3D, int x, int y, int z){
    int nx = data3D->nx;
    int ny = data3D->ny;
    int nz = data3D->nz;
    return data3D->data[z*(ny*nx) + y*(nx) + x];
}
void setData3D(Data3D* data3D, int x, int y, int z, float val){
    int nx = data3D->nx;
    int ny = data3D->ny;
    int nz = data3D->nz;
    data3D->data[z*(ny*nx) + y*(nx) + x] = val;
}

// To print the data in the 3D data structure
void printData3D(Data3D* data3D){
    int nx = data3D->nx;
    int ny = data3D->ny;
    int nz = data3D->nz;
    for (int z = 0; z < nz; z++){
        for (int y = 0; y < ny; y++){
            for (int x = 0; x < nx; x++){
                float val = getData3D(data3D, x, y, z);
                printf("%f ", val);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

// To free the data in the 3D data structure
void freeData3D(Data3D* data3D){
    free(data3D->data);
    free(data3D);
}