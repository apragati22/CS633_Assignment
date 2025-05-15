#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

#define MEMSET0(arr, size) do{ \
    memset(arr, 0, (size)*(sizeof(float))); \
}while(0);

float min(float x, float y){
    if(x < y) return x;
    else return y;
}

float max(float x, float y){
    if(x > y) return x;
    else return y;
}

typedef struct{
    float* data;
    int nx, ny, nz;
} Data3D;
Data3D* createData3D(int nx, int ny, int nz);
float getData3D(Data3D* data3D, int x, int y, int z);
void setData3D(Data3D* data3D, int x, int y, int z, float val);
void printData3D(Data3D* data3D);
void freeData3D(Data3D* data3D);


int main(int argc, char *argv[]) {
    double time1, time2, time3;
    double read_time, main_code_time, total_time;
    MPI_Init(&argc, &argv);
    int rank, size, err, err2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 10) {
        printf("Usage: %s <input_file> PX PY PZ NX NY NZ NC <output_file>\n", argv[0]);
        return 1;
    }
    int PX = atoi(argv[2]), PY = atoi(argv[3]), PZ = atoi(argv[4]);
    int NX = atoi(argv[5]), NY = atoi(argv[6]), NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);  
    int P = PX * PY * PZ;

    char input_file[1024];
    strcpy(input_file, argv[1]);
    char output_file[1024];
    strcpy(output_file, argv[9]);

    if (P != size) {
        if (rank == 0)
            printf("Error: Expected %d processes, but got %d\n", P, size);
        exit(0);
    }

    int local_nx = NX / PX;
    int local_ny = NY / PY;
    int local_nz = NZ / PZ;
    int local_nt = NC;
    long int local_data_size = local_nz*local_ny*local_nx*local_nt;
    long int total_data_size = NX*NY*NZ*NC;
    float* data;
    float* local_data;
    local_data = (float*)malloc(local_data_size * sizeof(float));
    data = (float*)malloc(total_data_size*sizeof(float));
    MEMSET0(data, total_data_size);


    int x = rank % PX;
    int y = (rank / PX) % PY;   
    int z = rank / (PX * PY);

    int sizes[4] = {NZ, NY, NX, NC};      // Total matrix size
    int subsizes[4] = {local_nz, local_ny, local_nx, local_nt};               // Size of the subarray block
    int starts[4] = {0, 0, 0, 0};                 // Starting at element (1,1,1,1)  
    MPI_Datatype subarray_type;
    MPI_Type_create_subarray(4, sizes, subsizes, starts,
         MPI_ORDER_C, MPI_FLOAT, &subarray_type);
    MPI_Type_commit(&subarray_type);

    int displ[P];
    int sendcounts[P];

    for(int i = 0; i < P; i++) {
        int x = i % PX;
        int y = (i / PX) % PY;   
        int z = i / (PX * PY);
        displ[i] = (z*PX*PY*NX*NY*NZ/(PX*PY*PZ) + y*PX*NX*NY/(PX*PY) + x*(NX/PX))*NC;
        sendcounts[i] = 1;
    }
    MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &subarray_type);
    MPI_Type_commit(&subarray_type);
    MPI_Request req[P];
    MPI_Status status[P];
    MPI_Request req0;
    MPI_Status stat0;
        FILE *file = fopen(input_file, "r");
        if (file == NULL) {
            printf("Error opening file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    time1 = MPI_Wtime();
    if(rank == 0) {
        for(int z = 0 ; z<NZ ; ++z){
            for(int y = 0 ; y<NY ; ++y){
                for(int x = 0 ; x<NX ; ++x){
                    for (int t = 0; t<NC; ++t) {
                        float value;
                        fscanf(file, "%f", &value);
                        data[z*NX*NY*NC + y*NX*NC + x*NC + t] = value;
                    } 
                }
            }
        }
        // for(int z=0; z<NZ; z++){
        //     for(int y=0; y<NY; y++){
        //         for(int x=0; x<NX; x++){
        //             for(int t=0; t<NC; t++){
        //                 printf("%lf ", data[z*NY*NX*NC + y*NX*NC + x*NC + t]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // Send the subarray to all processes
        for(int i = 0; i < P; i++) {
            MPI_Isend(&(data[displ[i]]), 1, subarray_type, i, 0, MPI_COMM_WORLD, &req[i]);
        }
    } 
    // Receive the subarray from process 0
    MPI_Irecv(local_data, local_data_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req0);
    
    if(rank == 0) MPI_Waitall(P, &(req[0]), &(status[0]));
    MPI_Wait(&req0, &stat0);

    MPI_Type_free(&subarray_type);
    time2 = MPI_Wtime();

    #if 0 
    // Print local data for each rank
    for(int r = 0; r < P; r++){
        if(r == rank){
            printf("Rank %d:\n", r);
            for(int z=0; z<NZ; z++){
                for(int y=0; y<NY; y++){
                    for(int x=0; x<NX; x++){
                        for(int t=0; t<NC; t++){
                            printf("%lf ", local_data[z*NY*NX*NC + y*NX*NC + x*NC + t]);
                        }
                        printf("\n");
                    }
                }
            }
            printf("\n");
        }
    }
    #endif

    Data3D* x_left = createData3D(local_nt, local_ny, local_nz);
    Data3D* x_right = createData3D(local_nt, local_ny, local_nz);
    Data3D* y_front = createData3D(local_nt, local_nx, local_nz);
    Data3D* y_back = createData3D(local_nt, local_nx, local_nz);
    Data3D* z_bottom = createData3D(local_nt, local_nx, local_ny);
    Data3D* z_top = createData3D(local_nt, local_nx, local_ny);

    int request_count = 0;
    MPI_Request req2[12];
    MPI_Status stat2[12];

    MPI_Datatype vector_type1;
    MPI_Type_vector(local_ny*local_nz, local_nt, local_nx*local_nt, MPI_FLOAT, &vector_type1);
    MPI_Type_commit(&vector_type1);

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
        // err = MPI_Isend(local_data->data, local_nx*local_ny, MPI_FLOAT, rank+PX*PY, rank, MPI_COMM_WORLD, &req2[request_count++]);
        err = MPI_Isend((local_data + (local_nx*local_ny*(local_nz-1)*local_nt)), local_nx*local_ny*local_nt, MPI_FLOAT, rank+PX*PY, rank, MPI_COMM_WORLD, &req2[request_count++]);
        err2 = MPI_Irecv(z_top->data, local_nx*local_ny*local_nt, MPI_FLOAT, rank+PX*PY, rank+PX*PY, MPI_COMM_WORLD, &req2[request_count++]);
        if (err != MPI_SUCCESS) {
            printf("Rank %d: MPI_Isend error (top face), code = %d\n", rank, err);
        }
        if (err2 != MPI_SUCCESS) {
            printf("Rank %d: MPI_Irecv error (top face), code = %d\n", rank, err);
        }
    }
    MPI_Waitall(request_count, req2, stat2);
    MPI_Type_free(&vector_type1);
    MPI_Type_free(&vector_type2);

    long long local_min_cnt[NC];
    long long local_max_cnt[NC];
    float local_min_val[NC];
    for(int i = 0; i < NC; i++) {
        local_min_val[i] = FLT_MAX;
        local_min_cnt[i] = 0;
    }
    float local_max_val[NC];
    for(int i = 0; i < NC; i++) {
        local_max_val[i] = -FLT_MAX;
        local_max_cnt[i] = 0;
    }
    
    float min_temp, max_temp;
    for (int z = 0; z < local_nz; z++) {
        for (int y = 0; y < local_ny; y++) {
            for (int x = 0; x < local_nx; x++) {
                for(int t = 0; t < local_nt; t++){
                    min_temp = FLT_MAX;
                    max_temp = -FLT_MAX;
                    if(x > 0) {
                        min_temp = min(min_temp, (local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x-1)*local_nt + t]));
                        max_temp = max(max_temp, (local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x-1)*local_nt + t]));
                    }
                    else if(rank %PX > 0) {
                        min_temp = min(min_temp, getData3D(x_left, t, y, z));
                        max_temp = max(max_temp, getData3D(x_left, t, y, z));
                    }
                    
                    if (x < local_nx - 1){
                        min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x+1)*local_nt + t]);
                        max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + y*local_nx*local_nt + (x+1)*local_nt + t]);
                    }
                    else if (rank % PX < PX - 1) {
                        min_temp = min(min_temp, getData3D(x_right,t,y,z));
                        max_temp = max(max_temp, getData3D(x_right,t,y,z));
                    }

                    if (y > 0) {
                        min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + (y-1)*local_nx*local_nt + x*local_nt + t]);
                        max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + (y-1)*local_nx*local_nt + x*local_nt + t]);
                    }
                    else if ((rank / PX) % PY > 0) {
                        min_temp = min(min_temp,getData3D(y_front,t,x,z));
                        max_temp = max(max_temp,getData3D(y_front,t,x,z));
                    }

                    if (y < local_ny - 1) {
                        min_temp = min(min_temp, local_data[z*local_ny*local_nx*local_nt + (y+1)*local_nx*local_nt + x*local_nt + t]);
                        max_temp = max(max_temp, local_data[z*local_ny*local_nx*local_nt + (y+1)*local_nx*local_nt + x*local_nt + t]);
                    }
                    else if ((rank / PX) % PY < PY - 1) {
                        min_temp = min(min_temp, getData3D(y_back,t,x,z));
                        max_temp = max(max_temp, getData3D(y_back,t,x,z));
                    }

                    if (z > 0) {
                        min_temp = min(min_temp, local_data[(z-1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                        max_temp = max(max_temp, local_data[(z-1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                    }
                    else if (rank / (PX * PY) > 0) {
                        min_temp = min(min_temp, getData3D(z_bottom,t,x,y));
                        max_temp = max(max_temp, getData3D(z_bottom,t,x,y));
                    }

                    if (z < local_nz - 1) {
                        min_temp = min(min_temp, local_data[(z+1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                        max_temp = max(max_temp, local_data[(z+1)*local_ny*local_nx*local_nt + y*local_nx*local_nt + x*local_nt + t]);
                    }
                    else if (rank / (PX * PY) < PZ - 1) {
                        min_temp = min(min_temp,getData3D(z_top,t,x,y));
                        max_temp = max(max_temp,getData3D(z_top,t,x,y));
                    }

                    // Compute local minima and maxima
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
    long long global_min_count[NC];
    long long global_max_count[NC];
    float global_minimum[NC];
    float global_maximum[NC];
    for(int i = 0; i < NC; i++) {
        global_minimum[i] = FLT_MAX;
        global_min_count[i]=0;
    }
    for(int i = 0; i < NC; i++) {
        global_maximum[i] = -FLT_MAX;
        global_max_count[i]=0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_max_cnt, &global_max_count, NC, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_cnt, &global_min_count, NC, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_val, &global_minimum, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_val, &global_maximum, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    time3 = MPI_Wtime();
    read_time = time2 - time1;
    main_code_time = time3 - time2;
    total_time = time3 - time1;
    double max_read_time, max_main_code_time, max_total_time;
    MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&main_code_time, &max_main_code_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0){
        FILE *out_file;
        out_file = fopen(output_file, "w");
        if (out_file == NULL) {
            printf("Error opening %s!\n", output_file);
            exit(0);
        }
        for(int i = 0; i < NC-1; i++){
            fprintf(out_file, "(%lld, %lld), ", global_min_count[i], global_max_count[i]);
        }
        fprintf(out_file, "(%lld, %lld)\n", global_min_count[NC-1], global_max_count[NC-1]);
        for(int i = 0; i < NC-1; i++){
            fprintf(out_file, "(%f, %f), ", global_minimum[i], global_maximum[i]);
        }
        fprintf(out_file, "(%f, %f)\n", global_minimum[NC-1], global_maximum[NC-1]);
        fprintf(out_file, "%lf, %lf, %lf\n", max_read_time, max_main_code_time, max_total_time);
        fclose(out_file);
    }
        fclose(file);
    MPI_Finalize();
    return 0;
}

Data3D* createData3D(int nx, int ny, int nz){
    Data3D* data3D = (Data3D*) malloc(sizeof(Data3D));
    data3D->data = (float*) malloc(nz*ny*nx*sizeof(float));
    MEMSET0(data3D->data, nx*ny*nz);
    data3D->nx = nx;
    data3D->ny = ny;
    data3D->nz = nz;
    return data3D;
}
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
void freeData3D(Data3D* data3D){
    free(data3D->data);
    free(data3D);
}
