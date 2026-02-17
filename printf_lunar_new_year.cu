#include <iostream>
#include <cuda_runtime.h>

// 16x16 Horse Bitmask (Year of the Horse 2026)
__constant__ unsigned short horse_mask[16] = {
    0x0000, 0x0C00, 0x1E00, 0x1F00, 
    0x1F80, 0x1FE0, 0x1FF0, 0x1FF8, 
    0x3FFC, 0x7FFE, 0x7E3E, 0xE21E, 
    0xC01E, 0x001E, 0x001E, 0x0000
};

__global__ void print_lunar_horse() {
    // Only use 16 threads (one for each row) to keep printf order clean
    int y = threadIdx.x; 
    if (y >= 16) return;

    // Thread 0 prints the header
    if (y == 0) {
        printf("\n  --- LUNAR NEW YEAR 2026: YEAR OF THE HORSE ---\n\n");
    }

    // We still want threads to wait for the header
    __syncthreads();

    char row_string[18]; // 16 chars + newline + null terminator
    unsigned short row_data = horse_mask[y];

    // Each thread builds its own row string in its local registers
    for (int x = 0; x < 16; x++) {
        bool is_horse_part = (row_data >> (15 - x)) & 1;
        row_string[x] = is_horse_part ? 'M' : ' ';
    }
    row_string[16] = '\n';
    row_string[17] = '\0';

    // To prevent the "jumbled line" issue, we can use a simple atomic or
    // a loop to force the rows to print in order (y=0, then y=1...)
    for (int i = 0; i < 16; i++) {
        if (y == i) {
            printf("%s", row_string);
        }
        // This ensures the hardware-level "Drill Sergeant" 
        // lets each row take its turn.
        __syncthreads(); 
    }
}

int main() {
    // Launch 16 threads in 1D
    print_lunar_horse<<<1, 16>>>();
    
    cudaDeviceSynchronize();
    std::cout << "\n  [Sent from the stables of your RTX 3060]" << std::endl;
    return 0;
}