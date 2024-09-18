#include <stdint.h>  // For uint64_t
#include <string.h>  // For memcpy

#define KECCAK_ROUNDS 24

// Keccak Rho offsets
__device__ const int keccak_rho_offsets[24] = {
    1,  3,  6, 10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};

// Keccak Pi permutation indices
__device__ const int keccak_pi[24] = {
    10, 7,  11, 17, 18, 3,  5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2,  20, 14, 22, 9,  6,  1};

// Constants used in Keccak's round function
__device__ const uint64_t keccak_round_constants[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

// Keccak state array is 5x5 of 64-bit values (total 1600 bits)
__device__ void keccak_f(uint64_t* state) {
    uint64_t temp_state[5] = {0};

    // Each round of the Keccak function
    for (int round = 0; round < KECCAK_ROUNDS; ++round) {
        // Theta step
        for (int i = 0; i < 5; ++i) {
            temp_state[i] = state[i] ^ state[5 + i] ^ state[10 + i] ^ state[15 + i] ^ state[20 + i];
        }
        for (int i = 0; i < 5; ++i) {
            uint64_t t = temp_state[(i + 4) % 5] ^ __funnelshift_l(temp_state[(i + 1) % 5], temp_state[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                state[i + j] ^= t;
            }
        }

        // Rho and Pi steps
        uint64_t last = state[1];
        for (int i = 0; i < 24; ++i) {
            int j = keccak_rho_offsets[i];
            temp_state[0] = state[keccak_pi[i]];
            state[keccak_pi[i]] = __funnelshift_l(last, last, j);
            last = temp_state[0];
        }

        // Chi step
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) {
                temp_state[i] = state[i + j];
            }
            for (int i = 0; i < 5; ++i) {
                state[i + j] ^= (~temp_state[(i + 1) % 5]) & temp_state[(i + 2) % 5];
            }
        }

        // Iota step
        state[0] ^= keccak_round_constants[round];
    }
}

// Padding and absorbing the input data into the Keccak state
__device__ void keccak_absorb(uint64_t* state, const unsigned char* input, size_t length) {
    for (size_t i = 0; i < length / 8; ++i) {
        state[i] ^= ((uint64_t*)input)[i];
    }
}

// The full Keccak256 function
__device__ void keccak256(unsigned char* hash, const unsigned char* input, size_t input_len) {
    uint64_t state[25] = {0};
    keccak_absorb(state, input, input_len);

    // Padding
    unsigned char padding[136] = {0};
    padding[0] = 1;  // Standard padding
    padding[135] = 0x80;
    keccak_absorb(state, padding, 136);

    // Permutation
    keccak_f(state);

    // Extracting the hash from the state
    memcpy(hash, state, 32);
}

// CUDA mining kernel using keccak256
extern "C" __global__ void mine_patterns(
    const unsigned char* starts_pattern,
    const unsigned char* ends_pattern,
    const unsigned char* salt_buffer,
    const unsigned char* proxy_init_code_hash,
    unsigned char* result,
    int starts_pattern_len,
    int ends_pattern_len,
    int salt_len,
    int nonce_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nonce_max) {
        unsigned char salt[32];
        memcpy(salt, salt_buffer, salt_len);
        uint64_t nonce = idx;
        memcpy(&salt[salt_len], &nonce, sizeof(uint64_t));

        // Compute keccak256 hash
        unsigned char hash[32];
        keccak256(hash, salt, 32);

        // Debug: output the hash value for the first thread
        if (idx == 0) {
            printf("Hash for first nonce: ");
            for (int i = 0; i < 32; i++) {
                printf("%02x", hash[i]);
            }
            printf("\n");
        }

        // Check if the hash matches the patterns
        bool starts_match = true;
        bool ends_match = true;

        for (int i = 0; i < starts_pattern_len; i++) {
            if (hash[i] != starts_pattern[i]) {
                starts_match = false;
                break;
            }
        }

        for (int i = 0; i < ends_pattern_len; i++) {
            if (hash[32 - ends_pattern_len + i] != ends_pattern[i]) {
                ends_match = false;
                break;
            }
        }

        if (starts_match && ends_match) {
            memcpy(result, hash, 32);
        }
    }
}
