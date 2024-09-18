mod cli;
mod mine;

use clap::Parser;
use alloy_primitives::{address, Address};
use cust::prelude::*;
use cust::memory::DeviceBuffer;
use std::ffi::CString;

use {
    cli::Maldon,
    mine::{Create2Miner, Create3Miner, Miner},
};

const CREATE2_DEFAULT_FACTORY: Address = address!("0000000000ffe8b47b3e2130213b802212439497");
const CREATE3_DEFAULT_FACTORY: Address = address!("2dfcc7415d89af828cbef005f0d072d8b3f23183");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    cust::init(CudaFlags::all())?;

    // Select CUDA device and create context
    let device = Device::get_device(0)?;
    let context = Context::new(device)?;

    let (address, salt) = match Maldon::parse() {
        Maldon::Create2 {
            deployer,
            factory,
            init_code_hash,
            starts_pattern,
            ends_pattern,
        } => {
            let factory = factory.unwrap_or(CREATE2_DEFAULT_FACTORY);

            let starts_pattern_chars = &starts_pattern.clone().map_or(0, |p| p.chars().count());
            let ends_pattern_chars = &ends_pattern.clone().map_or(0, |p| p.chars().count());

            let character_sum = starts_pattern_chars + ends_pattern_chars;

            if character_sum > 20 {
                println!("Character sum is greater than 20. Exiting.");
                return Ok(());
            }

            // Convert the optional patterns to byte slices
            let starts_pattern_bytes = starts_pattern
                .map_or(Vec::new(), |p| p.into_bytes().unwrap_or_else(|_| Vec::new()));
            let ends_pattern_bytes = ends_pattern
                .map_or(Vec::new(), |p| p.into_bytes().unwrap_or_else(|_| Vec::new()));

            // Allocate GPU memory and copy the patterns to the device
            let mut d_starts_pattern = DeviceBuffer::from_slice(&starts_pattern_bytes)?;
            let mut d_ends_pattern = DeviceBuffer::from_slice(&ends_pattern_bytes)?;
            let mut d_result = DeviceBuffer::from_slice(&vec![0u8; 32])?;

            // Define your CUDA kernel as PTX code
            let ptx_code = r#"
                    extern "C" __global__ void mine_patterns(
                const unsigned char* starts_pattern,
                const unsigned char* ends_pattern,
                unsigned char* result, int size) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
                    result[i] = starts_pattern[i] + ends_pattern[i];  // Example operation
                }
            }
            "#;

            // Compile the PTX code into a CUDA module and load the kernel
            let ptx_cstr = CString::new(ptx_code)?;
            let module = Module::from_ptx_cstr(&ptx_cstr, &[])?;
            let stream = Stream::new(StreamFlags::DEFAULT, None)?;

            // Launch the CUDA kernel
            unsafe {
                launch!(module.mine_patterns<<<1, 32, 0, stream>>>(
                    d_starts_pattern.as_device_ptr(),
                    d_ends_pattern.as_device_ptr(),
                    d_result.as_device_ptr()
                ))?;
            }

            stream.synchronize()?;

            // Retrieve the results from the GPU
            let mut result = vec![0u8; 32];
            d_result.copy_to(&mut result)?;

            Create2Miner::new(factory, deployer, init_code_hash)
                .mine(&result, &[])
        }
        Maldon::Create3 {
            deployer,
            factory,
            starts_pattern,
            ends_pattern,
        } => {
            let factory = factory.unwrap_or(CREATE3_DEFAULT_FACTORY);

            let starts_pattern_chars = &starts_pattern.clone().map_or(0, |p| p.chars().count());
            let ends_pattern_chars = &ends_pattern.clone().map_or(0, |p| p.chars().count());

            let character_sum = starts_pattern_chars + ends_pattern_chars;

            if character_sum > 20 {
                println!("Character sum is greater than 20. Exiting.");
                return Ok(());
            }

            // Convert the optional patterns to byte slices
            let starts_pattern_bytes = starts_pattern
                .map_or(Vec::new(), |p| p.into_bytes().unwrap_or_else(|_| Vec::new()));
            let ends_pattern_bytes = ends_pattern
                .map_or(Vec::new(), |p| p.into_bytes().unwrap_or_else(|_| Vec::new()));

            // Similar CUDA processing as Create2 here...
            Create3Miner::new(factory, deployer)
                .mine(&starts_pattern_bytes, &ends_pattern_bytes)
        }
    };

    println!("Found salt {salt:?} ==> {address:?}");

    Ok(())
}
