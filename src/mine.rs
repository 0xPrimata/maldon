use cust::prelude::*;
use cust::memory::DeviceBuffer;
use alloy_primitives::{Address, FixedBytes};
use std::ffi::CString;
use std::fs;

pub(super) trait Miner {
    /// Runs the Miner.
    fn mine(&self, starts_pattern: &[u8], ends_pattern: &[u8]) -> (Address, FixedBytes<32>);
}

/// A CREATE2 Miner.
#[derive(Debug, Clone, Copy)]
pub(super) struct Create2Miner {
    factory: Address,
    deployer: Address,
    init_code_hash: FixedBytes<32>,
}

impl Create2Miner {
    const MAX_INCREMENTING_NONCE: u64 = u64::MAX >> 2;

    pub(super) fn new(factory: Address, deployer: Address, init_code_hash: FixedBytes<32>) -> Self {
        Self {
            factory,
            deployer,
            init_code_hash,
        }
    }
}

impl Miner for Create2Miner {
    fn mine(&self, starts_pattern: &[u8], ends_pattern: &[u8]) -> (Address, FixedBytes<32>) {
        let mut hash_buffer = [0u8; 85];
        hash_buffer[0] = 0xff;
        hash_buffer[1..21].copy_from_slice(self.factory.as_slice());
        hash_buffer[21..41].copy_from_slice(self.deployer.as_slice());
        hash_buffer[53..85].copy_from_slice(self.init_code_hash.as_slice());

        // CUDA setup
        cust::init(CudaFlags::empty()).expect("Failed to initialize CUDA");
        let device = Device::get_device(0).expect("Failed to get CUDA device");
        let _context = Context::new(device).expect("Failed to create CUDA context");

        // Allocate GPU memory for starts_pattern, ends_pattern, and result
        let starts_pattern_device = DeviceBuffer::from_slice(starts_pattern).expect("Failed to allocate starts pattern on GPU");
        let ends_pattern_device = DeviceBuffer::from_slice(ends_pattern).expect("Failed to allocate ends pattern on GPU");
        let result_device = DeviceBuffer::from_slice(&[0u8; 32]).expect("Failed to allocate result buffer on GPU");

        // Transfer salt buffer and other necessary data to the GPU
        let salt_device = DeviceBuffer::from_slice(&hash_buffer).expect("Failed to transfer salt to GPU");

        // Use a dummy proxy_init_code_hash for Create2Miner (not used, but needed for kernel signature)
        let dummy_proxy_init_code_hash = DeviceBuffer::from_slice(&[0u8; 32]).expect("Failed to allocate dummy buffer on GPU");

        // Load the PTX kernel from the file
        let ptx_source = fs::read_to_string("./mine_patterns.ptx").expect("Failed to read PTX file");
        let ptx_cstr = CString::new(ptx_source).expect("Failed to convert PTX to CString");

        // Compile and load the PTX module
        let module = Module::from_ptx_cstr(&ptx_cstr, &[]).expect("Failed to load PTX module");
        let stream = Stream::new(StreamFlags::DEFAULT, None).expect("Failed to create CUDA stream");

        // Set a smaller nonce_max for testing
        let threads_per_block = 256u32; // Block size
        let number_of_blocks = 1024u32;
        let nonce_max = 1_000_000u32;

        println!("Launching kernel for CREATE2 with {} blocks and {} threads per block", number_of_blocks, threads_per_block);

        // Launch the CUDA kernel
        unsafe {
            launch!(module.mine_patterns<<<number_of_blocks, threads_per_block, 0, stream>>>(
                starts_pattern_device.as_device_ptr(),
                ends_pattern_device.as_device_ptr(),
                salt_device.as_device_ptr(),
                dummy_proxy_init_code_hash.as_device_ptr(),
                result_device.as_device_ptr(),
                starts_pattern.len() as i32,
                ends_pattern.len() as i32,
                hash_buffer.len() as i32,
                nonce_max as i32
            )).expect("Kernel launch failed for CREATE2");
        }

        stream.synchronize().expect("Failed to synchronize CUDA stream for CREATE2");

        // Copy results back from GPU
        let mut result = [0u8; 32];
        result_device.copy_to(&mut result).expect("Failed to copy result from GPU for CREATE2");

        println!("CREATE2 mining result: {:?}", result);

        let address = Address::from_slice(&result[12..32]);
        let salt = FixedBytes::<32>::from_slice(&hash_buffer[21..53]);

        (address, salt)
    }
}

/// A CREATE3 Miner with added verbosity.
#[derive(Debug, Clone, Copy)]
pub(super) struct Create3Miner {
    factory: Address,
    deployer: Address,
}

impl Create3Miner {
    const PROXY_INIT_CODE_HASH: [u8; 32] = [
        0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54,
        0x2f, 0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49,
        0x7c, 0x1f,
    ];

    pub fn new(factory: Address, deployer: Address) -> Self {
        Self { factory, deployer }
    }
}

impl Miner for Create3Miner {
    fn mine(&self, starts_pattern: &[u8], ends_pattern: &[u8]) -> (Address, FixedBytes<32>) {
        let mut salt_buffer = [0u8; 52];
        salt_buffer[0..20].copy_from_slice(self.deployer.as_slice());

        let mut proxy_create_buffer = [0u8; 23];
        proxy_create_buffer[0..2].copy_from_slice(&[0xd6, 0x94]);
        proxy_create_buffer[22] = 0x01;

        // CUDA setup
        cust::init(CudaFlags::empty()).expect("Failed to initialize CUDA");
        let device = Device::get_device(0).expect("Failed to get CUDA device");
        let _context = Context::new(device).expect("Failed to create CUDA context");

        // Allocate GPU memory for starts_pattern, ends_pattern, and result
        let starts_pattern_device = DeviceBuffer::from_slice(starts_pattern).expect("Failed to allocate starts pattern on GPU");
        let ends_pattern_device = DeviceBuffer::from_slice(ends_pattern).expect("Failed to allocate ends pattern on GPU");
        let result_device = DeviceBuffer::from_slice(&[0u8; 32]).expect("Failed to allocate result buffer on GPU");

        // Transfer salt buffer and PROXY_INIT_CODE_HASH to GPU
        let salt_device = DeviceBuffer::from_slice(&salt_buffer).expect("Failed to transfer salt to GPU");
        let proxy_init_code_hash_device = DeviceBuffer::from_slice(&Create3Miner::PROXY_INIT_CODE_HASH).expect("Failed to transfer proxy init code hash to GPU");

        // Load the PTX kernel from the file
        let ptx_source = fs::read_to_string("./mine_patterns.ptx").expect("Failed to read PTX file");
        let ptx_cstr = CString::new(ptx_source).expect("Failed to convert PTX to CString");

        // Compile and load the PTX module
        let module = Module::from_ptx_cstr(&ptx_cstr, &[]).expect("Failed to load PTX module");
        let stream = Stream::new(StreamFlags::DEFAULT, None).expect("Failed to create CUDA stream");

        let threads_per_block = 256u32; // Block size
        let number_of_blocks = 1024u32; // Set to a safe default

        println!(
            "Launching kernel for CREATE3 with {} blocks, {} threads per block",
            number_of_blocks, threads_per_block
        );

        println!(
            "CUDA memory info - starts_pattern length: {}, ends_pattern length: {}, salt_buffer length: {}",
            starts_pattern.len(),
            ends_pattern.len(),
            salt_buffer.len()
        );

        // Use a smaller nonce_max for testing
        let nonce_max = 1_000_000u32;

        // Launch the CUDA kernel
        unsafe {
            let launch_result = launch!(module.mine_patterns<<<number_of_blocks, threads_per_block, 0, stream>>>(
                starts_pattern_device.as_device_ptr(),
                ends_pattern_device.as_device_ptr(),
                salt_device.as_device_ptr(),
                proxy_init_code_hash_device.as_device_ptr(),
                result_device.as_device_ptr(),
                starts_pattern.len() as i32,
                ends_pattern.len() as i32,
                salt_buffer.len() as i32,
                nonce_max as i32 // Reduced nonce_max for testing
            ));

            if let Err(e) = launch_result {
                eprintln!("Kernel launch failed for CREATE3: {:?}", e);
                return (Address::repeat_byte(0), FixedBytes::<32>::default());
            }
        }

        stream.synchronize().expect("Failed to synchronize CUDA stream for CREATE3");

        // Copy results back from GPU
        let mut result = [0u8; 32];
        result_device.copy_to(&mut result).expect("Failed to copy result from GPU for CREATE3");

        println!("CREATE3 mining result: {:?}", result);

        let address = Address::from_slice(&result[12..32]);
        let salt = FixedBytes::<32>::from_slice(&salt_buffer[0..32]);

        (address, salt)
    }
}
