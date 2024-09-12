mod cli;
mod mine;

use clap::Parser;

use alloy_primitives::{address, Address};

use {
    cli::Maldon,
    mine::{Create2Miner, Create3Miner, Miner},
};

const CREATE2_DEFAULT_FACTORY: Address = address!("0000000000ffe8b47b3e2130213b802212439497");
const CREATE3_DEFAULT_FACTORY: Address = address!("2dfcc7415d89af828cbef005f0d072d8b3f23183");

fn main() -> Result<(), hex::FromHexError> {
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

            // Convert the optional patterns to byte slices or an empty byte slice
            let starts_pattern_bytes = starts_pattern.map_or(Ok(Vec::new()), |p| p.into_bytes())?;
            let ends_pattern_bytes = ends_pattern.map_or(Ok(Vec::new()), |p| p.into_bytes())?;

            Create2Miner::new(factory, deployer, init_code_hash)
                .mine(&starts_pattern_bytes, &ends_pattern_bytes)
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

            // Convert the optional patterns to byte slices or an empty byte slice
            let starts_pattern_bytes = starts_pattern.map_or(Ok(Vec::new()), |p| p.into_bytes())?;
            let ends_pattern_bytes = ends_pattern.map_or(Ok(Vec::new()), |p| p.into_bytes())?;

            Create3Miner::new(factory, deployer)
                .mine(&starts_pattern_bytes, &ends_pattern_bytes)
        }
    };

    println!("Found salt {salt:?} ==> {address:?}");

    Ok(())
}
