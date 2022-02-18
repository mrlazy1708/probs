#![feature(generic_associated_types)]
#![feature(type_alias_impl_trait)]

extern crate nalgebra as na;
extern crate ndarray as nd;
extern crate nshare as ns;

pub mod distribution;
pub mod domain;
pub mod sampler;

use sampler::Global;
#[test]
fn test() {
    sampler::univar::Icdf::new()
	.gibbs(nd::Dim([2, 2]), 100) // 2x2 domain, skip first 100 samples
	.sample(|m: &nd::Array2<u8>| m.sum() as f64);
}