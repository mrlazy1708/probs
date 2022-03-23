#![feature(generic_associated_types)]
#![feature(type_alias_impl_trait)]

extern crate num;
extern crate rand;

extern crate nalgebra as na;
extern crate ndarray as nd;
extern crate nshare as ns;

pub mod domain;
pub mod sampler;
pub mod distribution;

pub use domain::*;
pub use sampler::*;
