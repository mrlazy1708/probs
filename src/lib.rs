#![feature(generic_associated_types)]
#![feature(type_alias_impl_trait)]

extern crate num;
extern crate rand;

extern crate nalgebra as na;
extern crate ndarray as nd;
extern crate nshare as ns;

pub mod dist;
pub mod randvar;
pub mod sampler;

pub use randvar::*;
pub use sampler::*;
