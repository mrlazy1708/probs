#[allow(unused_imports)]
use super::*;

#[allow(unused_imports)]
use super::{domain, domain::*};

#[allow(unused_imports)]
use super::{sampler, sampler::*};

pub mod univar {
    use num::*;

    pub fn uniform<D: ToPrimitive>() -> impl Fn(&D) -> f64 {
        move |_| 1.0
    }

    pub fn normal<D: ToPrimitive>(mu: f64, sigma: f64) -> impl Fn(&D) -> f64 {
        move |x| (-(x.to_f64().unwrap() - mu).powi(2) / (2.0 * sigma.powi(2))).exp()
    }

    pub fn cauchy<D: ToPrimitive>(t: f64, s: f64) -> impl Fn(&D) -> f64 {
        use std::f64::consts::PI;
        move |x| 1.0 / (PI * s * (1.0 + (x.to_f64().unwrap() - t) / s).powi(2))
    }
}
