use super::*;

pub mod univar {
    use super::*;

    pub fn uniform<D: num::ToPrimitive>() -> impl Fn(&D) -> f64 {
        move |_| 1.0
    }

    pub fn gaussian<D: num::ToPrimitive>(mu: f64, sigma: f64) -> impl Fn(&D) -> f64 {
        move |x| (-(x.to_f64().unwrap() - mu).powi(2) / (2.0 * sigma.powi(2))).exp()
    }
}

pub mod multivar {
    use super::*;

    pub fn uniform<D: num::ToPrimitive>() -> impl Fn(&nd::Array1<D>) -> f64 {
        move |_| 1.0
    }

    pub fn gaussian<D: num::ToPrimitive, const R: usize>(
        mu: na::SVector<f64, R>,
        sigma: na::SMatrix<f64, R, R>,
    ) -> impl Fn(&nd::Array1<D>) -> f64 {
        let sigma = sigma.try_inverse().unwrap();
        move |xs| {
            let xs = ns::ToNalgebra::into_nalgebra(xs.map(|x| x.to_f64().unwrap())) - mu;
            (-(sigma * xs).dot(&xs) / 2.0).exp()
        }
    }
}
