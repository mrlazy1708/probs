use super::*;

pub trait Sampler<D> {
    type Iter<F: FnMut(&D) -> f64>: Iterator<Item = D>;
    fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Iter<F>;

    fn burn(self, skip: usize) -> adapter::BurnIn<D, Self>
    where
        Self: Sized,
    {
        adapter::BurnIn::new(self, skip)
    }
}

#[doc = "Sample from one-dimensional domain."]
pub mod univar {
    use super::*;

    pub use icdf::Sampler as Icdf;

    #[doc = "Inverse Transform Sampling."]
    pub mod icdf {
        use super::*;

        pub struct Sampler<D: Finite>(std::marker::PhantomData<D>);
        impl<D: Finite> Sampler<D> {
            #[allow(unused)]
            pub fn new() -> Self {
                Sampler(std::marker::PhantomData)
            }
        }
        impl<D: Finite> super::Sampler<D> for Sampler<D> {
            type Iter<F: FnMut(&D) -> f64> = impl Iterator<Item = D>;
            fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Iter<F> {
                use std::ops::AddAssign;
                let xs: Vec<D> = D::traverse().collect();
                let ys: Vec<f64> = xs.iter().map(pdf).collect();
                let zs: Vec<f64> = ys
                    .iter()
                    .scan(0.0, |z, y| {
                        z.add_assign(y);
                        Some(z.clone())
                    })
                    .collect();

                let sum: f64 = ys.iter().sum();
                assert!(sum > 0.0, "pdf non-positive.");
                assert!(sum.is_finite(), "pdf overflow.");

                let mut aux = rand::thread_rng();
                std::iter::from_fn(move || {
                    use rand::Rng;
                    let aux = aux.gen_range(0.0..sum);
                    let pos = zs.binary_search_by(|z| z.partial_cmp(&aux).unwrap());
                    let pos = pos.unwrap_or_else(|pos| pos);
                    Some(xs[pos].clone())
                })
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn gaussian() {
                super::test::sample(
                    Icdf::<float::X<256>>::new(),
                    distribution::univar::gaussian(0.5, 0.2),
                );
            }
        }
    }
}

#[doc = "Sample from multiple correlated domain."]
pub mod multivar {
    use super::*;

    pub use metropolis::Sampler as Metropolis;
    // pub use gibbs::Sampler as Gibbs;

    #[doc = "Metropolis-Hausting Sampling Algorithm."]
    pub mod metropolis {
        use super::*;

        pub struct Sampler<D: Uniform, R: nd::Dimension>(nd::Array<D, R>);
        impl<D: Uniform, R: nd::Dimension> Sampler<D, R> {
            #[allow(unused)]
            pub fn new(init: nd::Array<D, R>) -> Self {
                Sampler(init)
            }
        }
        impl<D: Uniform, R: nd::Dimension> super::Sampler<nd::Array<D, R>> for Sampler<D, R> {
            type Iter<F: FnMut(&nd::Array<D, R>) -> f64> = impl Iterator<Item = nd::Array<D, R>>;
            fn sample<F: FnMut(&nd::Array<D, R>) -> f64>(&self, mut pdf: F) -> Self::Iter<F> {
                let mut state = self.0.clone();
                let (shape, ptr) = (state.raw_dim(), state.as_mut_ptr());

                let mut uniform = D::uniform();
                let mut accept = rand::thread_rng();
                std::iter::repeat_with(move || unsafe {
                    nd::ArrayViewMut::from_shape_ptr(shape.clone(), ptr).into_iter()
                })
                .flatten()
                .filter_map(move |value| {
                    use rand::Rng;
                    let old_p = pdf(&state);

                    let new_value = uniform.next().expect("uniform_sampler failed.");
                    let old_value = std::mem::replace(value, new_value);

                    let new_p = pdf(&state);
                    if accept.gen_range(0.0..=old_p) < new_p {
                        Some(state.clone()) /* Accepted. Return state. */
                    } else {
                        drop(std::mem::replace(value, old_value));
                        None /* Not accepted. Revert to old_value. */
                    }
                })
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn gaussian() {
                super::test::sample(
                    multivar::Metropolis::<f64, nd::Ix1>::new(nd::Array::zeros([2])),
                    distribution::multivar::gaussian(
                        na::vector![0.5, 0.5],
                        na::matrix![
                            0.01, 0.006;
                            0.006, 0.02;
                        ],
                    ),
                );
            }
        }
    }
}

#[doc = "Sampler adapters."]
pub mod adapter {
    use super::*;

    pub struct BurnIn<D, S: Sampler<D>>(std::marker::PhantomData<D>, S, usize);
    impl<D, S: Sampler<D>> BurnIn<D, S> {
        #[allow(unused)]
        pub fn new(sampler: S, skip: usize) -> Self {
            BurnIn(std::marker::PhantomData, sampler, skip)
        }
    }
    impl<D: Scalar, S: Sampler<D>> Sampler<D> for BurnIn<D, S> {
        type Iter<F: FnMut(&D) -> f64> = std::iter::Skip<S::Iter<F>>;
        fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Iter<F> {
            self.1.sample(pdf).skip(self.2)
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                                    TEST                                    */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod test {
    use super::*;
    use std::fmt::Display;

    pub fn sample<D: 'static + Display + Send>(sampler: impl Sampler<D>, pdf: impl Fn(&D) -> f64) {
        let (sender, receiver) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .open("target/sample.txt")
                .unwrap();
            loop {
                let x = receiver.recv().unwrap();
                writeln!(file, "{}", x).unwrap();
            }
        });

        use tqdm::Iter;
        sampler
            .sample(pdf)
            .tqdm()
            .for_each(|x| sender.send(x).unwrap());
    }
}
