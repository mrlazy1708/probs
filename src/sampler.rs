use super::*;

pub trait Sampler<D: na::Scalar> {
    type Iter<F: FnMut(&D) -> f64>: Iterator<Item = D>;
    fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Iter<F>;

    fn burn(self, skip: usize) -> adapter::Burn<D, Self>
    where
        Self: Sized,
    {
        adapter::Burn::new(self, skip)
    }

    fn pick(self, interval: usize) -> adapter::Pick<D, Self>
    where
        Self: Sized,
    {
        adapter::Pick::new(self, interval)
    }

    fn gibbs<R: nd::Dimension>(self, dim: R) -> multivar::Gibbs<D, R, Self>
    where
        Self: Sized,
        D: Domain,
    {
        multivar::Gibbs::new(dim, self)
    }
}

#[doc = "Sample from univariate domain"]
pub mod univar {
    use super::*;

    pub use icdf::Sampler as Icdf;
    pub use metropolis::Sampler as Metropolis;

    #[doc = "Inverse Transform Sampling"]
    pub mod icdf {
        use super::*;

        pub struct Sampler<D: Domain + Discrete> {
            pd: std::marker::PhantomData<D>,
        }
        impl<D: Domain + Discrete> Sampler<D> {
            #[allow(unused)]
            pub fn new() -> Self {
                Sampler {
                    pd: std::marker::PhantomData,
                }
            }
        }
        impl<D: Domain + Discrete> super::Sampler<D> for Sampler<D> {
            type Iter<F: FnMut(&D) -> f64> = impl Iterator<Item = D>;
            fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Iter<F> {
                use std::ops::AddAssign;
                let xs: Vec<D> = D::iter().collect();
                let ys: Vec<f64> = xs.iter().map(pdf).collect();
                let zs: Vec<f64> = ys
                    .iter()
                    .scan(0.0, |z, y| {
                        z.add_assign(y);
                        Some(z.clone())
                    })
                    .collect();

                let sum: f64 = ys.iter().sum();
                assert!(sum > 0.0, "pdf isn't positive");
                assert!(sum.is_finite(), "pdf overflow");

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
            use modular::*;

            #[test]
            fn gaussian() {
                super::test::sample(
                    univar::Icdf::<Z<256>>::new(),
                    dist::univar::gaussian(128.0, 32.0),
                );
            }
        }
    }

    #[doc = "Metropolis-Hausting Sampling"]
    pub mod metropolis {
        use super::*;
        use std::sync::*;

        pub struct Sampler<D: Domain, P: Fn(&D) -> D> {
            pd: std::marker::PhantomData<D>,
            pub proposal: Arc<P>,
        }
        impl<D: Domain, P: Fn(&D) -> D> Sampler<D, P> {
            #[allow(unused)]
            pub fn new(proposal: P) -> Self {
                Sampler {
                    pd: std::marker::PhantomData,
                    proposal: Arc::new(proposal),
                }
            }
        }
        impl<D: Domain, P: Fn(&D) -> D> super::Sampler<D> for Sampler<D, P> {
            type Iter<F: FnMut(&D) -> f64> = impl Iterator<Item = D>;
            fn sample<F: FnMut(&D) -> f64>(&self, mut pdf: F) -> Self::Iter<F> {
                let proposal = self.proposal.clone();
                let mut state = D::random().next().unwrap();
                let mut prob = pdf(&state);

                let mut aux = rand::thread_rng();
                std::iter::from_fn(move || {
                    let new_state = proposal(&state);
                    let new_prob = pdf(&new_state);

                    use rand::Rng;
                    let aux = aux.gen_range(0.0..1.0);
                    if aux <= new_prob / prob {
                        state = new_state;
                        prob = new_prob;
                    }

                    Some(state.clone())
                })
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use modular::*;

            #[test]
            fn gaussian() {
                super::test::sample(
                    univar::Metropolis::new(|&_| Z::<256>::random().next().unwrap()),
                    dist::univar::gaussian(128.0, 32.0),
                );
            }
        }
    }
}

#[doc = "Sample from multiple correlated domain"]
pub mod multivar {
    use super::*;

    pub use gibbs::Sampler as Gibbs;

    #[doc = "Gibbs Sampling Algorithm"]
    pub mod gibbs {
        use super::*;
        use std::sync::*;

        pub struct Sampler<D: Domain, R: nd::Dimension + 'static, S: super::Sampler<D>> {
            pd: std::marker::PhantomData<D>,
            pub dim: R,
            pub sampler: Arc<S>,
        }
        impl<D: Domain, R: nd::Dimension + 'static, S: super::Sampler<D>> Sampler<D, R, S> {
            #[allow(unused)]
            pub fn new(dim: R, sampler: S) -> Self {
                Sampler {
                    pd: std::marker::PhantomData,
                    dim,
                    sampler: Arc::new(sampler),
                }
            }
        }
        impl<D: Domain, R: nd::Dimension + 'static, S: super::Sampler<D>>
            super::Sampler<nd::Array<D, R>> for Sampler<D, R, S>
        {
            type Iter<F: FnMut(&nd::Array<D, R>) -> f64> = impl Iterator<Item = nd::Array<D, R>>;
            fn sample<F: FnMut(&nd::Array<D, R>) -> f64>(&self, mut pdf: F) -> Self::Iter<F> {
                let mut init = D::random();
                let mut state =
                    nd::Array::from_shape_fn(self.dim.clone(), |_| init.next().unwrap());

                let sampler = self.sampler.clone();
                let (dim, ptr) = (state.raw_dim(), state.as_mut_ptr());

                std::iter::repeat_with(move || unsafe {
                    nd::ArrayViewMut::from_shape_ptr(dim.clone(), ptr).into_iter()
                })
                .flatten()
                .map(move |old_value| {
                    let new_value = sampler
                        .sample(|value| {
                            drop(std::mem::replace(old_value, value.clone()));
                            pdf(&state)
                        })
                        .next()
                        .unwrap();
                    drop(std::mem::replace(old_value, new_value));
                    state.clone()
                })
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use modular::*;

            #[test]
            fn gaussian() {
                use sampler::Sampler;
                super::test::sample(
                    univar::Icdf::<Z<256>>::new().gibbs(nd::Dim([2])).burn(1000),
                    dist::multivar::gaussian(
                        na::vector![128.0, 128.0],
                        na::matrix![
                            128.0, 32.0;
                            32.0, 64.0;
                        ],
                    ),
                );
            }
        }
    }
}

#[doc = "Sampler adapters"]
pub mod adapter {
    use super::*;

    pub use burn::Sampler as Burn;
    pub use pick::Sampler as Pick;

    #[doc = "Discard non-equilibrium samples"]
    pub mod burn {
        use super::*;

        pub struct Sampler<D: na::Scalar, S: super::Sampler<D>> {
            pd: std::marker::PhantomData<D>,
            pub sampler: S,
            pub skip: usize,
        }
        impl<D: na::Scalar, S: super::Sampler<D>> Sampler<D, S> {
            #[allow(unused)]
            pub fn new(sampler: S, skip: usize) -> Self {
                Sampler {
                    pd: std::marker::PhantomData,
                    sampler,
                    skip,
                }
            }
        }
        impl<D: na::Scalar, S: super::Sampler<D>> super::Sampler<D> for Sampler<D, S> {
            type Iter<F: FnMut(&D) -> f64> = impl Iterator<Item = D>;
            fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Iter<F> {
                self.sampler.sample(pdf).skip(self.skip)
            }
        }
    }

    #[doc = "Pick samples over intervals"]
    pub mod pick {
        use super::*;

        pub struct Sampler<D: na::Scalar, S: super::Sampler<D>> {
            pd: std::marker::PhantomData<D>,
            pub sampler: S,
            pub interval: usize,
        }
        impl<D: na::Scalar, S: super::Sampler<D>> Sampler<D, S> {
            #[allow(unused)]
            pub fn new(sampler: S, interval: usize) -> Self {
                Sampler {
                    pd: std::marker::PhantomData,
                    sampler,
                    interval,
                }
            }
        }
        impl<D: na::Scalar, S: super::Sampler<D>> super::Sampler<D> for Sampler<D, S> {
            type Iter<F: FnMut(&D) -> f64> = impl Iterator<Item = D>;
            fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Iter<F> {
                let mut sampler = self.sampler.sample(pdf);
                let interval = self.interval;
                std::iter::from_fn(move || {
                    (1..interval).for_each(|_| drop(sampler.next()));
                    sampler.next()
                })
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                                    TEST                                    */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod test {
    use super::*;

    pub fn sample<D: na::Scalar + Send>(sampler: impl Sampler<D>, pdf: impl Fn(&D) -> f64) {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .open("target/test.sample.txt")
                .unwrap();

            loop {
                let x = receiver.recv().unwrap();
                writeln!(file, "{:?}", x).unwrap();
            }
        });

        use tqdm::Iter;
        sampler
            .sample(pdf)
            .tqdm()
            .for_each(|x| sender.send(x).unwrap());
    }
}
