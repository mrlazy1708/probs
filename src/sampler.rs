use super::{domain, domain::*};

pub trait Method<D: Domain>: Default {
    type Sampler<F: FnMut(&D) -> f64>: Iterator<Item = D>;
    fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Sampler<F>;

    fn gibbs<const R: usize>(self) -> multivar::gibbs::Method<D, Self, R> {
        multivar::gibbs::Method::<D, Self, R>::default()
    }
}

pub trait Sampler<D: Domain>: Iterator<Item = D> {}

#[doc = "Sample from certain domain."]
pub mod univar {
    use super::*;

    pub use icdf::Method as Icdf;
    pub use slice::Method as Slice;

    #[doc = "Inverse Transform Sampling."]
    pub mod icdf {
        use super::*;
        use std::marker::PhantomData;

        pub struct Method<D: FiniteDomain>(PhantomData<(D,)>);
        impl<D: FiniteDomain> Default for Method<D> {
            fn default() -> Self {
                Self(PhantomData)
            }
        }
        impl<D: FiniteDomain> super::Method<D> for Method<D> {
            type Sampler<F: FnMut(&D) -> f64> = Sampler<D>;
            fn sample<F: FnMut(&D) -> f64>(&self, mut pdf: F) -> Self::Sampler<F> {
                Sampler {
                    sum: rand::thread_rng(),
                    cdf: D::traverse()
                        .map(|x| (x.clone(), pdf(&x)))
                        .scan(0.0, |c, (x, p)| {
                            *c = *c + p;
                            Some((x, *c))
                        })
                        .collect(),
                }
            }
        }

        pub struct Sampler<D: FiniteDomain> {
            sum: rand::rngs::ThreadRng,
            cdf: Vec<(D, f64)>,
        }
        impl<D: FiniteDomain> super::Sampler<D> for Sampler<D> {}
        impl<D: FiniteDomain> Iterator for Sampler<D> {
            type Item = D;
            fn next(&mut self) -> Option<Self::Item> {
                let sum = self.cdf.last().unwrap().1;
                match sum {
                    sum if sum > 0.0 && sum.is_finite() => {
                        use rand::Rng;
                        let value = self.sum.gen_range(0.0..sum);
                        let pos = self
                            .cdf
                            .binary_search_by(|(_value, p)| p.partial_cmp(&value).unwrap());

                        self.cdf
                            .get(pos.unwrap_or_else(|pos| pos))
                            .map(|(value, _p)| value.clone())
                    }
                    _ => None, // in case of p(x) === 0.0 or NaN
                }
            }
        }

        #[cfg(test)]
        pub mod test {
            use super::{super::Method, *};
            use domain::test::X;

            #[test]
            fn burst() {
                use tqdm::Iter;
                icdf::Method::<X<256>>::default()
                    .sample(univar::test::normal::<256>(0.5, 0.2))
                    .take(100000)
                    .tqdm()
                    .for_each(drop);
            }

            #[test]
            fn oneshot() {
                use tqdm::Iter;
                (0..100000)
                    .tqdm()
                    .map(|_| {
                        icdf::Method::<X<256>>::default()
                            .sample(univar::test::normal::<256>(0.5, 0.2))
                            .next()
                    })
                    .for_each(drop);
            }
        }
    }

    #[doc = "Slice Sampling."]
    pub mod slice {
        use super::*;
        use std::marker::PhantomData;

        pub struct Method<D: Domain, const N: usize>(PhantomData<(D,)>);
        impl<D: Domain, const N: usize> Default for Method<D, N> {
            fn default() -> Self {
                Self(PhantomData)
            }
        }
        impl<D: Domain, const N: usize> super::Method<D> for Method<D, N> {
            type Sampler<F: FnMut(&D) -> f64> = std::iter::Skip<Sampler<D, F>>;
            fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Sampler<F> {
                Sampler {
                    aux: rand::thread_rng(),
                    pdf,

                    state: D::uniform().next(),
                }
                .skip(N)
            }
        }

        pub struct Sampler<D: Domain, F: FnMut(&D) -> f64> {
            aux: rand::rngs::ThreadRng,
            pdf: F,

            state: Option<D>,
        }
        impl<D: Domain, F: FnMut(&D) -> f64> super::Sampler<D> for Sampler<D, F> {}
        impl<D: Domain, F: FnMut(&D) -> f64> Iterator for Sampler<D, F> {
            type Item = D;
            fn next(&mut self) -> Option<Self::Item> {
                self.state = self
                    .state
                    .take()
                    .or_else(|| D::uniform().next())
                    .and_then(|x| match (self.pdf)(&x) {
                        y if y > 0.0 && y.is_finite() => {
                            use rand::Rng;
                            let y = self.aux.gen_range(0.0..=y);
                            D::uniform().skip_while(|x| (self.pdf)(x) < y).next()
                        }
                        _ => self.next(), // in case of p(x) == 0.0 or NaN
                    });

                self.state.clone()
            }
        }

        #[cfg(test)]
        pub mod test {
            use super::{super::Method, *};
            use domain::test::X;

            #[test]
            fn burst() {
                use tqdm::Iter;
                slice::Method::<X<256>, 100>::default()
                    .sample(univar::test::normal::<256>(0.5, 0.2))
                    .take(100000)
                    .tqdm()
                    .for_each(drop);
            }

            #[test]
            fn oneshot() {
                use tqdm::Iter;
                (0..100000)
                    .tqdm()
                    .map(|_| {
                        slice::Method::<X<256>, 100>::default()
                            .sample(univar::test::normal::<256>(0.5, 0.2))
                            .next()
                    })
                    .for_each(drop);
            }

            #[test]
            fn ill_shaped() {
                slice::Method::<X<256>, 100>::default()
                    .sample(|&X(x)| {
                        if x == 0.0 / 256.0 || x == 255.0 / 256.0 {
                            1.0000
                        } else {
                            0.0001
                        }
                    })
                    .skip(1000)
                    .take(1000)
                    .for_each(|X(x)| println!("{:?}", x));
            }
        }
    }

    #[cfg(test)]
    pub mod test {
        use super::*;
        use domain::test::X;

        pub fn normal<const N: usize>(mu: f64, sigma: f64) -> impl Fn(&X<N>) -> f64 {
            move |&X(x)| (-(x - mu).powi(2) / (2.0 * sigma.powi(2))).exp()
        }
    }
}

#[doc = "Sample from multiple correlated domain."]
pub mod multivar {
    use super::*;

    pub use gibbs::Method as Gibbs;

    #[doc = "Gibbs Sampling with static dimension."]
    pub mod gibbs {
        use super::*;
        use std::marker::PhantomData;

        use std::convert::TryInto;

        pub struct Method<D: Domain, S: super::Method<D>, const R: usize>(PhantomData<(D, S)>);
        impl<D: Domain, S: super::Method<D>, const R: usize> Default for Method<D, S, R> {
            fn default() -> Self {
                Self(PhantomData)
            }
        }
        impl<D: Domain, S: super::Method<D>, const R: usize> super::Method<[D; R]> for Method<D, S, R> {
            type Sampler<F: FnMut(&[D; R]) -> f64> = Sampler<D, S, F, R>;
            fn sample<F: FnMut(&[D; R]) -> f64>(&self, pdf: F) -> Self::Sampler<F> {
                Sampler {
                    sub: PhantomData,
                    pdf,

                    state: D::uniform().take(R).collect::<Vec<D>>().try_into().ok(),
                }
            }
        }

        pub struct Sampler<D: Domain, S: super::Method<D>, F: FnMut(&[D; R]) -> f64, const R: usize> {
            sub: PhantomData<S>,
            pdf: F,

            state: Option<[D; R]>,
        }
        impl<D: Domain, S: super::Method<D>, F: FnMut(&[D; R]) -> f64, const R: usize>
            super::Sampler<[D; R]> for Sampler<D, S, F, R>
        {
        }
        impl<D: Domain, S: super::Method<D>, F: FnMut(&[D; R]) -> f64, const R: usize> Iterator
            for Sampler<D, S, F, R>
        {
            type Item = [D; R];
            fn next(&mut self) -> Option<Self::Item> {
                let state: Option<[D; R]> = self.state.take();
                self.state = state.and_then(|mut state| {
                    (0..R)
                        .flat_map(|index| {
                            let new = S::default()
                                .sample(|x: &D| {
                                    state[index] = x.clone();
                                    (self.pdf)(&state)
                                })
                                .next()
                                .or_else(|| D::uniform().next());
                            if let Some(new) = new.as_ref() {
                                state[index] = new.clone();
                            }

                            new
                        })
                        .collect::<Vec<D>>()
                        .try_into()
                        .ok()
                });

                self.state.clone()
            }
        }

        #[cfg(test)]
        pub mod test {
            use super::{super::Method, *};
            use domain::test::X;

            #[test]
            fn dim2() {
                univar::slice::Method::<X<256>, 100>::default()
                    .gibbs::<2>()
                    .sample(multivar::test::normal::<256, 2>(
                        [0.5, 0.5],
                        [[0.01, 0.006], [0.006, 0.02]],
                    ))
                    .take(1000)
                    .for_each(|xs| println!("{:?}", xs.map(|X(x)| x)));
            }
        }
    }

    #[cfg(test)]
    pub mod test {
        use super::*;

        use domain::test::X;
        pub fn normal<const N: usize, const R: usize>(
            mu: [f64; R],
            sigma: [[f64; R]; R],
        ) -> impl Fn(&[X<N>; R]) -> f64 {
            extern crate nalgebra as na;
            let mu: na::SVector<f64, R> = na::SVector::from(mu);
            let sigma: na::SMatrix<f64, R, R> = na::SMatrix::from(sigma);
            let isigma = sigma.try_inverse().unwrap();
            move |xs| {
                let xs: [f64; R] = xs.map(|X(x)| x);
                let xs: na::SVector<f64, R> = na::SVector::from(xs);
                (-(isigma * (xs - mu)).dot(&(xs - mu)) / 2.0).exp()
            }
        }
    }
}
