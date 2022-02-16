#[allow(unused_imports)]
use super::{domain, domain::*};

extern crate nalgebra as na;
extern crate ndarray as nd;
extern crate nshare as ns;

pub trait Global<D: Domain>: Clone {
    type Sampler<F: FnMut(&D) -> f64>: Iterator<Item = D>;
    fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> <Self as Global<D>>::Sampler<F>;

    fn gibbs<const R: usize>(
        self,
        dim: ndarray::Dim<[usize; R]>,
        skip: usize,
    ) -> multivar::Gibbs<D, Self, R> {
        multivar::Gibbs::new(self, dim, skip)
    }
}

pub trait Local<D: Domain + std::ops::Index<I>, I>: Global<D> {
    type Sampler<F: FnMut(&D, I) -> f64>: Iterator<Item = D>;
    fn sample<F: FnMut(&D, I) -> f64>(&self, pdf: F) -> <Self as Local<D, I>>::Sampler<F>;
}

#[doc = "Sample from certain domain."]
pub mod univar {
    use super::*;

    pub use icdf::Method as Icdf;
    pub use slice::Method as Slice;

    #[doc = "Inverse Transform Sampling."]
    pub mod icdf {
        use super::*;
        use std::marker::PhantomData;

        #[derive(Clone)]
        pub struct Method<D: FiniteDomain>(PhantomData<(D,)>);
        impl<D: FiniteDomain> Method<D> {
            #[allow(unused)]
            pub fn new() -> Self {
                Method(PhantomData)
            }
        }
        impl<D: FiniteDomain> Global<D> for Method<D> {
            type Sampler<F: FnMut(&D) -> f64> = Sampler<D>;
            fn sample<F: FnMut(&D) -> f64>(&self, mut pdf: F) -> Self::Sampler<F> {
                Sampler {
                    sum: rand::thread_rng(),
                    cdf: D::traverse()
                        .map(|x| (x.clone(), pdf(&x)))
                        .scan(0.0, |c, (x, p)| {
                            *c = *c + p;
                            assert!(c.is_finite(), "PDF overflow");
                            Some((x, *c))
                        })
                        .collect(),
                }
            }
        }

        pub struct Sampler<D: FiniteDomain> {
            pub sum: rand::rngs::ThreadRng,
            pub cdf: Vec<(D, f64)>,
        }
        impl<D: FiniteDomain> Iterator for Sampler<D> {
            type Item = D;
            fn next(&mut self) -> Option<Self::Item> {
                let sum = self.cdf.last().unwrap().1;
                match sum {
                    sum if sum > 0.0 => {
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
            use super::*;
            use domain::test::X;

            #[test]
            fn burst() {
                use tqdm::Iter;
                Icdf::<X<256>>::new()
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
                        Icdf::<X<256>>::new()
                            .sample(univar::test::normal::<256>(0.5, 0.2))
                            .next()
                    })
                    .for_each(drop);
            }

            #[test]
            fn ill_shaped() {
                Icdf::<X<256>>::new()
                    .sample(|&X(x)| {
                        if x == 0.0 / 256.0 || x == 255.0 / 256.0 {
                            1.0000
                        } else {
                            0.0001
                        }
                    })
                    .take(100)
                    .for_each(|X(x)| println!("{:?}", x));
            }
        }
    }

    #[doc = "Slice Sampling."]
    pub mod slice {
        use super::*;
        use std::marker::PhantomData;

        #[derive(Clone)]
        pub struct Method<D: Domain>(PhantomData<(D,)>, usize);
        impl<D: Domain> Method<D> {
            #[allow(unused)]
            pub fn new(skip: usize) -> Self {
                Method(PhantomData, skip)
            }
        }
        impl<D: Domain> Global<D> for Method<D> {
            type Sampler<F: FnMut(&D) -> f64> = std::iter::Skip<Sampler<D, F>>;
            fn sample<F: FnMut(&D) -> f64>(&self, pdf: F) -> Self::Sampler<F> {
                Sampler {
                    aux: rand::thread_rng(),
                    pdf,

                    state: D::uniform().next(),
                }
                .skip(self.1)
            }
        }

        pub struct Sampler<D: Domain, F: FnMut(&D) -> f64> {
            pub aux: rand::rngs::ThreadRng,
            pub pdf: F,

            pub state: Option<D>,
        }
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
            use super::*;
            use domain::test::X;

            #[test]
            fn burst() {
                use tqdm::Iter;
                Slice::<X<256>>::new(100)
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
                        Slice::<X<256>>::new(100)
                            .sample(univar::test::normal::<256>(0.5, 0.2))
                            .next()
                    })
                    .for_each(drop);
            }

            #[test]
            fn ill_shaped() {
                Slice::<X<256>>::new(1000)
                    .sample(|&X(x)| {
                        if x == 0.0 / 256.0 || x == 255.0 / 256.0 {
                            1.0000
                        } else {
                            0.0001
                        }
                    })
                    .take(100)
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

        #[derive(Clone)]
        pub struct Method<D: Domain, S: Global<D>, const R: usize>(
            PhantomData<(D,)>,
            S,
            nd::Dim<[usize; R]>,
            usize,
        );
        impl<D: Domain, S: Global<D>, const R: usize> Method<D, S, R> {
            #[allow(unused)]
            pub fn new(sub: S, dim: nd::Dim<[usize; R]>, skip: usize) -> Self {
                Method(PhantomData, sub, dim, skip)
            }
        }
        impl<D: Domain, S: Global<D>, const R: usize> Global<nd::Array<D, nd::Dim<[usize; R]>>>
            for Method<D, S, R>
        where
            nd::Dim<[usize; R]>: nd::Dimension,
        {
            type Sampler<F: FnMut(&nd::Array<D, nd::Dim<[usize; R]>>) -> f64> =
                std::iter::Skip<global::Sampler<D, S, F, R>>;
            fn sample<F: FnMut(&nd::Array<D, nd::Dim<[usize; R]>>) -> f64>(
                &self,
                pdf: F,
            ) -> Self::Sampler<F> {
                let mut init = D::uniform();
                global::Sampler {
                    sub: self.1.clone(),
                    pdf,

                    state: Some(nd::Array::from_shape_fn(self.2, |_| init.next().unwrap())),
                }
                .skip(self.3)
            }
        }

        pub mod global {
            use super::*;

            pub struct Sampler<
                D: Domain,
                S: Global<D>,
                F: FnMut(&nd::Array<D, nd::Dim<[usize; R]>>) -> f64,
                const R: usize,
            > {
                pub sub: S,
                pub pdf: F,

                pub state: Option<nd::Array<D, nd::Dim<[usize; R]>>>,
            }
            impl<
                    D: Domain,
                    S: Global<D>,
                    F: FnMut(&nd::Array<D, nd::Dim<[usize; R]>>) -> f64,
                    const R: usize,
                > Iterator for Sampler<D, S, F, R>
            where
                nd::Dim<[usize; R]>: nd::Dimension,
            {
                type Item = nd::Array<D, nd::Dim<[usize; R]>>;
                fn next(&mut self) -> Option<Self::Item> {
                    if let Some(mut state) = Option::take(&mut self.state) {
                        let new_state: Option<Vec<D>> = unsafe {
                            let ptr = state.as_mut_ptr();
                            nd::ArrayViewMut::from_shape_ptr(state.raw_dim(), ptr)
                                .iter_mut()
                                .map(|value| {
                                    let new = self
                                        .sub
                                        .sample(|x| {
                                            *value = x.clone();
                                            (self.pdf)(&state)
                                        })
                                        .next()
                                        .or_else(|| D::uniform().next());
                                    if let Some(new) = new.as_ref() {
                                        *value = new.clone();
                                    }

                                    new
                                })
                                .collect()
                        };

                        if let Some(new_state) = new_state {
                            self.state = nd::Array::from_shape_vec(state.raw_dim(), new_state).ok();
                        }
                    }

                    self.state.clone()
                }
            }
        }

        #[cfg(test)]
        pub mod test {
            use super::*;
            use domain::test::X;

            #[test]
            fn dim2() {
                univar::Icdf::<X<256>>::new()
                    .gibbs(nd::Dim([2]), 100)
                    .sample(multivar::test::normal::<256, 2>(
                        na::vector![0.5, 0.5],
                        na::matrix![
                            0.01, 0.006;
                            0.006, 0.02;
                        ],
                    ))
                    .take(1000)
                    .for_each(|xs| println!("{:?}", xs.map(|X(x)| x)));
            }

            #[test]
            fn ill_shaped() {
                univar::Icdf::<X<10>>::new()
                    .gibbs(nd::Dim([100]), 100)
                    .sample(|xs| {
                        xs.iter()
                            .map(|&X(x)| (if x == 0.5 { 99.0 } else { -1.0 }) * 0.05)
                            .sum::<f64>()
                            .exp()
                    })
                    .take(100)
                    .for_each(|xs| println!("{:?}", xs.map(|X(x)| x)));
            }
        }
    }

    #[cfg(test)]
    pub mod test {
        use super::*;
        use domain::test::X;

        pub fn normal<const N: usize, const R: usize>(
            mu: na::SVector<f64, R>,
            sigma: na::SMatrix<f64, R, R>,
        ) -> impl Fn(&nd::Array<X<N>, nd::Dim<[usize; 1]>>) -> f64 {
            let isigma = sigma.try_inverse().unwrap();
            move |xs| {
                use ns::ToNalgebra;
                let xs = xs.map(|&X(x)| x).into_nalgebra() - mu;
                (-(isigma * xs).dot(&xs) / 2.0).exp()
            }
        }
    }
}
