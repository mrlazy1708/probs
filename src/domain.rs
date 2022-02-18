#[allow(unused_imports)]
use super::*;

pub trait Domain: nalgebra::Scalar {
    type Iter: Iterator<Item = Self>;
    fn uniform() -> <Self as Domain>::Iter;
}

pub trait Finite: Domain {
    type Iter: Iterator<Item = Self>;
    fn traverse() -> <Self as Finite>::Iter;
}

pub mod dimension {
    use super::*;

    pub mod fixed {
        use super::*;

        impl<D: Domain, const R: usize> Domain for nd::Array<D, nd::Dim<[usize; R]>>
        where
            nd::Dim<[usize; R]>: nd::Dimension,
        {
            type Iter = impl Iterator<Item = Self>;
            fn uniform() -> <Self as Domain>::Iter {
                std::iter::empty() // Never used!
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                                  PRE-IMPL                                  */
/* -------------------------------------------------------------------------- */

macro_rules! impl_domain {
    [$($Num: ty), *] => {
        $(
            impl Domain for $Num {
                type Iter = impl Iterator<Item = Self>;
                fn uniform() -> <Self as Domain>::Iter {
                    use rand::Rng;
                    let mut gen = rand::thread_rng();
                    std::iter::from_fn(move || Some(gen.gen::<$Num>()))
                }
            }
        )*
    };
}
impl_domain![u8, u16, u32, u64, u128, usize];
impl_domain![i8, i16, i32, i64, i128, isize];
impl_domain![f32, f64];

macro_rules! impl_finite {
    [$($Num: ty), *] => {
        $(
            impl Finite for $Num {
                type Iter = impl Iterator<Item = Self>;
                fn traverse() -> <Self as Finite>::Iter {
                    (<$Num>::MIN)..(<$Num>::MAX)
                }
            }
        )*
    };
}
impl_finite![u8, u16, u32, u64, u128, usize];
impl_finite![i8, i16, i32, i64, i128, isize];

pub mod integer {
    use super::*;
    use std::fmt::Debug;

    #[derive(Clone, PartialEq, Debug)]
    pub struct X<const N: usize>(pub usize);
    impl<const N: usize> num::ToPrimitive for X<N> {
        fn to_u64(&self) -> Option<u64> {
            self.0.to_u64()
        }
        fn to_i64(&self) -> Option<i64> {
            self.0.to_i64()
        }
    }

    impl<const N: usize> Domain for X<N> {
        type Iter = impl Iterator<Item = Self>;
        fn uniform() -> <Self as Domain>::Iter {
            use rand::Rng;
            let mut gen = rand::thread_rng();
            std::iter::from_fn(move || Some(X(gen.gen_range(0..N))))
        }
    }

    impl<const N: usize> Finite for X<N> {
        type Iter = impl Iterator<Item = Self>;
        fn traverse() -> <Self as Finite>::Iter {
            (0..N).map(X)
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn uniform() {
            for x in X::<256>::uniform() {
                assert!((0..256).contains(&x.0));
                println!("{:?}", x);
            }
        }

        #[test]
        fn traverse() {
            assert!(X::<256>::traverse().eq((0..256).map(X)));
        }
    }
}

pub mod float {
    use super::*;
    use std::fmt::Debug;

    #[derive(Clone, PartialEq, Debug)]
    pub struct X<const N: usize>(pub f64);
    impl<const N: usize> num::ToPrimitive for X<N> {
        fn to_u64(&self) -> Option<u64> {
            self.0.to_u64()
        }
        fn to_i64(&self) -> Option<i64> {
            self.0.to_i64()
        }

        fn to_f64(&self) -> Option<f64> {
            Some(self.0)
        }
    }

    impl<const N: usize> Domain for X<N> {
        type Iter = impl Iterator<Item = Self>;
        fn uniform() -> <Self as Domain>::Iter {
            use rand::Rng;
            let mut gen = rand::thread_rng();
            std::iter::from_fn(move || {
                let value = gen.gen_range(0.0..1.0);
                Some(X((value * N as f64).floor() / N as f64))
            })
        }
    }

    impl<const N: usize> Finite for X<N> {
        type Iter = impl Iterator<Item = Self>;
        fn traverse() -> <Self as Finite>::Iter {
            (0..N).map(|x| X(x as f64 / N as f64))
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn uniform() {
            for rv in X::<256>::uniform().take(100) {
                println!("{:?}", rv);
                assert_eq!(rv.0 % (1.0 / 256.0), 0.0);
            }
        }

        #[test]
        fn traverse() {
            let mut count = 0.0;
            for rv in X::<256>::traverse() {
                println!("{:?}", rv);
                assert!(count < 1.0);
                assert_eq!(rv.0, count);
                count += 1.0 / 256.0;
            }
        }
    }
}
