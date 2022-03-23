use super::*;

/* -------------------------------------------------------------------------- */
/*                                 Detinition                                 */
/* -------------------------------------------------------------------------- */

pub use na::Scalar;

pub trait Uniform: Scalar {
    type Iter: Iterator<Item = Self>;
    fn uniform() -> <Self as Uniform>::Iter;
}

pub trait Finite: Scalar {
    type Iter: Iterator<Item = Self>;
    fn traverse() -> <Self as Finite>::Iter;
}

/* -------------------------------------------------------------------------- */
/*                                  PROVIDED                                  */
/* -------------------------------------------------------------------------- */

/* --------------------------------- Uniform -------------------------------- */

macro_rules! impl_uniform {
        [$($Num: ty), *] => {
            $(
                impl Uniform for $Num {
                    type Iter = impl Iterator<Item = Self>;
                    fn uniform() -> <Self as Uniform>::Iter {
                        use rand::Rng;
                        let mut gen = rand::thread_rng();
                        std::iter::from_fn(move || Some(gen.gen::<$Num>()))
                    }
                }
            )*
        };
    }
impl_uniform![u8, u16, u32, u64, u128, usize];
impl_uniform![i8, i16, i32, i64, i128, isize];
impl_uniform![f32, f64];

/* --------------------------------- Finite --------------------------------- */

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

/* -------------------------------- Remainder ------------------------------- */

pub mod integer {
    use super::*;

    #[derive(Clone, PartialEq, std::fmt::Debug)]
    pub struct X<const N: usize>(pub usize);
    impl<const N: usize> num::ToPrimitive for X<N> {
        fn to_u64(&self) -> Option<u64> {
            self.0.to_u64()
        }
        fn to_i64(&self) -> Option<i64> {
            self.0.to_i64()
        }
    }

    impl<const N: usize> std::fmt::Display for X<N> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl<const N: usize> Uniform for X<N> {
        type Iter = impl Iterator<Item = Self>;
        fn uniform() -> <Self as Uniform>::Iter {
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
}

pub mod float {
    use super::*;

    #[derive(Clone, PartialEq, std::fmt::Debug)]
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

    impl<const N: usize> std::fmt::Display for X<N> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl<const N: usize> Uniform for X<N> {
        type Iter = impl Iterator<Item = Self>;
        fn uniform() -> <Self as Uniform>::Iter {
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
}
