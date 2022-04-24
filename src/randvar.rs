use super::*;

#[doc = "Measurable Space"]
pub trait Domain: na::Scalar {
    type Iter: Iterator<Item = Self>;
    fn random() -> Self::Iter;
}

#[doc = "Discrete Random Variable"]
pub trait Discrete: na::Scalar {
    type Iter: Iterator<Item = Self>;
    fn iter() -> Self::Iter;
}

/* -------------------------------------------------------------------------- */
/*                                  PROVIDED                                  */
/* -------------------------------------------------------------------------- */

/* ------------------------------- Continuous ------------------------------- */

macro_rules! impl_domain {
        [$($Num: ty), *] => {
            $(
                impl Domain for $Num {
                    type Iter = impl Iterator<Item = Self>;
                    fn random() -> Self::Iter {
                        use rand::Rng;
                        let mut gen = rand::thread_rng();
                        std::iter::from_fn(move || Some(gen.gen_range(0.0..1.0)))
                    }
                }
            )*
        };
    }
impl_domain![f32, f64];

/* -------------------------------- Discrete -------------------------------- */

pub mod modular {
    use super::*;
    use std::fmt::*;

    #[derive(Clone, PartialEq, Debug)]
    pub struct Z<const N: usize>(pub usize);
    impl<const N: usize> Domain for Z<N> {
        type Iter = impl Iterator<Item = Self>;
        fn random() -> Self::Iter {
            use rand::Rng;
            let mut gen = rand::thread_rng();
            std::iter::from_fn(move || Some(gen.gen_range(0..N)).map(Z))
        }
    }
    impl<const N: usize> Discrete for Z<N> {
        type Iter = impl Iterator<Item = Self>;
        fn iter() -> Self::Iter {
            (0..N).map(Z)
        }
    }

    impl<const N: usize> num::ToPrimitive for Z<N> {
        fn to_u64(&self) -> Option<u64> {
            Some(self.0 as u64)
        }
        fn to_i64(&self) -> Option<i64> {
            Some(self.0 as i64)
        }
    }
}

// pub mod float {
//     use super::*;

//     #[derive(Clone, PartialEq, std::fmt::Debug)]
//     pub struct X<const N: usize>(pub f64);
//     impl<const N: usize> num::ToPrimitive for X<N> {
//         fn to_u64(&self) -> Option<u64> {
//             self.0.to_u64()
//         }
//         fn to_i64(&self) -> Option<i64> {
//             self.0.to_i64()
//         }

//         fn to_f64(&self) -> Option<f64> {
//             Some(self.0)
//         }
//     }

//     impl<const N: usize> std::fmt::Display for X<N> {
//         fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//             write!(f, "{}", self.0)
//         }
//     }

//     impl<const N: usize> Uniform for X<N> {
//         type Iter = impl Iterator<Item = Self>;
//         fn uniform() -> <Self as Uniform>::Iter {
//             use rand::Rng;
//             let mut gen = rand::thread_rng();
//             std::iter::from_fn(move || {
//                 let value = gen.gen_range(0.0..1.0);
//                 Some(X((value * N as f64).floor() / N as f64))
//             })
//         }
//     }

//     impl<const N: usize> Finite for X<N> {
//         type Iter = impl Iterator<Item = Self>;
//         fn traverse() -> <Self as Finite>::Iter {
//             (0..N).map(|x| X(x as f64 / N as f64))
//         }
//     }
// }
