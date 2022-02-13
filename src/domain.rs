use std::fmt::Debug;

pub trait Domain: Clone + Copy + Debug {
    type Iter: Iterator<Item = Self>;
    fn random() -> Self::Iter;
}

#[doc = "Static dimensioned domain."]
impl<D: Domain, const R: usize> Domain for [D; R] {
    type Iter = std::iter::Empty<Self>;
    fn random() -> Self::Iter {
        panic!("Never used")
    }
}

/* -------------------------------------------------------------------------- */
/*                                    TEST                                    */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
pub mod test {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub struct X<const N: usize>(pub f64);
    impl<const N: usize> Domain for X<N> {
        type Iter = Iter<rand::rngs::ThreadRng, N>;
        fn random() -> Self::Iter {
            Iter(rand::thread_rng())
        }
    }

    pub struct Iter<R: rand::Rng, const N: usize>(R);
    impl<R: rand::Rng, const N: usize> Iterator for Iter<R, N> {
        type Item = X<N>;
        fn next(&mut self) -> Option<Self::Item> {
            let value = self.0.gen_range(0.0..1.0);
            Some(X((value * N as f64).floor() / N as f64))
        }
    }

    #[test]
    fn random() {
        for rv in X::<256>::random().take(100) {
            println!("{:?}", rv);
            assert_eq!(rv.0 % (1.0 / 256.0), 0.0);
        }
    }
}
