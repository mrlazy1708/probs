use std::fmt::Debug;

pub trait Domain: Clone + Copy + Debug {
    type Iter: Iterator<Item = Self>;
    fn uniform() -> <Self as Domain>::Iter;
}

pub trait FiniteDomain: Domain {
    type Iter: Iterator<Item = Self>;
    fn traverse() -> <Self as FiniteDomain>::Iter;
}

pub mod static_dim {
    use super::*;

    pub struct UniformIter<D: Domain, const R: usize>([<D as Domain>::Iter; R]);
    impl<D: Domain, const R: usize> Domain for [D; R] {
        type Iter = UniformIter<D, R>;
        fn uniform() -> <Self as Domain>::Iter {
            UniformIter(
                std::iter::repeat_with(|| D::uniform())
                    .collect::<Vec<_>>()
                    .try_into()
                    .ok()
                    .unwrap(),
            )
        }
    }
    impl<D: Domain, const R: usize> Iterator for UniformIter<D, R> {
        type Item = [D; R];
        fn next(&mut self) -> Option<Self::Item> {
            self.0
                .iter_mut()
                .map(|iter| iter.next())
                .collect::<Option<Vec<_>>>()
                .and_then(|values| values.try_into().ok())
        }
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

    pub struct UniformIter<R: rand::Rng, const N: usize>(R);
    impl<const N: usize> Domain for X<N> {
        type Iter = UniformIter<rand::rngs::ThreadRng, N>;
        fn uniform() -> <Self as Domain>::Iter {
            UniformIter(rand::thread_rng())
        }
    }
    impl<R: rand::Rng, const N: usize> Iterator for UniformIter<R, N> {
        type Item = X<N>;
        fn next(&mut self) -> Option<Self::Item> {
            let value = self.0.gen_range(0.0..1.0);
            Some(X((value * N as f64).floor() / N as f64))
        }
    }

    pub struct TraverseIter<const N: usize>(usize);
    impl<const N: usize> FiniteDomain for X<N> {
        type Iter = TraverseIter<N>;
        fn traverse() -> <Self as FiniteDomain>::Iter {
            TraverseIter(0)
        }
    }
    impl<const N: usize> Iterator for TraverseIter<N> {
        type Item = X<N>;
        fn next(&mut self) -> Option<Self::Item> {
            if self.0 < N {
                let result = Some(X(self.0 as f64 / N as f64));
                self.0 += 1;
                result
            } else {
                None
            }
        }
    }

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
