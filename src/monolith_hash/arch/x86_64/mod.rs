#[cfg(target_feature = "avx2")]
mod goldilocks_avx2;
#[cfg(target_feature = "avx2")]
mod utils_avx2;
#[cfg(target_feature = "avx2")]
pub(crate) mod monolith_avx2;