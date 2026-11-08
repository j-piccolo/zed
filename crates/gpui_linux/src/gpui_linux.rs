#![cfg(any(target_os = "linux", target_os = "freebsd"))]
mod linux;

pub use linux::{current_platform, linux_platform};
/// myshell: shaped and grabbing layer windows.
#[cfg(feature = "wayland")]
pub use linux::myshell;
