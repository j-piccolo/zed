use thiserror::Error;

/// What the compositor says about a session lock asked for with
/// [`App::lock_session`](crate::App::lock_session).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SessionLockEvent {
    /// The session is locked: every output shows this app's lock surfaces, or nothing at all.
    /// A screen locker can report itself ready from here.
    Locked,
    /// The compositor has taken the unlock asked for with
    /// [`App::unlock_session`](crate::App::unlock_session). A screen locker can exit from here:
    /// exiting any sooner may leave the session locked.
    Unlocked,
    /// The lock is over without [`App::unlock_session`](crate::App::unlock_session): the
    /// compositor refused it, for instance because another client holds a lock, or ended it.
    /// Lock surface windows stay open until the app closes them, and show nothing.
    Finished,
}

/// An error indicating that an action failed because the compositor doesn't support the
/// `ext_session_lock_v1` protocol.
#[derive(Debug, Error)]
#[error("Compositor doesn't support ext_session_lock_v1")]
pub struct SessionLockNotSupportedError;

/// An error indicating that the app asked for a second session lock while it still holds one.
#[derive(Debug, Error)]
#[error("This app already holds a session lock")]
pub struct SessionAlreadyLockedError;

/// An error indicating that a [`WindowKind::LockSurface`](crate::WindowKind::LockSurface) window
/// was opened without a session lock, or without a display to cover.
#[derive(Debug, Error)]
pub enum LockSurfaceError {
    /// No session lock: call [`App::lock_session`](crate::App::lock_session) first.
    #[error("A lock surface needs a session lock; call App::lock_session first")]
    NotLocking,
    /// A lock surface covers one output, named by `WindowOptions::display_id`.
    #[error("A lock surface needs a display, from WindowOptions::display_id")]
    NoDisplay,
}
