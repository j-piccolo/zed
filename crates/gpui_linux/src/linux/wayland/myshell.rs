//! myshell: what the shell asks of a layer window beyond gpui's
//! `PlatformWindow`, by gpui `WindowId`.
//!
//! myshell draws its rail sheets in one kept layer surface per bar, larger
//! than any sheet, hidden and shown with `minimize`/`activate` (see
//! `window.rs`) instead of opening an `xdg_popup` per sheet. Two things an
//! `xdg_popup` gave for free need saying here:
//!
//! - Its shape. [`set_layer_view`] gives the rect the sheet covers: the
//!   layer surface becomes that rect, showing the window's top-left (where
//!   the sheet is drawn) through a `wp_viewport` crop, so input, blur and
//!   what the compositor composites are the sheet's alone, as the popup's
//!   were. It goes out with the frame being drawn, so a sheet's first frame
//!   has its place, blur and input with it.
//! - Its grab. A layer surface cannot take one, so [`grab_when_shown`] has
//!   an invisible 1x1 `xdg_popup` child of it take the grab once it is on
//!   screen (a grab's root must be mapped). The compositor then treats it as
//!   it treated the sheet popup: a press on another client's surface ends
//!   the grab (and still reaches that surface), and keys go to the popup,
//!   which are routed to the sheet's window. The grab ending (`popup_done`)
//!   runs the caller's `on_end`. Hiding the window drops the grab without
//!   it.
//!
//! A hidden layer window shrinks its swapchain to 1x1 (window-sized images
//! are RAM on Asahi, and a kept window is mostly hidden) and grows it back
//! when shown, which costs allocating them before its first frame.
//! [`keep_swapchain_when_hidden`] opts a window whose show must be fastest
//! (the launcher) out.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};

use gpui::{Bounds, Pixels, WindowId};
use wayland_backend::client::ObjectId;
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    protocol::{wl_buffer, wl_shm, wl_surface},
};
use wayland_protocols::xdg::shell::client::{xdg_popup, xdg_surface};
use wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1;

use super::client::{Globals, WaylandClientStatePtr};
use super::window::WaylandWindowStatePtr;

thread_local! {
    /// Views asked for and not yet drawn, per window.
    static SHAPES: RefCell<HashMap<WindowId, Option<Bounds<Pixels>>>> = RefCell::default();
    /// Grabs asked for, waiting for their window to be on screen.
    static REQUESTS: RefCell<HashMap<WindowId, Box<dyn FnOnce()>>> = RefCell::default();
    /// Grabs held, by the invisible popups that hold them.
    static GRABS: RefCell<Vec<Grab>> = const { RefCell::new(Vec::new()) };
    /// Windows that keep their swapchain at full size while hidden.
    static KEEP_SWAPCHAIN: RefCell<HashSet<WindowId>> = RefCell::default();
}

/// Keep layer window `window`'s swapchain at its size while it is hidden,
/// so showing it allocates nothing. A window opened hidden has already
/// shrunk it: its first show grows it, and it stays from then on.
pub fn keep_swapchain_when_hidden(window: WindowId) {
    KEEP_SWAPCHAIN.with(|keep| keep.borrow_mut().insert(window));
}

pub(crate) fn keeps_swapchain(window: WindowId) -> bool {
    KEEP_SWAPCHAIN.with(|keep| keep.borrow().contains(&window))
}

/// Show only the top-left `view.size` of layer window `window`, as a layer
/// surface that size at `view.origin` (logical px from the output's
/// top-left; the window's exclusive zone should be -1). The compositor sees
/// a surface just that size: input, blur (rounded by the window's client
/// corner radii) and its share of the screen. The window keeps its own
/// size and swapchain, so a view can move and resize from frame to frame at
/// no cost. An empty view shows nothing that takes input or blur; `None`
/// gives back the surface as opened. Applies with the window's next frame.
pub fn set_layer_view(window: WindowId, view: Option<Bounds<Pixels>>) {
    SHAPES.with(|shapes| shapes.borrow_mut().insert(window, view));
}

/// Take a popup grab for layer window `window` once it is shown (now, if it
/// is on screen: from its next compositor frame). `on_end` runs when the
/// compositor ends the grab; not when the window is hidden or closed.
/// Replaces a grab asked for and not started yet.
pub fn grab_when_shown(window: WindowId, on_end: impl FnOnce() + 'static) {
    REQUESTS.with(|requests| requests.borrow_mut().insert(window, Box::new(on_end)));
}

/// Whether layer window `window` holds a grab or has one on its way.
pub fn has_grab(window: WindowId) -> bool {
    REQUESTS.with(|requests| requests.borrow().contains_key(&window))
        || GRABS.with(|grabs| grabs.borrow().iter().any(|grab| grab.window == window))
}

pub(crate) fn take_layer_view(window: WindowId) -> Option<Option<Bounds<Pixels>>> {
    SHAPES.with(|shapes| shapes.borrow_mut().remove(&window))
}

pub(crate) fn take_grab_request(window: WindowId) -> Option<Box<dyn FnOnce()>> {
    REQUESTS.with(|requests| requests.borrow_mut().remove(&window))
}

/// Drop `window`'s grab and any grab asked for, without `on_end`: it was
/// hidden or closed.
pub(crate) fn end_grabs(window: WindowId) {
    REQUESTS.with(|requests| requests.borrow_mut().remove(&window));
    let ended: Vec<Grab> = GRABS.with(|grabs| {
        grabs
            .borrow_mut()
            .extract_if(.., |grab| grab.window == window)
            .collect()
    });
    for grab in ended {
        grab.destroy();
    }
}

/// `window` is gone: forget everything about it.
pub(crate) fn forget(window: WindowId) {
    end_grabs(window);
    SHAPES.with(|shapes| shapes.borrow_mut().remove(&window));
    KEEP_SWAPCHAIN.with(|keep| keep.borrow_mut().remove(&window));
}

/// The window a grab popup's surface stands for (keyboard focus on the
/// popup is focus on its window).
pub(crate) fn grab_parent(surface: &ObjectId) -> Option<WaylandWindowStatePtr> {
    GRABS.with(|grabs| {
        grabs
            .borrow()
            .iter()
            .find(|grab| &grab.surface.id() == surface)
            .map(|grab| grab.parent.clone())
    })
}

/// User data of a grab popup's objects.
pub(crate) struct GrabTag;

struct Grab {
    window: WindowId,
    parent: WaylandWindowStatePtr,
    client: WaylandClientStatePtr,
    surface: wl_surface::WlSurface,
    xdg_surface: xdg_surface::XdgSurface,
    popup: xdg_popup::XdgPopup,
    buffer: Option<wl_buffer::WlBuffer>,
    on_end: Option<Box<dyn FnOnce()>>,
}

impl Grab {
    fn destroy(self) {
        // Its surface may have the keyboard (keys reach the sheet through
        // it), and no `leave` will come for a surface we destroy.
        self.client.keyboard_surface_destroyed(&self.surface.id());
        self.popup.destroy();
        self.xdg_surface.destroy();
        self.surface.destroy();
        if let Some(buffer) = self.buffer {
            buffer.destroy();
        }
    }
}

/// Start the grab for `window` (on screen, its layer surface mapped): a 1x1
/// transparent popup at its top-left that takes no input, grabbing with
/// `serial`.
pub(crate) fn start_grab(
    parent: WaylandWindowStatePtr,
    client: WaylandClientStatePtr,
    window: WindowId,
    layer_surface: &zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
    globals: &Globals,
    serial: u32,
    on_end: Box<dyn FnOnce()>,
) {
    end_grabs(window);
    let qh = &globals.qh;
    let surface = globals.compositor.create_surface(qh, GrabTag);
    let region = globals.compositor.create_region(qh, ());
    surface.set_input_region(Some(&region));
    region.destroy();
    let xdg_surface = globals.wm_base.get_xdg_surface(&surface, qh, GrabTag);
    let positioner = globals.wm_base.create_positioner(qh, ());
    positioner.set_size(1, 1);
    positioner.set_anchor_rect(0, 0, 1, 1);
    let popup = xdg_surface.get_popup(None, &positioner, qh, GrabTag);
    positioner.destroy();
    layer_surface.get_popup(&popup);
    popup.grab(&globals.seat, serial);
    surface.commit();
    let buffer = transparent_pixel(globals);
    GRABS.with(|grabs| {
        grabs.borrow_mut().push(Grab {
            window,
            parent,
            client,
            surface,
            xdg_surface,
            popup,
            buffer,
            on_end: Some(on_end),
        })
    });
}

/// A 1x1 fully transparent shm buffer.
fn transparent_pixel(globals: &Globals) -> Option<wl_buffer::WlBuffer> {
    // SAFETY: a plain syscall; the fd is owned from here on.
    let fd = unsafe { libc::memfd_create(c"myshell-grab".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return None;
    }
    // SAFETY: `fd` is a new descriptor nothing else owns.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: `fd` is valid; a new memfd reads as zeros (transparent).
    if unsafe { libc::ftruncate(fd.as_raw_fd(), 4) } != 0 {
        return None;
    }
    let pool = globals.shm.create_pool(fd.as_fd(), 4, &globals.qh, ());
    let buffer = pool.create_buffer(0, 1, 1, 4, wl_shm::Format::Argb8888, &globals.qh, ());
    pool.destroy();
    Some(buffer)
}

impl Dispatch<wl_surface::WlSurface, GrabTag> for WaylandClientStatePtr {
    fn event(
        _: &mut Self,
        _: &wl_surface::WlSurface,
        _: wl_surface::Event,
        _: &GrabTag,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<xdg_surface::XdgSurface, GrabTag> for WaylandClientStatePtr {
    fn event(
        _: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &GrabTag,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let xdg_surface::Event::Configure { serial } = event else {
            return;
        };
        xdg_surface.ack_configure(serial);
        GRABS.with(|grabs| {
            if let Some(grab) = grabs
                .borrow()
                .iter()
                .find(|grab| &grab.xdg_surface == xdg_surface)
            {
                grab.surface.attach(grab.buffer.as_ref(), 0, 0);
                grab.surface.damage_buffer(0, 0, 1, 1);
                grab.surface.commit();
            }
        });
    }
}

impl Dispatch<xdg_popup::XdgPopup, GrabTag> for WaylandClientStatePtr {
    fn event(
        _: &mut Self,
        popup: &xdg_popup::XdgPopup,
        event: xdg_popup::Event,
        _: &GrabTag,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if !matches!(event, xdg_popup::Event::PopupDone) {
            return;
        }
        let ended = GRABS.with(|grabs| {
            let mut grabs = grabs.borrow_mut();
            let pos = grabs.iter().position(|grab| &grab.popup == popup)?;
            Some(grabs.remove(pos))
        });
        if let Some(mut grab) = ended {
            let on_end = grab.on_end.take();
            grab.destroy();
            if let Some(on_end) = on_end {
                on_end();
            }
        }
    }
}

/// With `MYSHELL_FRAMES=causes`, a line as myshell's own frame traces have
/// it, for latency work: `present` when `window`'s frame has been handed to
/// the compositor, `shown` when the compositor's frame callback says it has
/// drawn it.
pub(crate) fn trace_frame(what: &str, window: WindowId) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("MYSHELL_FRAMES").is_some_and(|v| v == "causes")) {
        return;
    }
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    eprintln!("myshell frame {at:.6} {what}: {window:?}");
}
