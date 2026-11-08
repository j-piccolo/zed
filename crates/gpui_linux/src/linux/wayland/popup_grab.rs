use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};

use wayland_backend::client::ObjectId;
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle,
    protocol::{wl_buffer, wl_shm, wl_surface},
};
use wayland_protocols::xdg::shell::client::{xdg_popup, xdg_surface};
use wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1;

use super::client::{Globals, WaylandClientStatePtr, get_window};

/// User data of the objects behind a [`PopupGrab`]: the surface id of the window it grabs for.
pub(crate) struct PopupGrabParent(ObjectId);

/// An input grab held for a layer-shell window by an invisible 1x1 `xdg_popup` child of it.
///
/// A layer surface can't take a grab itself, but its popups can, and the compositor treats the
/// grab as it treats a grabbing popup's: keyboard focus goes to the popup, which stands for its
/// window, and a press on another client's surface ends the grab and still reaches that surface.
pub(crate) struct PopupGrab {
    surface: wl_surface::WlSurface,
    xdg_surface: xdg_surface::XdgSurface,
    popup: xdg_popup::XdgPopup,
    buffer: Option<wl_buffer::WlBuffer>,
    on_end: Option<Box<dyn FnOnce()>>,
}

impl PopupGrab {
    /// Takes a grab with `serial` for the mapped layer surface of the window whose surface is
    /// `parent`, through a transparent popup at its top-left that takes no input.
    pub(crate) fn start(
        globals: &Globals,
        layer_surface: &zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        parent: ObjectId,
        serial: u32,
        on_end: Box<dyn FnOnce()>,
    ) -> Self {
        let qh = &globals.qh;
        let surface = globals
            .compositor
            .create_surface(qh, PopupGrabParent(parent.clone()));
        let input_region = globals.compositor.create_region(qh, ());
        surface.set_input_region(Some(&input_region));
        input_region.destroy();
        let xdg_surface =
            globals
                .wm_base
                .get_xdg_surface(&surface, qh, PopupGrabParent(parent.clone()));
        let positioner = globals.wm_base.create_positioner(qh, ());
        positioner.set_size(1, 1);
        positioner.set_anchor_rect(0, 0, 1, 1);
        let popup = xdg_surface.get_popup(None, &positioner, qh, PopupGrabParent(parent));
        positioner.destroy();
        layer_surface.get_popup(&popup);
        popup.grab(&globals.seat, serial);
        surface.commit();
        Self {
            surface,
            xdg_surface,
            popup,
            buffer: transparent_pixel(globals),
            on_end: Some(on_end),
        }
    }

    pub(crate) fn surface_id(&self) -> ObjectId {
        self.surface.id()
    }

    pub(crate) fn holds(&self, popup: &xdg_popup::XdgPopup) -> bool {
        &self.popup == popup
    }

    pub(crate) fn take_on_end(&mut self) -> Option<Box<dyn FnOnce()>> {
        self.on_end.take()
    }

    fn map(&self) {
        self.surface.attach(self.buffer.as_ref(), 0, 0);
        self.surface.damage_buffer(0, 0, 1, 1);
        self.surface.commit();
    }

    /// Ends the grab without running its `on_end`.
    pub(crate) fn destroy(self, client: &WaylandClientStatePtr) {
        // No `wl_keyboard.leave` comes for a surface the client destroys.
        client.popup_grab_ended(&self.surface.id());
        self.popup.destroy();
        self.xdg_surface.destroy();
        self.surface.destroy();
        if let Some(buffer) = self.buffer {
            buffer.destroy();
        }
    }
}

fn transparent_pixel(globals: &Globals) -> Option<wl_buffer::WlBuffer> {
    // SAFETY: a plain syscall, whose descriptor is owned from here on.
    let fd = unsafe { libc::memfd_create(c"gpui-popup-grab".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        log::warn!(
            "failed to create a buffer for a popup grab: {}",
            std::io::Error::last_os_error()
        );
        return None;
    }
    // SAFETY: `fd` is a new descriptor that nothing else owns.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: `fd` is valid, and a new memfd reads as zeros, which is transparent.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), 4) } != 0 {
        log::warn!(
            "failed to size a buffer for a popup grab: {}",
            std::io::Error::last_os_error()
        );
        return None;
    }
    let pool = globals.shm.create_pool(fd.as_fd(), 4, &globals.qh, ());
    let buffer = pool.create_buffer(0, 1, 1, 4, wl_shm::Format::Argb8888, &globals.qh, ());
    pool.destroy();
    Some(buffer)
}

impl Dispatch<wl_surface::WlSurface, PopupGrabParent> for WaylandClientStatePtr {
    fn event(
        _: &mut Self,
        _: &wl_surface::WlSurface,
        _: wl_surface::Event,
        _: &PopupGrabParent,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<xdg_surface::XdgSurface, PopupGrabParent> for WaylandClientStatePtr {
    fn event(
        this: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        parent: &PopupGrabParent,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let xdg_surface::Event::Configure { serial } = event else {
            return;
        };
        xdg_surface.ack_configure(serial);
        let client = this.get_client();
        let mut state = client.borrow_mut();
        let Some(window) = get_window(&mut state, &parent.0) else {
            return;
        };
        drop(state);
        window.with_popup_grab(|grab| {
            if &grab.xdg_surface == xdg_surface {
                grab.map();
            }
        });
    }
}

impl Dispatch<xdg_popup::XdgPopup, PopupGrabParent> for WaylandClientStatePtr {
    fn event(
        this: &mut Self,
        popup: &xdg_popup::XdgPopup,
        event: xdg_popup::Event,
        parent: &PopupGrabParent,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if !matches!(event, xdg_popup::Event::PopupDone) {
            return;
        }
        let client = this.get_client();
        let mut state = client.borrow_mut();
        let Some(window) = get_window(&mut state, &parent.0) else {
            return;
        };
        drop(state);
        window.popup_grab_done(popup);
    }
}
