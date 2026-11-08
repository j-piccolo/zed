#![cfg_attr(target_family = "wasm", no_main)]

//! Locks the session for five seconds, or until a key is pressed, with a lock surface on every
//! display. Needs a compositor with `ext-session-lock-v1`.

fn run_example() {
    #[cfg(all(target_os = "linux", feature = "wayland"))]
    example::main();

    #[cfg(not(all(target_os = "linux", feature = "wayland")))]
    panic!("This example requires the `wayland` feature and a linux system.");
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}

#[cfg(all(target_os = "linux", feature = "wayland"))]
mod example {
    use std::time::Duration;

    use gpui::{
        App, Context, FocusHandle, FontWeight, KeyDownEvent, Window, WindowKind, WindowOptions,
        div, prelude::*, rems, rgb, session_lock::SessionLockEvent, white,
    };
    use gpui_platform::application;

    struct LockScreen {
        focus_handle: FocusHandle,
    }

    impl Render for LockScreen {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            window.focus(&self.focus_handle, cx);
            div()
                .track_focus(&self.focus_handle)
                .on_key_down(|_: &KeyDownEvent, _, cx| cx.unlock_session())
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .bg(rgb(0x1e1e2e))
                .text_color(white())
                .text_size(rems(3.))
                .font_weight(FontWeight::BOLD)
                .child("Locked: press any key")
        }
    }

    pub fn main() {
        application().run(|cx: &mut App| {
            cx.lock_session(|event, cx| match event {
                SessionLockEvent::Locked => println!("locked"),
                // Exiting before the compositor has taken the unlock may leave the session locked.
                SessionLockEvent::Unlocked | SessionLockEvent::Finished => cx.quit(),
            })
            .expect("the compositor should support ext-session-lock-v1");

            for display in cx.displays() {
                cx.open_window(
                    WindowOptions {
                        titlebar: None,
                        display_id: Some(display.id()),
                        kind: WindowKind::LockSurface,
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|cx| LockScreen {
                            focus_handle: cx.focus_handle(),
                        })
                    },
                )
                .unwrap();
            }

            cx.spawn(async move |cx| {
                cx.background_executor().timer(Duration::from_secs(5)).await;
                cx.update(|cx| cx.unlock_session());
            })
            .detach();
        });
    }
}
