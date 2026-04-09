//! Visual viewer binary — opens a wgpu window, renders a synthetic point
//! cloud, and applies IPC highlight commands in real-time.
//!
//! Terminal 1:  cargo run
//! Terminal 2:  uv run python scripts/smoke_ipc.py
//!
//! Controls:
//!   Left-drag   — orbit
//!   Right-drag  — pan
//!   Scroll      — zoom
//!   R           — reset camera
//!   Q / Esc     — quit

use std::sync::Arc;

use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

use threecrate_gpu::renderer::{PointCloudRenderer, PointVertex, RenderConfig};
use scene_query_viewer::{
    ipc::receiver::{IpcReceiver, ViewerCommand, DEFAULT_SOCKET_PATH},
    viewer::{overlay::ColorBuffer, render_loop::ViewerLoop, threecrate_buffer::DEFAULT_COLOR},
};

// ---------- geometry ---------------------------------------------------------

const POINT_COUNT: usize = 2_000;
const QUAD_HALF: f32 = 0.02;
const VERT_SIZE: f32 = 16.0;
const VERTS_PER_POINT: usize = 6;

fn make_sphere_vertices() -> Vec<PointVertex> {
    use std::f32::consts::TAU;
    let golden = (1.0 + 5f32.sqrt()) / 2.0;
    let mut verts = Vec::with_capacity(POINT_COUNT * VERTS_PER_POINT);
    for i in 0..POINT_COUNT {
        let t = i as f32 / POINT_COUNT as f32;
        let theta = (1.0 - 2.0 * t).clamp(-1.0, 1.0).acos();
        let phi = TAU * (i as f32 * golden).fract();
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();
        let s = QUAD_HALF;
        let n = [x, y, z];
        let c = DEFAULT_COLOR;
        let v = |dx: f32, dy: f32| PointVertex {
            position: [x + dx, y + dy, z],
            color: c,
            size: VERT_SIZE,
            normal: n,
        };
        let (bl, br, tr, tl) = (v(-s, -s), v(s, -s), v(s, s), v(-s, s));
        verts.extend_from_slice(&[bl, br, tr, bl, tr, tl]);
    }
    verts
}

// ---------- ColorBuffer impl -------------------------------------------------

struct QuadColorBuffer<'a> {
    vertices: &'a mut Vec<PointVertex>,
    default_color: [f32; 3],
}

impl ColorBuffer for QuadColorBuffer<'_> {
    fn set_color(&mut self, primitive_id: u32, color: [f32; 3]) {
        let base = primitive_id as usize * VERTS_PER_POINT;
        for i in 0..VERTS_PER_POINT {
            if let Some(v) = self.vertices.get_mut(base + i) {
                v.color = color;
            }
        }
    }
    fn reset_colors(&mut self, primitive_count: usize) {
        let end = (primitive_count * VERTS_PER_POINT).min(self.vertices.len());
        for v in &mut self.vertices[..end] {
            v.color = self.default_color;
        }
    }
}

// ---------- orbital camera ---------------------------------------------------

struct OrbitalCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: [f32; 3],
    aspect: f32,
}

impl OrbitalCamera {
    fn new(aspect: f32) -> Self {
        Self { yaw: 0.0, pitch: 0.3, distance: 3.0, target: [0.0; 3], aspect }
    }

    fn eye(&self) -> Point3<f32> {
        let (sy, cy) = (self.yaw.sin(), self.yaw.cos());
        let (sp, cp) = (self.pitch.sin(), self.pitch.cos());
        let t = &self.target;
        Point3::new(
            t[0] + self.distance * cp * sy,
            t[1] + self.distance * sp,
            t[2] + self.distance * cp * cy,
        )
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.5, 1.5);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let eye = self.eye();
        let forward = (Point3::from(self.target) - eye).normalize();
        let right = forward.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(&forward).normalize();
        let scale = self.distance * 0.001;
        let shift = right * (-dx * scale) + up * (dy * scale);
        self.target[0] += shift.x;
        self.target[1] += shift.y;
        self.target[2] += shift.z;
    }

    fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.1)).clamp(0.1, 50.0);
    }

    fn reset(&mut self) {
        *self = Self::new(self.aspect);
    }

    fn upload(&self, renderer: &mut PointCloudRenderer<'_>) {
        let eye = self.eye();
        let target = Point3::from(self.target);
        let view = Matrix4::look_at_rh(&eye, &target, &Vector3::new(0.0, 1.0, 0.0));
        let proj =
            Perspective3::new(self.aspect, 45f32.to_radians(), 0.1, 100.0).to_homogeneous();
        renderer.update_camera(view, proj, eye.coords);
    }
}

// ---------- winit app --------------------------------------------------------

/// Sent from the IPC relay task to wake the event loop.
struct IpcWakeup;

struct ViewerApp {
    viewer_loop: ViewerLoop,
    vertices: Vec<PointVertex>,
    camera: OrbitalCamera,
    left_pressed: bool,
    right_pressed: bool,
    last_cursor: Option<(f64, f64)>,
    window: Option<Arc<Window>>,
    renderer: Option<PointCloudRenderer<'static>>,
}

impl ViewerApp {
    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl ApplicationHandler<IpcWakeup> for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("scene-query  |  drag=orbit  right-drag=pan  scroll=zoom  R=reset")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));

        let window = Arc::new(
            event_loop.create_window(window_attrs).expect("create window"),
        );

        // SAFETY: the Arc is kept in self.window for the lifetime of the program.
        let window_ref: &'static Window =
            unsafe { std::mem::transmute::<&Window, &'static Window>(window.as_ref()) };

        let mut renderer =
            match pollster::block_on(PointCloudRenderer::new(window_ref, RenderConfig::default())) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("GPU init failed: {e}");
                    eprintln!("This viewer requires a GPU (Metal on macOS).");
                    eprintln!("Run 'cargo test' and 'uv run python scripts/smoke_ipc.py' (with cargo run in headless mode) to test without a GPU.");
                    event_loop.exit();
                    return;
                }
            };

        let size = window.inner_size();
        self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        self.camera.upload(&mut renderer);

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.request_redraw();

        println!("────────────────────────────────────────────────");
        println!(" scene-query highlight viewer");
        println!(" drag=orbit  right-drag=pan  scroll=zoom  R=reset");
        println!(" Highlights: uv run python scripts/smoke_ipc.py");
        println!("────────────────────────────────────────────────");
    }

    /// Called when the IPC relay task sends a wakeup — request one redraw.
    fn user_event(&mut self, _: &ActiveEventLoop, _: IpcWakeup) {
        self.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key: Key::Named(NamedKey::Escape),
                    state: ElementState::Pressed, ..
                }, ..
            } => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key: Key::Character(ref c),
                    state: ElementState::Pressed, ..
                }, ..
            } => match c.as_str() {
                "q" | "Q" => event_loop.exit(),
                "r" | "R" => {
                    self.camera.reset();
                    self.request_redraw();
                }
                _ => {}
            },

            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => self.left_pressed = state == ElementState::Pressed,
                MouseButton::Right => self.right_pressed = state == ElementState::Pressed,
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x, position.y);
                if let Some(last) = self.last_cursor {
                    let (dx, dy) = ((pos.0 - last.0) as f32, (pos.1 - last.1) as f32);
                    if self.left_pressed {
                        self.camera.orbit(dx, dy);
                        self.request_redraw();
                    } else if self.right_pressed {
                        self.camera.pan(dx, dy);
                        self.request_redraw();
                    }
                }
                self.last_cursor = Some(pos);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                self.camera.zoom(scroll);
                self.request_redraw();
            }

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size);
                }
                self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
                self.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let Some(renderer) = &mut self.renderer else { return };

                self.camera.upload(renderer);

                let vl = &mut self.viewer_loop;
                let verts = &mut self.vertices;
                let mut buf = QuadColorBuffer { vertices: verts, default_color: DEFAULT_COLOR };
                vl.frame(&mut buf, POINT_COUNT);

                if let Err(e) = renderer.render(&self.vertices) {
                    eprintln!("render error: {e}");
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Block until the next OS event — no polling, no GPU spin.
        event_loop.set_control_flow(ControlFlow::Wait);
    }
}

// ---------- main -------------------------------------------------------------

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let (cmd_tx, cmd_rx) = mpsc::channel::<ViewerCommand>(64);

    // Build the event loop with a custom user-event type so the IPC relay can
    // wake it without polling.
    let event_loop = EventLoop::<IpcWakeup>::with_user_event()
        .build()
        .expect("event loop");
    let proxy = event_loop.create_proxy();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.spawn(async move {
        // IPC receiver writes commands to ipc_tx.
        let (ipc_tx, mut ipc_rx) = mpsc::channel::<ViewerCommand>(64);
        tokio::spawn(async move {
            if let Err(e) = IpcReceiver::new(DEFAULT_SOCKET_PATH, ipc_tx).run().await {
                eprintln!("IPC receiver error: {e}");
            }
        });

        // Relay: forward commands to ViewerLoop channel, then wake the event loop.
        while let Some(cmd) = ipc_rx.recv().await {
            if cmd_tx.send(cmd).await.is_err() {
                break;
            }
            // Wake the winit event loop so it calls RedrawRequested.
            let _ = proxy.send_event(IpcWakeup);
        }
    });

    let mut app = ViewerApp {
        viewer_loop: ViewerLoop::new(cmd_rx),
        vertices: make_sphere_vertices(),
        camera: OrbitalCamera::new(1280.0 / 720.0),
        left_pressed: false,
        right_pressed: false,
        last_cursor: None,
        window: None,
        renderer: None,
    };

    event_loop.run_app(&mut app).expect("event loop run");
}
