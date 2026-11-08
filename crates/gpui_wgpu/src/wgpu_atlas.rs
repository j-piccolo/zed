use anyhow::{Context as _, Result};
use collections::{FxHashMap, FxHashSet};
use etagere::{BucketedAtlasAllocator, size2};
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, DevicePixels, PlatformAtlas, Point, Size,
};
use parking_lot::Mutex;
use std::{borrow::Cow, ops, sync::Arc};

use crate::WgpuContext;

/// The tiles and textures on one device, shared by every window drawing with it, so that a
/// glyph or icon is rasterized and uploaded once rather than once per window.
///
/// Glyphs and SVGs are kept for the atlas's life, as they are in a single window's atlas.
/// Image tiles are counted by the windows holding them, and freed once none does: a window
/// only gives up an image it removed itself, or held when it closed.
///
/// [`WgpuAtlas::clear`] clears the tiles of every window. Each window notices it through
/// [`WgpuAtlas::take_cleared`] and repaints in full, since its scene may name tiles that are
/// gone.
pub(crate) struct SharedAtlas {
    state: Mutex<SharedAtlasState>,
}

struct SharedAtlasState {
    tiles: AtlasState<WgpuAtlasTextures>,
    image_holder_counts: FxHashMap<AtlasKey, u32>,
    /// Bumped each time the atlas is cleared.
    epoch: u64,
}

impl SharedAtlas {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_texture_format: wgpu::TextureFormat,
    ) -> Arc<Self> {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        Arc::new(Self {
            state: Mutex::new(SharedAtlasState {
                tiles: AtlasState::new(WgpuAtlasTextures {
                    device,
                    queue,
                    max_texture_size,
                    color_texture_format,
                    storage: WgpuAtlasStorage::default(),
                    pending_uploads: Vec::new(),
                    next_texture_generation: 0,
                }),
                image_holder_counts: FxHashMap::default(),
                epoch: 0,
            }),
        })
    }
}

impl SharedAtlasState {
    fn release_image(&mut self, key: &AtlasKey) {
        let Some(count) = self.image_holder_counts.get_mut(key) else {
            return;
        };
        *count -= 1;
        if *count == 0 {
            self.image_holder_counts.remove(key);
            self.tiles.remove(key);
        }
    }
}

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    size2(size.width.0, size.height.0)
}

fn etagere_point_to_device(point: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels(point.x),
        y: DevicePixels(point.y),
    }
}

/// A window's handle onto the [`SharedAtlas`] of its device.
pub struct WgpuAtlas(Mutex<WindowAtlas>);

struct WindowAtlas {
    shared: Arc<SharedAtlas>,
    /// Image keys this window inserted and has not removed.
    images: FxHashSet<AtlasKey>,
    /// The shared epoch `images` belongs to.
    images_epoch: u64,
    /// The shared epoch this window last drew in.
    drawn_epoch: u64,
}

impl WindowAtlas {
    fn join(shared: Arc<SharedAtlas>) -> Self {
        let epoch = shared.state.lock().epoch;
        WindowAtlas {
            shared,
            images: FxHashSet::default(),
            images_epoch: epoch,
            drawn_epoch: epoch,
        }
    }

    fn release_images(&mut self) {
        let mut state = self.shared.state.lock();
        if self.images_epoch == state.epoch {
            for key in self.images.drain() {
                state.release_image(&key);
            }
        }
        self.images.clear();
        self.images_epoch = state.epoch;
    }
}

struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

struct WgpuAtlasTextures {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_texture_size: u32,
    color_texture_format: wgpu::TextureFormat,
    storage: WgpuAtlasStorage,
    pending_uploads: Vec<PendingUpload>,
    next_texture_generation: u64,
}

pub struct WgpuTextureInfo {
    pub view: wgpu::TextureView,
    /// Distinguishes new textures that reuse an [`AtlasTextureId`].
    pub generation: u64,
}

impl WgpuAtlas {
    /// An atlas of its own, shared with no other window.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_texture_format: wgpu::TextureFormat,
    ) -> Self {
        Self::join(SharedAtlas::new(device, queue, color_texture_format))
    }

    /// A handle onto the atlas shared by the windows drawing with `context`.
    pub fn from_context(context: &WgpuContext) -> Self {
        Self::join(context.shared_atlas())
    }

    fn join(shared: Arc<SharedAtlas>) -> Self {
        WgpuAtlas(Mutex::new(WindowAtlas::join(shared)))
    }

    /// Writes every window's queued uploads, so all tiles handed out so far
    /// are on the queue ahead of this frame.
    pub fn before_frame(&self) {
        let window = self.0.lock();
        window.shared.state.lock().tiles.backend.flush_uploads();
    }

    /// Returns the view backing `id`, or `None` once every tile in it has been
    /// removed. A scene can still reference such a texture when a cached view
    /// replays a paint from before the image was dropped, so callers must skip
    /// those sprites rather than assume the texture exists.
    pub fn get_texture_info(&self, id: AtlasTextureId) -> Option<WgpuTextureInfo> {
        let window = self.0.lock();
        let state = window.shared.state.lock();
        let texture = state.tiles.backend.storage.get(id)?;
        Some(WgpuTextureInfo {
            view: texture.view.clone(),
            generation: texture.generation,
        })
    }

    /// Clears all cached textures and tiles, forcing them to be recreated.
    /// Use this for incremental recovery when the device is still valid.
    ///
    /// The tiles are every window's: the others see it in [`Self::take_cleared`].
    pub fn clear(&self) {
        let mut window = self.0.lock();
        let epoch = {
            let mut state = window.shared.state.lock();
            state.tiles.clear(|textures| {
                textures.storage = WgpuAtlasStorage::default();
                textures.pending_uploads.clear();
            });
            state.image_holder_counts.clear();
            state.epoch += 1;
            state.epoch
        };
        window.images.clear();
        window.images_epoch = epoch;
        window.drawn_epoch = epoch;
    }

    /// Whether another window cleared the atlas since this one last asked.
    /// Its scene may then name tiles that are gone, so it must repaint in
    /// full before drawing.
    pub fn take_cleared(&self) -> bool {
        let mut window = self.0.lock();
        let epoch = window.shared.state.lock().epoch;
        std::mem::replace(&mut window.drawn_epoch, epoch) != epoch
    }

    /// Handles device lost by moving to the atlas shared on `context`'s device.
    /// Textures there are made lazily as this window repaints.
    pub fn handle_device_lost(&self, context: &WgpuContext) {
        let mut window = self.0.lock();
        window.release_images();
        *window = WindowAtlas::join(context.shared_atlas());
    }
}

impl Drop for WgpuAtlas {
    fn drop(&mut self) {
        self.0.get_mut().release_images();
    }
}

impl PlatformAtlas for WgpuAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut window = self.0.lock();
        let WindowAtlas {
            shared,
            images,
            images_epoch,
            ..
        } = &mut *window;
        let mut state = shared.state.lock();
        if !matches!(key, AtlasKey::Image(_)) {
            return state.tiles.get_or_insert_with(key, build);
        }

        if *images_epoch != state.epoch {
            images.clear();
            *images_epoch = state.epoch;
        }
        let tile = state.tiles.get_or_insert_with(key.clone(), build)?;
        if tile.is_some() && !images.contains(&key) {
            *state.image_holder_counts.entry(key.clone()).or_default() += 1;
            images.insert(key);
        }
        Ok(tile)
    }

    /// Releases this window's hold on an image. Glyphs and SVGs are shared
    /// by every window and kept for the atlas's life.
    fn remove(&self, key: &AtlasKey) {
        let mut window = self.0.lock();
        let WindowAtlas {
            shared,
            images,
            images_epoch,
            ..
        } = &mut *window;
        let mut state = shared.state.lock();
        if *images_epoch == state.epoch && images.remove(key) {
            state.release_image(key);
        }
    }
}

impl AtlasBackend for WgpuAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let tile = self.allocate(size, kind).context("failed to allocate")?;
        self.upload_texture(tile.texture_id, tile.bounds, bytes);
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;
        let Some(texture_slot) = self.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                self.pending_uploads
                    .retain(|upload| upload.id != texture.id);
                self.storage[id.kind]
                    .free_list
                    .push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl WgpuAtlasTextures {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        let max_texture_size = self.max_texture_size as i32;
        let max_atlas_size = Size {
            width: DevicePixels(max_texture_size),
            height: DevicePixels(max_texture_size),
        };

        let size = min_size.min(&max_atlas_size).max(&DEFAULT_ATLAS_SIZE);
        let format = match kind {
            AtlasTextureKind::Monochrome => wgpu::TextureFormat::R8Unorm,
            AtlasTextureKind::Subpixel | AtlasTextureKind::Polychrome => self.color_texture_format,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();
        let generation = self.next_texture_generation;
        self.next_texture_generation = self.next_texture_generation.wrapping_add(1);

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            generation,
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list
                .textures
                .get_mut(ix)
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list
                .textures
                .last_mut()
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        }
    }

    fn upload_texture(&mut self, id: AtlasTextureId, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let data = self
            .storage
            .get(id)
            .map(|texture| swizzle_upload_data(bytes, texture.format))
            .unwrap_or_else(|| bytes.to_vec());

        self.pending_uploads
            .push(PendingUpload { id, bounds, data });
    }

    fn flush_uploads(&mut self) {
        for upload in self.pending_uploads.drain(..) {
            let Some(texture) = self.storage.get(upload.id) else {
                continue;
            };
            let bytes_per_pixel = texture.bytes_per_pixel();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.0 as u32,
                        y: upload.bounds.origin.y.0 as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bounds.size.width.0 as u32 * bytes_per_pixel as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.0 as u32,
                    height: upload.bounds.size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    subpixel_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl WgpuAtlasStorage {
    fn get(&self, id: AtlasTextureId) -> Option<&WgpuAtlasTexture> {
        self[id.kind]
            .textures
            .get(id.index as usize)
            .and_then(|t| t.as_ref())
    }
}

struct WgpuAtlasTexture {
    id: AtlasTextureId,
    generation: u64,
    allocator: BucketedAtlasAllocator,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
}

impl WgpuAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn swizzle_upload_data(bytes: &[u8], format: wgpu::TextureFormat) -> Vec<u8> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => {
            let mut data = bytes.to_vec();
            for pixel in data.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            data
        }
        _ => bytes.to_vec(),
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use gpui::block_on;
    use gpui::{ImageId, RenderImageParams};
    use std::sync::Arc;

    fn test_device_and_queue() -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("wgpu_atlas_test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults()
                        .using_resolution(adapter.limits())
                        .using_alignment(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    default_queue: wgpu::QueueDescriptor::default(),
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?;
            Ok((Arc::new(device), Arc::new(queue)))
        })
    }

    #[test]
    fn before_frame_skips_uploads_for_removed_texture() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;

        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let size = Size {
            width: DevicePixels(1),
            height: DevicePixels(1),
        };
        let mut build = || Ok(Some((size, Cow::Owned(vec![0, 0, 0, 255]))));

        // Regression test: before the fix, this panicked in flush_uploads
        atlas
            .get_or_insert_with(key.clone(), &mut build)?
            .expect("tile should be created");
        atlas.remove(&key);
        atlas.before_frame();
        Ok(())
    }

    #[test]
    fn remove_deallocates_tile_space_for_reuse() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let make_key = |image_id: usize| {
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(image_id),
                frame_index: 0,
            })
        };
        let insert = |key: AtlasKey, size: Size<DevicePixels>| {
            let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let keeper_key = make_key(1);
        let big_key_a = make_key(2);
        let big_key_b = make_key(3);

        let keeper_tile = insert(keeper_key, small);
        let tile_a = insert(big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert(big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
        Ok(())
    }

    #[test]
    fn reused_texture_id_has_new_generation() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let size = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };
        let make_key = |image_id| {
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(image_id),
                frame_index: 0,
            })
        };
        let insert = |key: AtlasKey| {
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((
                        size,
                        Cow::Owned(vec![0; size.width.0 as usize * size.height.0 as usize * 4]),
                    )))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let first_key = make_key(1);
        let first_tile = insert(first_key.clone());
        let first_generation = atlas
            .get_texture_info(first_tile.texture_id)
            .context("first texture should exist")?
            .generation;
        atlas.remove(&first_key);
        assert!(atlas.get_texture_info(first_tile.texture_id).is_none());

        let second_tile = insert(make_key(2));
        let second_generation = atlas
            .get_texture_info(second_tile.texture_id)
            .context("second texture should exist")?
            .generation;

        assert_eq!(second_tile.texture_id, first_tile.texture_id);
        assert_ne!(second_generation, first_generation);
        Ok(())
    }

    #[test]
    fn swizzle_upload_data_preserves_bgra_uploads() {
        let input = vec![0x10, 0x20, 0x30, 0x40];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Bgra8Unorm),
            input
        );
    }

    #[test]
    fn swizzle_upload_data_converts_bgra_to_rgba() {
        let input = vec![0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Rgba8Unorm),
            vec![0x30, 0x20, 0x10, 0x40, 0xCC, 0xBB, 0xAA, 0xDD]
        );
    }

    fn image_key(image_id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(image_id),
            frame_index: 0,
        })
    }

    fn pixel() -> Result<Option<(Size<DevicePixels>, Cow<'static, [u8]>)>> {
        let size = Size {
            width: DevicePixels(1),
            height: DevicePixels(1),
        };
        Ok(Some((size, Cow::Owned(vec![0, 0, 0, 255]))))
    }

    /// Atlases for `COUNT` windows drawing with one device.
    fn window_atlases<const COUNT: usize>(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> [WgpuAtlas; COUNT] {
        let shared = SharedAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        std::array::from_fn(|_| WgpuAtlas::join(shared.clone()))
    }

    fn pending_uploads(atlas: &WgpuAtlas) -> usize {
        let window = atlas.0.lock();
        let state = window.shared.state.lock();
        state.tiles.backend.pending_uploads.len()
    }

    #[test]
    fn windows_sharing_an_atlas_share_tiles() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let [a, b] = window_atlases(device, queue);
        let key = image_key(1);

        let mut builds = 0;
        let tile_a = a
            .get_or_insert_with(key.clone(), &mut || {
                builds += 1;
                pixel()
            })?
            .expect("tile");
        let tile_b = b
            .get_or_insert_with(key, &mut || {
                builds += 1;
                pixel()
            })?
            .expect("tile");
        assert_eq!(builds, 1);
        assert_eq!(tile_a, tile_b);

        // The upload is queued once, and the other window's frame writes it.
        assert_eq!(pending_uploads(&a), 1);
        b.before_frame();
        assert_eq!(pending_uploads(&a), 0);
        Ok(())
    }

    #[test]
    fn image_tile_lives_until_its_last_holder_removes_it() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let [a, b, c] = window_atlases(device, queue);
        let key = image_key(1);

        let tile = a
            .get_or_insert_with(key.clone(), &mut pixel)?
            .expect("tile");
        b.get_or_insert_with(key.clone(), &mut pixel)?;

        // A window that never held it cannot free it.
        c.remove(&key);
        a.remove(&key);
        a.remove(&key);
        assert!(b.get_texture_info(tile.texture_id).is_some());

        b.remove(&key);
        assert!(b.get_texture_info(tile.texture_id).is_none());
        Ok(())
    }

    #[test]
    fn closing_window_releases_its_images() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let [a, b, bystander] = window_atlases(device, queue);
        let key = image_key(1);

        let tile = a
            .get_or_insert_with(key.clone(), &mut pixel)?
            .expect("tile");
        b.get_or_insert_with(key, &mut pixel)?;
        drop(b);
        assert!(bystander.get_texture_info(tile.texture_id).is_some());
        drop(a);
        assert!(bystander.get_texture_info(tile.texture_id).is_none());
        Ok(())
    }

    #[test]
    fn clear_is_seen_by_every_window_and_forgets_old_holds() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let [a, b, c] = window_atlases(device, queue);
        let key = image_key(1);

        a.get_or_insert_with(key.clone(), &mut pixel)?;
        b.get_or_insert_with(key.clone(), &mut pixel)?;
        a.clear();
        assert!(!a.take_cleared());
        assert!(b.take_cleared());
        assert!(!b.take_cleared());

        // b's hold is from before the clear: removing it must not free c's.
        let tile = c
            .get_or_insert_with(key.clone(), &mut pixel)?
            .expect("tile");
        b.remove(&key);
        drop(b);
        assert!(c.get_texture_info(tile.texture_id).is_some());
        Ok(())
    }

    #[test]
    fn separate_atlases_share_nothing() -> anyhow::Result<()> {
        let (device_a, queue_a) = test_device_and_queue()?;
        let (device_b, queue_b) = test_device_and_queue()?;
        let a = WgpuAtlas::new(device_a, queue_a, wgpu::TextureFormat::Bgra8Unorm);
        let b = WgpuAtlas::new(device_b, queue_b, wgpu::TextureFormat::Bgra8Unorm);
        let tile = a
            .get_or_insert_with(image_key(1), &mut pixel)?
            .expect("tile");
        assert!(b.get_texture_info(tile.texture_id).is_none());
        Ok(())
    }
}
