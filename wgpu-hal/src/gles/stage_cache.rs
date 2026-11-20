use alloc::{borrow::Cow, collections::VecDeque, string::String};

const MAX_STAGES: usize = 64;
const MAX_SOURCE_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub(super) struct Key {
    stage: naga::ShaderStage,
    source: String,
}

impl Key {
    pub(super) fn new(stage: naga::ShaderStage, source: Cow<'_, str>) -> Option<Self> {
        if source.len() > MAX_SOURCE_BYTES {
            return None;
        }
        let source = match source {
            Cow::Owned(text) => text,
            Cow::Borrowed(text) => {
                let mut copy = String::new();
                copy.try_reserve_exact(text.len()).ok()?;
                copy.push_str(text);
                copy
            }
        };
        (source.capacity() <= MAX_SOURCE_BYTES).then_some(Self { stage, source })
    }
}

#[derive(Debug, Default)]
pub(super) struct StageCache {
    stages: VecDeque<(Key, glow::Shader)>,
}

impl StageCache {
    pub(super) fn get(&self, stage: naga::ShaderStage, source: &str) -> Option<glow::Shader> {
        self.stages
            .iter()
            .rev()
            .find_map(|(key, raw)| (key.stage == stage && key.source == source).then_some(*raw))
    }

    /// Only stages from a successfully linked program reach the cache.
    pub(super) fn insert(&mut self, key: Key, raw: glow::Shader) -> Option<glow::Shader> {
        if self.get(key.stage, &key.source).is_some() {
            return Some(raw);
        }
        let evicted = if self.stages.len() == MAX_STAGES {
            self.stages.pop_front().map(|(_, raw)| raw)
        } else {
            if self.stages.try_reserve(1).is_err() {
                return Some(raw);
            }
            None
        };
        self.stages.push_back((key, raw));
        evicted
    }

    pub(super) fn drain(&mut self) -> impl Iterator<Item = glow::Shader> + '_ {
        self.stages.drain(..).map(|(_, raw)| raw)
    }
}

#[cfg(all(test, native))]
mod tests {
    use super::*;
    use alloc::format;
    use core::num::NonZeroU32;
    use naga::ShaderStage::{Fragment, Vertex};

    fn shader(id: u32) -> glow::Shader {
        glow::NativeShader(NonZeroU32::new(id).unwrap())
    }

    #[test]
    fn exact_source_and_stage_identify_a_linked_shader() {
        let mut cache = StageCache::default();
        let key = |stage| Key::new(stage, Cow::Borrowed("same source")).unwrap();
        assert_eq!(cache.insert(key(Vertex), shader(1)), None);
        assert_eq!(cache.insert(key(Fragment), shader(2)), None);
        assert_eq!(cache.insert(key(Vertex), shader(3)), Some(shader(3)));
        assert_eq!(cache.get(Vertex, "same source"), Some(shader(1)));
        assert_eq!(cache.get(Fragment, "same source"), Some(shader(2)));
        assert_eq!(cache.get(Vertex, "same sourcf"), None);
        assert_eq!(
            cache.drain().collect::<alloc::vec::Vec<_>>(),
            [shader(1), shader(2)]
        );
        assert_eq!(cache.get(Vertex, "same source"), None);
    }

    #[test]
    fn short_source_with_an_oversized_allocation_stays_uncached() {
        let mut text = String::with_capacity(MAX_SOURCE_BYTES + 1);
        text.push('x');
        assert!(Key::new(Vertex, Cow::Owned(text)).is_none());
        let text = "x".repeat(MAX_SOURCE_BYTES);
        assert!(Key::new(Vertex, Cow::Borrowed(&text)).is_some());
    }

    #[test]
    fn full_cache_releases_the_oldest_stage_and_large_sources_stay_uncached() {
        let mut cache = StageCache::default();
        assert!(Key::new(Vertex, Cow::Owned("x".repeat(MAX_SOURCE_BYTES + 1))).is_none());
        for raw in 1..=MAX_STAGES as u32 {
            let key = Key::new(Fragment, Cow::Owned(format!("{raw}"))).unwrap();
            assert_eq!(cache.insert(key, shader(raw)), None);
        }
        let key = Key::new(Vertex, Cow::Borrowed("new")).unwrap();
        assert_eq!(cache.insert(key, shader(99)), Some(shader(1)));
        assert_eq!(cache.get(Fragment, "1"), None);
        assert_eq!(cache.get(Vertex, "new"), Some(shader(99)));
        assert_eq!(cache.drain().count(), MAX_STAGES);
    }
}
