use crate::core::{Action, CoreEngine, ScanChannel, ScanDecision, ScanRequest};

#[derive(Clone)]
pub struct EmbeddedMiddleware {
    engine: CoreEngine,
}

impl EmbeddedMiddleware {
    pub fn new(engine: CoreEngine) -> Self {
        Self { engine }
    }

    pub fn default() -> Self {
        Self::new(CoreEngine::default_with_local_model())
    }

    pub async fn scan_input(&self, prompt: impl Into<String>) -> EmbeddedScanResult {
        self.scan(prompt.into(), ScanChannel::Input).await
    }

    pub async fn scan_context(&self, context: impl Into<String>) -> EmbeddedScanResult {
        self.scan(context.into(), ScanChannel::Context).await
    }

    pub async fn scan_output(&self, response: impl Into<String>) -> EmbeddedScanResult {
        self.scan(response.into(), ScanChannel::Output).await
    }

    async fn scan(&self, content: String, channel: ScanChannel) -> EmbeddedScanResult {
        let decision = self
            .engine
            .scan(&ScanRequest {
                content,
                metadata: serde_json::json!({}),
                channel,
            })
            .await;

        EmbeddedScanResult::from(decision)
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddedScanResult {
    pub action: Action,
    pub risk_score: f32,
    pub sanitized_content: String,
}

impl From<ScanDecision> for EmbeddedScanResult {
    fn from(value: ScanDecision) -> Self {
        Self {
            action: value.action,
            risk_score: value.risk_score,
            sanitized_content: value.sanitized_content,
        }
    }
}
