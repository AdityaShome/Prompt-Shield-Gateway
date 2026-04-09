use prompt_shield_gateway::gateway;

#[tokio::main]
async fn main() {
    gateway::run_from_env().await;
}
