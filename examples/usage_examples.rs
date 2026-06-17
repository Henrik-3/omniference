//! Simple example showing the different ways to use Omniference
//!
//! This is a conceptual example. For working examples, see:
//! - library_usage.rs - Basic library usage
//! - embedded_axum.rs - Embed in existing Axum app
//! - standalone_server.rs - Run as standalone server
//! - discord_bot.rs - Discord bot integration

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	println!("🚀 Omniference Usage Examples");
	println!("============================");

	println!("\n📚 Available Examples:");
	println!("   • library_usage - Basic library usage");
	println!("   • embedded_axum - Embed in existing Axum app");
	println!("   • standalone_server - Run as standalone server");
	println!("   • discord_bot - Discord bot integration (requires --features discord)");

	println!("\n🔧 Running examples:");
	println!("   cargo run --example library_usage");
	println!("   cargo run --example embedded_axum");
	println!("   cargo run --example standalone_server");
	println!("   cargo run --example discord_bot --features discord");

	println!("\n✨ Check the individual example files for detailed implementation!");

	Ok(())
}
